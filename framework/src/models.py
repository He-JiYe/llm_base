"""模型模块：Decoder-only Transformer。

实现 GPT-2 风格的 Decoder-only Transformer 架构。
包含：Token嵌入、位置编码、多头因果自注意力、FFN、TransformerBlock、主体模型。
"""

import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ===== 位置编码 =====

class SinusoidalPositionalEmbedding(nn.Module):
    """正弦位置编码（不使用可学习参数）。

    使用正弦/余弦函数生成位置编码，支持任意序列长度。
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, hidden_dim: int, max_seq_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_seq_len, hidden_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float)
            * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class RotaryPositionEmbedding(nn.Module):
    """旋转位置编码 RoPE。

    不直接加在输入上，而是对 Q 和 K 进行旋转，
    使注意力机制天然感知相对位置。
    参考: RoFormer (Su et al., 2021)

    Args:
        head_dim: 每个注意力头的维度
        max_seq_len: 支持的最大序列长度
        base: 频率基数（默认 10000）
    """

    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        # 计算频率：theta_i = base^(-2i/d)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 预计算 cos/sin 表格
        self._update_cached(max_seq_len)

    def _update_cached(self, seq_len: int):
        """更新或扩展 cos/sin 缓存到指定长度。"""
        if hasattr(self, "cos_cached") and self.cos_cached.size(-2) >= seq_len:
            return
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        # freqs: (seq_len, head_dim/2)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # 拼接两半: (seq_len, head_dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """旋转后半维度: (x1, x2) -> (-x2, x1)。"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple:
        """对 Q 和 K 应用旋转位置编码。

        Args:
            q: (batch, num_heads, seq_len, head_dim)
            k: (batch, num_heads, seq_len, head_dim)

        Returns:
            (q_rotated, k_rotated)
        """
        seq_len = q.size(2)
        self._update_cached(seq_len)

        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        return q_embed, k_embed


# ===== 多头因果自注意力 =====

class MultiHeadCausalAttention(nn.Module):
    """多头因果自注意力机制。

    支持可选的 RoPE 旋转位置编码（use_rope=True 时替换 SinusoidalPE）。
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_rope: bool = False,
        max_seq_len: int = 8192,
    ):
        """初始化多头注意力。

        Args:
            hidden_dim: 模型嵌入维度
            num_heads: 注意力头数
            dropout: Dropout 概率
            use_rope: 是否使用 RoPE 旋转位置编码
            max_seq_len: 最大序列长度（影响 RoPE 缓存）
        """
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim 必须能被 num_heads 整除"

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope

        # Q, K, V 线性投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        # 旋转位置编码（可选）
        if use_rope:
            self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)

        # causal mask：上三角矩阵
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.full((1, 1, 8192, 8192), float("-inf")), diagonal=1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量 (batch_size, seq_len, hidden_dim)
            attention_mask: 注意力掩码 (batch_size, seq_len)

        Returns:
            注意力输出 (batch_size, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # 线性投影 → 多头重塑
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE 旋转位置编码（作用于 Q 和 K）
        if self.use_rope:
            q, k = self.rope(q, k)

        # 注意力分数
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # causal mask
        causal_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        attn_scores = attn_scores + causal_mask

        # padding mask
        if attention_mask is not None:
            attn_mask_expanded = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(attn_mask_expanded == 0, float("-inf"))

        # softmax
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # 输出投影
        output = self.out_proj(attn_output)
        return output


# ===== 前馈神经网络 =====

class FeedForward(nn.Module):
    """两层的全连接前馈网络。

    SwiGLU 或 ReLU 激活，hidden_dim -> ffn_dim -> hidden_dim。
    """

    def __init__(self, hidden_dim: int, ffn_multiplier: int = 4, dropout: float = 0.1):
        """初始化 FFN。

        Args:
            hidden_dim: 输入/输出维度
            ffn_multiplier: FFN 中间层倍率
            dropout: Dropout 概率
        """
        super().__init__()
        ffn_dim = hidden_dim * ffn_multiplier

        self.gate_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.down_proj = nn.Linear(ffn_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量 (batch_size, seq_len, hidden_dim)

        Returns:
            输出张量 (batch_size, seq_len, hidden_dim)
        """
        # SwiGLU 激活: silu(gate(x)) * up(x)
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        x = gate * up
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


# ===== Transformer 块 =====

class TransformerBlock(nn.Module):
    """Transformer 解码器块。

    顺序：LayerNorm -> Attention -> 残差 -> LayerNorm -> FFN -> 残差
    使用 Pre-LayerNorm 架构，支持 RoPE。
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        ffn_multiplier: int = 4,
        dropout: float = 0.1,
        use_rope: bool = False,
        max_seq_len: int = 8192,
    ):
        """初始化 Transformer 块。

        Args:
            hidden_dim: 模型嵌入维度
            num_heads: 注意力头数
            ffn_multiplier: FFN 中间层倍率
            dropout: Dropout 概率
            use_rope: 是否使用 RoPE
            max_seq_len: 最大序列长度
        """
        super().__init__()

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attention = MultiHeadCausalAttention(
            hidden_dim, num_heads, dropout,
            use_rope=use_rope, max_seq_len=max_seq_len,
        )

        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = FeedForward(hidden_dim, ffn_multiplier, dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        """前向传播。

        Args:
            x: 输入张量 (batch_size, seq_len, hidden_dim)
            attention_mask: 注意力掩码 (batch_size, seq_len)

        Returns:
            输出张量 (batch_size, seq_len, hidden_dim)
        """
        # Pre-LayerNorm 残差连接
        residual = x
        x = self.ln1(x)
        x = self.attention(x, attention_mask)
        x = residual + x

        # FFN 子层
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + x

        return x


# ===== 主体 Transformer 模型 =====

class DecoderOnlyTransformer(nn.Module):
    """Decoder-only Transformer 语言模型。

    支持 SinusoidalPE 和 RoPE 两种位置编码。
    支持权重绑定（token嵌入与输出投影共享参数）。
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        ffn_multiplier: int = 4,
        tie_weights: bool = True,
        use_rope: bool = False,
    ):
        """初始化模型。

        Args:
            vocab_size: 词表大小
            hidden_dim: 嵌入维度
            num_layers: Transformer 层数
            num_heads: 注意力头数
            max_seq_len: 最大序列长度
            dropout: Dropout 概率
            ffn_multiplier: FFN 中间层倍率
            tie_weights: 是否绑定权重
            use_rope: 是否使用 RoPE（替代 SinusoidalPE）
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope

        # ===== 嵌入层 =====
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        # RoPE 不添加位置嵌入到输入，SinusoidalPE 则直接加
        if not use_rope:
            self.position_embedding = SinusoidalPositionalEmbedding(hidden_dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        # ===== Transformer 层 =====
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim, num_heads, ffn_multiplier, dropout,
                use_rope=use_rope, max_seq_len=max_seq_len,
            )
            for _ in range(num_layers)
        ])

        # ===== 输出层 =====
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # ===== 权重绑定 =====
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight
            logger.info("启用了权重绑定（token嵌入与输出投影共享参数）")

        self._init_weights()
        pos_enc = "RoPE" if use_rope else "Sinusoidal"
        logger.info(
            f"模型初始化完成: {num_layers}层, {hidden_dim}维, {num_heads}头, "
            f"位置编码: {pos_enc}, 词表{vocab_size}, 最大序列长度{max_seq_len}"
        )

    def _init_weights(self):
        """使用正态分布初始化模型权重。"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
    ) -> dict:
        """前向传播。

        Args:
            input_ids: 输入 token id (batch_size, seq_len)
            attention_mask: 注意力掩码 (batch_size, seq_len)
            labels: 目标标签 (batch_size, seq_len)，用于计算损失

        Returns:
            dict 包含:
                - logits: 输出 logits (batch_size, seq_len, vocab_size)
                - loss: 交叉熵损失（若 labels 不为 None）
        """
        batch_size, seq_len = input_ids.shape

        # Token 嵌入
        x = self.token_embedding(input_ids)
        # 位置编码：Sinusoidal 直接加在输入，RoPE 在注意力中处理
        if not self.use_rope:
            x = self.position_embedding(x)
        x = self.dropout(x)

        # 通过 Transformer 层
        for layer in self.layers:
            x = layer(x, attention_mask)

        # 最终 LayerNorm 和输出投影
        x = self.ln_f(x)
        logits = self.lm_head(x)

        # 计算损失
        loss = None
        if labels is not None:
            # 将 logits 和 labels 展平
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,  # 忽略 padding 位置
            )

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
        eos_token_id: int = None,
        pad_token_id: int = 0,
    ) -> torch.Tensor:
        """自回归文本生成。

        支持多种解码策略：贪心、温度采样、Top-k、Top-p (nucleus)。

        Args:
            input_ids: 初始 token id 序列 (batch_size, seq_len)
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度（1.0=标准，<1.0=更确定，>1.0=更随机）
            top_k: Top-k 采样的 k 值
            top_p: Top-p 采样的 p 值
            eos_token_id: 结束 token id，生成到此 id 停止
            pad_token_id: padding token id

        Returns:
            生成的 token id 序列 (batch_size, total_len)
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # 截断到 max_seq_len
            if generated.size(1) > self.max_seq_len:
                generated = generated[:, -self.max_seq_len:]

            # 前向传播获取 logits
            attention_mask = (generated != pad_token_id).long()
            outputs = self.forward(generated, attention_mask=attention_mask)
            logits = outputs["logits"]

            # 取最后一个位置的 logits
            next_logits = logits[:, -1, :] / temperature

            # Top-k 过滤
            if top_k is not None and top_k > 0:
                # 保留 top_k 个最高概率的 token，其余设为 -inf
                top_k_values, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                threshold = top_k_values[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < threshold, float("-inf"))

            # Top-p (nucleus) 过滤
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # 移除累计概率超过 top_p 的 token
                sorted_indices_to_remove = cumulative_probs > top_p
                # 偏移一位，至少保留第一个 token
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits = next_logits.masked_fill(indices_to_remove, float("-inf"))

            # 采样
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 拼接
            generated = torch.cat([generated, next_token], dim=1)

            # 检查是否生成了 eos token
            if eos_token_id is not None and (next_token == eos_token_id).any():
                break

        return generated


# ===== 模型工厂 =====

def create_model(config: dict) -> DecoderOnlyTransformer:
    """根据配置创建模型。

    Args:
        config: 配置字典，需包含 'model' 子字典

    Returns:
        创建的模型实例
    """
    model_config = config.get("model", {})
    model = DecoderOnlyTransformer(
        vocab_size=model_config.get("vocab_size", 3500),
        hidden_dim=model_config.get("hidden_dim", 256),
        num_layers=model_config.get("num_layers", 6),
        num_heads=model_config.get("num_heads", 8),
        max_seq_len=model_config.get("max_seq_len", 256),
        dropout=model_config.get("dropout", 0.1),
        ffn_multiplier=model_config.get("ffn_multiplier", 4),
        tie_weights=True,
        use_rope=model_config.get("use_rope", False),
    )
    return model
