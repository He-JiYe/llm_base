"""训练模块：模型训练、验证、检查点管理。

包含完整的训练循环，支持：
- 训练/验证循环
- 检查点保存和恢复
- TensorBoard 日志记录
- 学习率预热与余弦退火
- 梯度裁剪
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from src.utils import compute_perplexity, count_parameters, strip_compile_prefix

logger = logging.getLogger(__name__)


# ===== 学习率调度器 =====

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """创建带预热的余弦退火学习率调度器。

    学习率从 0 线性升至 peak_lr（warmup_steps 步），
    然后按余弦曲线衰减至 min_lr_ratio * peak_lr。

    Args:
        optimizer: 优化器
        num_warmup_steps: 预热步数
        num_training_steps: 总训练步数
        min_lr_ratio: 最小学习率与峰值学习率的比例

    Returns:
        学习率调度器
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # 线性预热
            return float(current_step) / float(max(1, num_warmup_steps))
        # 余弦退火
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))) * (
            1.0 - min_lr_ratio
        ) + min_lr_ratio
        return cosine_decay.item()

    return LambdaLR(optimizer, lr_lambda)


# ===== 训练器 =====

class Trainer:
    """模型训练器。

    封装训练循环，支持检查点、日志、验证等功能。
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        config: dict,
        device: torch.device,
        run_dir: str = "runs",
        checkpoint_dir: str = "checkpoints",
    ):
        """初始化训练器。

        Args:
            model: 待训练的模型
            train_loader: 训练数据加载器
            valid_loader: 验证数据加载器
            config: 训练配置字典
            device: 计算设备
            run_dir: TensorBoard 日志目录
            checkpoint_dir: 检查点保存目录
        """
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.config = config
        self.device = device

        # 路径
        self.run_dir = Path(run_dir)
        self.checkpoint_dir = Path(checkpoint_dir)

        # 训练配置
        train_config = config.get("train", {})
        self.max_epochs = train_config.get("max_epochs", 10)
        self.learning_rate = train_config.get("learning_rate", 3e-4)
        self.weight_decay = train_config.get("weight_decay", 0.01)
        self.beta1 = train_config.get("beta1", 0.9)
        self.beta2 = train_config.get("beta2", 0.95)
        self.warmup_steps = train_config.get("warmup_steps", 1000)
        self.log_interval = train_config.get("log_interval", 10)
        self.eval_interval = train_config.get("eval_interval", 100)
        self.save_interval = train_config.get("save_interval", 1000)
        self.grad_clip = train_config.get("grad_clip", 1.0)

        # ===== 性能优化选项 =====
        # 混合精度训练 (AMP)
        self.use_amp = train_config.get("use_amp", False) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        # 编译优化 (torch.compile)
        use_compile = train_config.get("compile", False) and device.type == "cuda"
        if use_compile:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("模型已启用 torch.compile（reduce-overhead 模式）")
            except Exception as e:
                logger.warning(f"torch.compile 启用失败，回退到 eager 模式: {e}")

        # 计算总训练步数
        self.steps_per_epoch = len(train_loader)
        self.total_steps = self.steps_per_epoch * self.max_epochs

        # 优化器
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
        )

        # 学习率调度器
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.total_steps,
        )

        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_valid_loss = float("inf")
        self.train_losses = []
        self.valid_losses = []

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.run_dir))

        # 打印参数量
        n_params = count_parameters(model)
        logger.info(
            f"训练器初始化完成 | "
            f"参数总量: {n_params:,} | "
            f"每轮步数: {self.steps_per_epoch} | "
            f"总步数: {self.total_steps} | "
            f"设备: {device}"
        )

    def train_epoch(self) -> float:
        """训练一个 epoch。

        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}/{self.max_epochs}",
            unit="批",
            leave=False,
        )
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch.get("attention_mask")

            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # 前向传播（混合精度）
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(input_ids, labels=labels, attention_mask=attention_mask)
                loss = outputs["loss"]

            # 反向传播（梯度缩放）
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # 梯度裁剪（需先 unscale）
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            # 统计
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # 更新进度条信息
            current_lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{current_lr:.2e}",
            )

            # 日志
            if self.global_step % self.log_interval == 0:
                avg_loss = total_loss / num_batches
                perplexity = compute_perplexity(avg_loss)

                logger.info(
                    f"Epoch [{self.current_epoch}/{self.max_epochs}] "
                    f"Step [{self.global_step}/{self.total_steps}] "
                    f"Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} | "
                    f"LR: {current_lr:.6f}"
                )

                # TensorBoard
                self.writer.add_scalar("train/loss", avg_loss, self.global_step)
                self.writer.add_scalar("train/perplexity", perplexity, self.global_step)
                self.writer.add_scalar("train/lr", current_lr, self.global_step)

        pbar.close()
        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def evaluate(self) -> float:
        """在验证集上评估模型。

        Returns:
            验证集平均损失
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.valid_loader, desc="验证中", unit="批", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch.get("attention_mask")

            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(input_ids, labels=labels, attention_mask=attention_mask)
            loss = outputs["loss"]

            total_loss += loss.item()
            num_batches += 1
        pbar.close()

        avg_loss = total_loss / max(1, num_batches)
        perplexity = compute_perplexity(avg_loss)

        logger.info(
            f"=== 验证 | Step {self.global_step} | "
            f"Loss: {avg_loss:.4f} | PPL: {perplexity:.2f} ==="
        )

        self.writer.add_scalar("valid/loss", avg_loss, self.global_step)
        self.writer.add_scalar("valid/perplexity", perplexity, self.global_step)

        self.valid_losses.append((self.global_step, avg_loss))
        return avg_loss

    def train(self):
        """完整训练流程：运行多个 epoch。"""
        logger.info("=" * 50)
        logger.info("开始训练")
        logger.info("=" * 50)
        start_time = time.time()

        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch + 1
            train_loss = self.train_epoch()
            self.train_losses.append((self.global_step, train_loss))

            # 每个 epoch 结束进行验证
            valid_loss = self.evaluate()

            # 保存 epoch 检查点
            self.save_checkpoint(f"epoch_{self.current_epoch}")

            # 更新最佳模型
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                self.save_checkpoint("best")

            elapsed = time.time() - start_time
            logger.info(
                f"Epoch [{self.current_epoch}/{self.max_epochs}] 完成 | "
                f"Train Loss: {train_loss:.4f} | "
                f"Valid Loss: {valid_loss:.4f} | "
                f"耗时: {elapsed:.1f}s"
            )

        total_time = time.time() - start_time
        logger.info("=" * 50)
        logger.info(f"训练完成 | 总耗时: {total_time:.1f}s | 最佳验证损失: {self.best_valid_loss:.4f}")
        logger.info("=" * 50)

        self.writer.close()

    def save_checkpoint(self, tag: str = "best"):
        """保存模型检查点。

        保存模型参数、优化器状态、训练状态等。

        Args:
            tag: 检查点标识
        """
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path = self.checkpoint_dir / f"model_{tag}.pt"

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            "config": self.config,
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_valid_loss": self.best_valid_loss,
            "train_losses": self.train_losses,
            "valid_losses": self.valid_losses,
        }
        torch.save(checkpoint, path)
        logger.info(f"检查点已保存: {path}")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """加载检查点恢复训练。

        Args:
            checkpoint_path: 检查点文件路径（仅用于日志和存在性检查）

        Returns:
            是否成功加载
        """
        path = Path(checkpoint_path)
        if not path.exists():
            logger.error(f"检查点文件不存在: {checkpoint_path}")
            return False

        try:
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=False
            )

            state_dict = strip_compile_prefix(checkpoint["model_state_dict"])
            self.model.load_state_dict(state_dict)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            self.current_epoch = checkpoint.get("epoch", 0)
            self.global_step = checkpoint.get("global_step", 0)
            self.best_valid_loss = checkpoint.get("best_valid_loss", float("inf"))
            self.train_losses = checkpoint.get("train_losses", [])
            self.valid_losses = checkpoint.get("valid_losses", [])

            # 恢复 AMP scaler 状态
            if self.use_amp and checkpoint.get("scaler_state_dict") is not None:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

            logger.info(
                f"检查点已加载: {checkpoint_path} | "
                f"Epoch: {self.current_epoch} | Step: {self.global_step} | "
                f"Best Loss: {self.best_valid_loss:.4f}"
            )
            return True

        except Exception as e:
            logger.error(f"加载检查点失败: {e}")
            return False
