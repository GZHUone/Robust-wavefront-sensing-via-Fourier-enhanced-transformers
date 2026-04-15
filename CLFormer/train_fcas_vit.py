"""
训练脚本 - FSAS (Frequency Self-Attention) + ViT PSF到Zernike系数模型

支持两种模式：
1. 25系数模式：4张PSF图像 -> 25个Zernike系数
2. 77系数模式：5张PSF图像 -> 77个Zernike系数（7×11，第7行前3个为0）

用法:
    # 25系数模式
    python train_fcas_vit.py --train_dir /path/to/train --num_images 4 --num_coefficients 25

    # 77系数模式
    python train_fcas_vit.py --train_dir /path/to/train --num_images 5 --num_coefficients 77
"""

import os
import argparse
import glob
import pickle
import _pickle
import math
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

from psf_zernike_model_fcas_vit import create_model, count_parameters


# ==================== Loss Functions ====================

def get_loss_mask_77(device='cpu'):
    """
    获取loss计算的mask (77系数版本)
    第7行(index=6)的前3个位置为0，不参与loss计算

    Returns:
        mask: [77] tensor, 1表示参与计算，0表示不参与
    """
    mask = torch.ones(77, device=device)
    mask[66] = 0  # 第7行第1个
    mask[67] = 0  # 第7行第2个
    mask[68] = 0  # 第7行第3个
    return mask


class MaskedMSELoss(nn.Module):
    """带mask的MSE Loss (77系数版本)"""

    def __init__(self):
        super().__init__()
        self.mask = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: [B, 77] 预测值
            target: [B, 77] 目标值
        """
        if self.mask is None or self.mask.device != pred.device:
            self.mask = get_loss_mask_77(pred.device)

        # 只计算mask为1的位置
        diff = (pred - target) ** 2
        masked_diff = diff * self.mask.unsqueeze(0)

        # 计算平均loss (除以有效位置数量74)
        loss = masked_diff.sum() / (self.mask.sum() * pred.shape[0])
        return loss


# ==================== Dataset ====================

class PSFDataset(Dataset):
    """PSF数据集 - 支持25和77系数两种模式"""

    def __init__(self, data_dir, num_images=4, num_coefficients=25):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if not self.files:
            raise FileNotFoundError(f"No .npy files in {data_dir}")

        self.num_images = num_images
        self.num_coefficients = num_coefficients

        # 检测数据格式
        sample_data = np.load(self.files[0], allow_pickle=True)
        if isinstance(sample_data[1], dict):
            self.label_shape = sample_data[1]['gt_a'].shape
        else:
            self.label_shape = sample_data[1].shape

        print(f"Dataset: {data_dir}")
        print(f"  Samples: {len(self.files)}")
        print(f"  Mode: {num_images} images -> {num_coefficients} coefficients")
        print(f"  Input shape: {sample_data[0].shape}, Label shape: {self.label_shape}")

        # 验证数据集
        self._validate_dataset()

    def _validate_dataset(self):
        """验证数据集完整性"""
        print("Validating dataset integrity...")
        corrupted_files = []

        check_count = min(100, len(self.files))
        for i in range(check_count):
            try:
                data = np.load(self.files[i], allow_pickle=True)
                if len(data) < 2:
                    corrupted_files.append(self.files[i])
            except (EOFError, ValueError, OSError, pickle.UnpicklingError, _pickle.UnpicklingError):
                corrupted_files.append(self.files[i])

        if corrupted_files:
            print(f"Found {len(corrupted_files)} corrupted files in sample check")
            print("Note: Corrupted files will be skipped during training")
        else:
            print("Dataset validation passed")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        max_retries = 5
        for retry in range(max_retries):
            try:
                current_idx = (idx + retry) % len(self.files)
                data = np.load(self.files[current_idx], allow_pickle=True)

                if len(data) < 2:
                    raise ValueError("Incomplete data")

                # 输入: 取前num_images张图像
                inputs = torch.from_numpy(data[0][:self.num_images].astype(np.float32))

                # 标签处理
                if isinstance(data[1], dict):
                    labels_matrix = data[1]['gt_a'].astype(np.float32)
                else:
                    labels_matrix = data[1].astype(np.float32)

                if self.num_coefficients == 25:
                    # 25系数模式
                    coeffs = np.zeros(25, dtype=np.float32)
                    coeffs[0] = labels_matrix[0, 3]
                    coeffs[1:] = labels_matrix[1:7, :4].reshape(-1)
                    labels = torch.from_numpy(coeffs)
                elif self.num_coefficients == 77:
                    # 77系数模式: 取前7行前11列
                    labels_matrix = labels_matrix[:7, :11]
                    labels = torch.from_numpy(labels_matrix.reshape(-1))
                else:
                    raise ValueError(f"Unsupported num_coefficients: {self.num_coefficients}")

                return inputs, labels

            except (EOFError, ValueError, OSError, IndexError, pickle.UnpicklingError, _pickle.UnpicklingError) as e:
                if retry == max_retries - 1:
                    print(f"Error: All retries failed for index {idx}, returning zero data")
                    inputs = torch.zeros(self.num_images, 112, 112, dtype=torch.float32)
                    labels = torch.zeros(self.num_coefficients, dtype=torch.float32)
                    return inputs, labels
                continue


# ==================== Training Utils ====================

class EarlyStopping:
    """Early stopping helper"""

    def __init__(self, patience=50, min_delta=0.0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return

        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0


def setup_logger(log_path):
    logger = logging.getLogger("FCASViT")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path, mode="a")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def save_training_artifacts(history, checkpoint_dir):
    if not history:
        return
    df = pd.DataFrame(history)
    excel_path = os.path.join(checkpoint_dir, "train_history.xlsx")
    df.to_excel(excel_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o", markersize=3, linewidth=1.2)
    plt.plot(df["epoch"], df["val_loss"], label="Val Loss", marker="s", markersize=3, linewidth=1.2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (RMSE)")
    plt.title("Training vs Validation Loss (FSAS + ViT)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plot_path = os.path.join(checkpoint_dir, "train_history.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler=None, use_amp=False, max_grad_norm=1.0):
    model.train()
    running_loss = 0.0
    progress = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}", ncols=120)

    for batch_idx, (inputs, labels) in progress:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        try:
            if use_amp and scaler is not None:
                with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                    outputs = model(inputs)
                    mse = criterion(outputs, labels)
                    loss = torch.sqrt(mse + 1e-8)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                mse = criterion(outputs, labels)
                loss = torch.sqrt(mse + 1e-8)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            running_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n警告: Batch {batch_idx} 显存不足，跳过此batch")
                torch.cuda.empty_cache()
                continue
            else:
                raise e

        # 定期清理显存
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()

    return running_loss / len(dataloader)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, use_amp=False):
    model.eval()
    running_rmse = 0.0

    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp:
            with autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                outputs = model(inputs)
                loss_mse = criterion(outputs, labels)
        else:
            outputs = model(inputs)
            loss_mse = criterion(outputs, labels)

        loss_rmse = torch.sqrt(loss_mse + 1e-8)
        running_rmse += loss_rmse.item()

    return running_rmse / len(dataloader)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.checkpoint_dir, f"train_{timestamp}.log")
    logger = setup_logger(log_path)

    logger.info("=" * 80)
    logger.info("FSAS + ViT Model Training")
    logger.info(f"Mode: {args.num_images} PSF images -> {args.num_coefficients} Zernike coefficients")
    if args.num_coefficients == 77:
        logger.info("第7行前3个为0，不参与loss计算 (74个有效系数)")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Train dir: {args.train_dir}")
    logger.info(f"Val dir: {args.val_dir}")
    logger.info(f"Test dir: {args.test_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"LR: {args.lr}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Variant: {args.variant}")
    logger.info(f"Patch size: {args.patch_size}")
    logger.info(f"FSAS patch size: {args.fsas_patch_size}")
    logger.info(f"CBAM reduction: {args.cbam_reduction}")
    logger.info(f"CBAM kernel size: {args.cbam_kernel_size}")
    logger.info(f"Use ViT: {args.use_vit_module}")
    logger.info(f"Use AMP: {args.use_amp}")
    logger.info("=" * 80)

    # 数据集
    train_dataset = PSFDataset(args.train_dir, args.num_images, args.num_coefficients)
    val_dataset = PSFDataset(args.val_dir, args.num_images, args.num_coefficients)
    test_dataset = PSFDataset(args.test_dir, args.num_images, args.num_coefficients)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0
    )

    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    # 模型
    model = create_model(
        input_size=args.input_size,
        patch_size=args.patch_size,
        num_images=args.num_images,
        channels_per_image=1,
        num_coefficients=args.num_coefficients,
        variant=args.variant,
        pretrained=False,
        dropout=args.dropout,
        use_vit_module=args.use_vit_module,
        fsas_patch_size=args.fsas_patch_size,
        cbam_reduction=args.cbam_reduction,
        cbam_kernel_size=args.cbam_kernel_size,
        ca_reduction=args.ca_reduction,
    )

    params = count_parameters(model)
    logger.info(f"Model parameters: {params['total']:,} ({params['size_mb']:.2f} MB)")

    # 先将模型移到GPU
    model = model.to(device)

    # 然后使用DataParallel (如果有多个GPU)
    if torch.cuda.device_count() > 1:
        logger.info(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
        torch.cuda.empty_cache()
        model = nn.DataParallel(model)
        logger.info(f"Effective batch size per GPU: {args.batch_size // torch.cuda.device_count()}")

    # Loss function
    if args.num_coefficients == 77:
        criterion = MaskedMSELoss()
    else:
        criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler: Cosine with warmup
    warmup_epochs = args.epochs // 10
    def lr_fn(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_fn)

    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    scaler = GradScaler() if args.use_amp else None

    best_val = float("inf")
    best_path = os.path.join(args.checkpoint_dir, "model_best.pth")
    history = []

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, scaler, args.use_amp, args.max_grad_norm)
        val_loss = evaluate(model, val_loader, criterion, device, args.use_amp)
        scheduler.step()
        early_stopping(val_loss, epoch)

        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch}/{args.epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {current_lr:.2e}")
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr
        })

        if epoch % 10 == 0:
            save_training_artifacts(history, args.checkpoint_dir)
            epoch_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), epoch_path)

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            logger.info(f"  -> Best model saved")

        if early_stopping.early_stop:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Test
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location="cpu"))

    test_loss = evaluate(model, test_loader, criterion, device, args.use_amp)
    logger.info(f"Test RMSE: {test_loss:.4f}")

    # Save final
    final_path = os.path.join(args.checkpoint_dir, "model_final.pth")
    torch.save(model.state_dict(), final_path)
    save_training_artifacts(history, args.checkpoint_dir)
    logger.info("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FSAS + ViT Model")

    # Data
    parser.add_argument("--train_dir", type=str, required=True, help="Training data directory")
    parser.add_argument("--val_dir", type=str, required=True, help="Validation data directory")
    parser.add_argument("--test_dir", type=str, required=True, help="Test data directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_fcas_vit", help="Checkpoint directory")

    # Model
    parser.add_argument("--input_size", type=int, default=112, help="Input image size")
    parser.add_argument("--patch_size", type=int, default=8, help="Patch size for ViT")
    parser.add_argument("--fsas_patch_size", type=int, default=8, help="Patch size for FSAS")
    parser.add_argument("--cbam_reduction", type=int, default=16, help="CBAM channel reduction ratio")
    parser.add_argument("--cbam_kernel_size", type=int, default=7, help="CBAM spatial attention kernel size")
    parser.add_argument("--ca_reduction", type=int, default=32, help="CA (Coordinate Attention) channel reduction ratio")
    parser.add_argument("--num_images", type=int, default=4, choices=[4, 5], help="Number of input images (4 or 5)")
    parser.add_argument("--num_coefficients", type=int, default=25, choices=[25, 77], help="Number of output coefficients (25 or 77)")
    parser.add_argument("--variant", type=str, default="small", choices=["tiny", "small", "base", "large"], help="Model variant")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--use_vit_module", action="store_true", default=True, help="Use ViT backbone (vs Conv ablation)")

    # Training
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm")
    parser.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use AMP")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience")

    args = parser.parse_args()

    # Validate arguments
    if args.num_images == 4 and args.num_coefficients != 25:
        print("Warning: 4 images typically used with 25 coefficients")
    if args.num_images == 5 and args.num_coefficients != 77:
        print("Warning: 5 images typically used with 77 coefficients")

    main(args)
