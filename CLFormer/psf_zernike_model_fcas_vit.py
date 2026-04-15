"""
PSF到Zernike系数预测模型 - FSAS (Frequency Self-Attention) + ViT

架构设计：
    - 使用 FSAS 频率域自注意力模块替代传统傅里叶预处理
    - FSAS 在局部 patch 级别进行频率域注意力计算，效率更高
    - 使用 Vision Transformer 作为特征编码器
    - 简化架构，提高长距离依赖建模能力

数据流：
    输入 [B, 4/5, H, W] → FSAS模块 → ViT Encoder → CLS Token → MLP → Zernike系数 [B, 25/77]

参考：
    - FSAS: /data/gqzhu/data/abandon/FSAS.py
    - ViT: /data/gqzhu/data/benchmark/ViT/vit_model.py
"""

import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from einops import rearrange
import numbers

# 路径设置
CURRENT_DIR = Path(__file__).resolve().parent
VIT_MODEL_DIR = CURRENT_DIR.parent / "benchmark" / "ViT"

# 导入 ViT 模型
if str(VIT_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(VIT_MODEL_DIR))

from vit_model import VARIANT_TO_MODEL, create_timm_vit


FEATURE_DIM_BY_VARIANT = {
    "tiny": 192,
    "small": 384,
    "base": 768,
    "large": 1024,
}


# ==================== FSAS 模块 (来自 abandon/FSAS.py) ====================

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FSAS(nn.Module):
    """
    频率域自注意力模块（Frequency-Domain Self-Attention, FSAS）
    核心思想：在频率域计算注意力，替代传统空间域自注意力，提升长距离依赖建模效率
    """

    def __init__(self, dim, bias=False):
        super(FSAS, self).__init__()
        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1,
                                      padding=1, groups=dim * 6, bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')
        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)
        hidden_dw = self.to_hidden_dw(hidden)
        q, k, v = hidden_dw.chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2',
                            patch1=self.patch_size, patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2',
                            patch1=self.patch_size, patch2=self.patch_size)

        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())
        out_fft = q_fft * k_fft
        out = torch.fft.irfft2(out_fft, s=(self.patch_size, self.patch_size))

        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)',
                        patch1=self.patch_size, patch2=self.patch_size)

        out_norm = self.norm(out)
        output = v * out_norm
        output = self.project_out(output)
        return output


class ChannelAttention(nn.Module):
    """CBAM channel-attention branch."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced_channels = max(channels // reduction, 1)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """CBAM spatial-attention branch."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("SpatialAttention kernel_size must be odd.")
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        return self.sigmoid(attention)


class CBAMBlock(nn.Module):
    """Standard CBAM refinement block."""

    def __init__(self, channels: int, reduction: int = 16, spatial_kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels=channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


# ==================== Coordinate Attention (CVPR 2021) ====================

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttention(nn.Module):
    """
    Standard Coordinate Attention
    Paper: Coordinate Attention for Efficient Mobile Network Design
    Input:  [B, C, H, W]
    Output: [B, C, H, W]
    """

    def __init__(self, inp, reduction=32):
        super().__init__()

        mip = max(8, inp // reduction)

        # 分别沿 W / H 做池化
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # [B,C,H,1]
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # [B,C,1,W]

        # 共享变换
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        # 分别生成 H 和 W 方向注意力
        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # x_h: [B,C,H,1]
        x_h = self.pool_h(x)

        # x_w: [B,C,1,W] -> [B,C,W,1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # 拼接后共享编码
        y = torch.cat([x_h, x_w], dim=2)   # [B,C,H+W,1]
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 按 H/W 拆开
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)      # [B,C,1,W]

        # 生成两个方向的注意力图
        a_h = torch.sigmoid(self.conv_h(x_h))   # [B,C,H,1]
        a_w = torch.sigmoid(self.conv_w(x_w))   # [B,C,1,W]

        out = identity * a_h * a_w
        return out
# ==================== FSAS + CA + CBAM 融合模块 ====================

class FSASFourierModule(nn.Module):
    """
    使用 FSAS (Frequency Self-Attention) + CA + CBAM 的傅里叶模块

        1. 输入分两路：CA分支 + FSAS分支
        2. CA处理原始通道，捕获位置+通道信息
        3. FSAS计算频率域注意力
        4. CA输出与FSAS输出融合（concat + Conv）
        5. CBAM精炼融合特征
    
    流程：输入 → [CA分支 | FSAS分支] → concat → Fusion → CBAM → 输出
    """

    def __init__(
        self,
        channels: int,
        bias: bool = False,
        patch_size: int = 8,
        cbam_reduction: int = 16,
        cbam_kernel_size: int = 7,
        ca_reduction: int = 32,
    ):
        super().__init__()
        self.channels = channels
        self.patch_size = patch_size
        hidden_channels = max(channels * 2, 8)

        # CA 分支：处理原始通道，捕获位置+通道信息
        self.ca = CoordAttention(inp=channels, reduction=ca_reduction)

        # FSAS 频率域自注意力块
        self.fsas = FSAS(dim=channels, bias=bias)

        # 融合层：拼接CA输出和FSAS输出
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=bias),
            nn.BatchNorm2d(channels),
        )

        # CBAM 模块：融合后精炼
        self.cbam = CBAMBlock(
            channels=channels,
            reduction=cbam_reduction,
            spatial_kernel_size=cbam_kernel_size
        )

        self.residual_scale = nn.Parameter(torch.tensor(0.1))
        self.output_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CA 分支：捕获原始通道的位置+通道信息
        ca_features = self.ca(x)
        
        # FSAS 分支：频率域注意力
        fsas_features = self.fsas(x)
        
        # 融合 CA 和 FSAS 特征（与原来相同的融合方式）
        fused_features = self.fusion(torch.cat([ca_features, fsas_features], dim=1))
        
        # CBAM 精炼
        out = self.cbam(x + self.residual_scale * fused_features)
        return self.output_act(out)


# ==================== Conv Ablation Backbone ====================

class ConvAblationBackbone(nn.Module):
    """当 ViT 禁用时的卷积降级 backbone"""

    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        stage1 = max(32, out_dim // 8)
        stage2 = max(64, out_dim // 4)
        stage3 = max(128, out_dim // 2)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, stage1, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stage1),
            nn.GELU(),
            nn.Conv2d(stage1, stage2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stage2),
            nn.GELU(),
            nn.Conv2d(stage2, stage3, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stage3),
            nn.GELU(),
            nn.Conv2d(stage3, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


# ==================== 主模型：FSAS + ViT ====================

class FCASViTPSFModel(nn.Module):
    """
    PSF到Zernike系数模型 - FSAS 频率域注意力 + CA + CBAM + ViT
    
    新架构（方案B）：
        1. 输入分两路：CA分支 + FSAS分支
        2. CA处理原始通道，捕获位置+通道信息
        3. FSAS计算频率域注意力
        4. CA输出与FSAS输出融合（concat + Conv）
        5. CBAM精炼融合特征

    流程：输入 → [CA分支 | FSAS分支] → concat → Fusion → CBAM → ViT → 输出

    输入: [B, 4/5, H, W] PSF图像
    输出: [B, 25/77] Zernike系数
    """

    def __init__(
        self,
        input_size: int = 112,
        patch_size: int = 16,
        num_images: int = 4,
        channels_per_image: int = 1,
        num_coefficients: int = 25,
        variant: str = "small",
        pretrained: bool = True,
        dropout: float = 0.1,
        use_vit_module: bool = True,
        fsas_patch_size: int = 8,
        cbam_reduction: int = 16,
        cbam_kernel_size: int = 7,
        ca_reduction: int = 32,
    ):
        super().__init__()

        if variant not in VARIANT_TO_MODEL:
            raise ValueError(f"Unsupported ViT variant: {variant}")

        self.input_size = input_size
        self.patch_size = patch_size
        self.num_images = num_images
        self.channels_per_image = channels_per_image
        self.num_coefficients = num_coefficients
        self.total_channels = num_images * channels_per_image
        self.variant = variant
        self.use_vit_module = use_vit_module
        self.fsas_patch_size = fsas_patch_size
        self.cbam_reduction = cbam_reduction
        self.cbam_kernel_size = cbam_kernel_size

        # 输入归一化
        self.stem_norm = nn.BatchNorm2d(self.total_channels)

        # FSAS + CA + CBAM 频率域自注意力模块
        self.fsas_module = FSASFourierModule(
            channels=self.total_channels,
            bias=False,
            patch_size=fsas_patch_size,
            cbam_reduction=cbam_reduction,
            cbam_kernel_size=cbam_kernel_size,
            ca_reduction=ca_reduction,
        )

        # ViT 或 Conv backbone
        if use_vit_module:
            model_name = VARIANT_TO_MODEL[variant]
            self.backbone, self.using_pretrained = create_timm_vit(
                model_name=model_name,
                total_channels=self.total_channels,
                pretrained=pretrained,
            )
            with torch.no_grad():
                dummy_input = torch.randn(1, self.total_channels, input_size, input_size)
                feature_dim = int(self.backbone(dummy_input).shape[1])
            self.conv_backbone = None
        else:
            self.using_pretrained = False
            self.backbone = None
            feature_dim = FEATURE_DIM_BY_VARIANT[variant]
            self.conv_backbone = ConvAblationBackbone(
                in_channels=self.total_channels,
                out_dim=feature_dim,
            )

        # MLP 回归头
        hidden_dim = max(feature_dim // 2, num_coefficients * 2)
        bottleneck_dim = max(feature_dim // 4, num_coefficients)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, num_coefficients),
        )

    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input [B, C, H, W], got {tuple(x.shape)}")
        if x.shape[1] != self.total_channels:
            raise ValueError(f"Expected {self.total_channels} channels, got {x.shape[1]}")

        x = self.stem_norm(x)
        x = self.fsas_module(x)

        if self.use_vit_module:
            return self.backbone(x)
        return self.conv_backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encode_features(x)
        return self.head(features)


def create_model(
    input_size: int = 112,
    patch_size: int = 16,
    num_images: int = 4,
    channels_per_image: int = 1,
    num_coefficients: int = 25,
    variant: str = "small",
    pretrained: bool = True,
    dropout: float = 0.1,
    use_vit_module: bool = True,
    fsas_patch_size: int = 8,
    cbam_reduction: int = 16,
    cbam_kernel_size: int = 7,
    ca_reduction: int = 32,
) -> FCASViTPSFModel:
    """Factory wrapper for creating FSAS + CA + CBAM + ViT model"""
    return FCASViTPSFModel(
        input_size=input_size,
        patch_size=patch_size,
        num_images=num_images,
        channels_per_image=channels_per_image,
        num_coefficients=num_coefficients,
        variant=variant,
        pretrained=pretrained,
        dropout=dropout,
        use_vit_module=use_vit_module,
        fsas_patch_size=fsas_patch_size,
        cbam_reduction=cbam_reduction,
        cbam_kernel_size=cbam_kernel_size,
        ca_reduction=ca_reduction,
    )


def count_parameters(model: nn.Module) -> Dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "size_mb": total * 4 / (1024 * 1024),
    }


if __name__ == "__main__":
    print("=" * 80)
    print("FSAS + ViT PSF-to-Zernike model smoke test")
    print("=" * 80)

    ablation_settings = [
        {"use_vit_module": True, "variant": "small", "num_images": 5, "num_coefficients": 77},
        {"use_vit_module": True, "variant": "base", "num_images": 5, "num_coefficients": 77},
        {"use_vit_module": True, "variant": "tiny", "num_images": 4, "num_coefficients": 25},
        {"use_vit_module": False, "variant": "small", "num_images": 5, "num_coefficients": 77},
    ]

    for settings in ablation_settings:
        print("-" * 80)
        print(f"Settings: {settings}")
        print("-" * 80)

        model = create_model(
            input_size=112,
            patch_size=8,
            pretrained=False,
            dropout=0.1,
            **settings,
        )
        params = count_parameters(model)

        x = torch.randn(2, settings["num_images"], 112, 112)
        with torch.no_grad():
            y = model(x)

        print(f"Total params: {params['total']:,}")
        print(f"Trainable params: {params['trainable']:,}")
        print(f"Model size: {params['size_mb']:.2f} MB")
        print(f"Input shape:  {tuple(x.shape)}")
        print(f"Output shape: {tuple(y.shape)}")

    print("=" * 80)
    print("Smoke test completed")
    print("=" * 80)
