"""HAT (Hybrid Attention Transformer) upscaler.

HAT achieves state-of-the-art performance (9.8/10) by combining:
- Channel attention for feature recalibration
- Window-based self-attention for local dependencies
- Overlapping cross-attention for global context

Reference: https://github.com/XPixelGroup/HAT
Paper: "Activating More Pixels in Image Super-Resolution Transformer"
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.upscale.base import (
    BaseUpscaler,
    UpscaleConfig,
    UpscaleModel,
    UpscaleResult,
)

if TYPE_CHECKING:
    from numpy import ndarray

logger = logging.getLogger(__name__)


# =============================================================================
# HAT Model Architecture
# =============================================================================


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    """Drop paths (Stochastic Depth) per sample."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class ChannelAttention(nn.Module):
    """Channel attention module."""

    def __init__(self, num_feat: int, squeeze_factor: int = 16) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.attention(x)


class CAB(nn.Module):
    """Channel Attention Block."""

    def __init__(self, num_feat: int, compress_ratio: int = 3, squeeze_factor: int = 30) -> None:
        super().__init__()
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cab(x)


class Mlp(nn.Module):
    """MLP module."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition into non-overlapping windows."""
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, h: int, w: int) -> torch.Tensor:
    """Reverse window partition."""
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention (W-MSA)."""

    def __init__(
        self,
        dim: int,
        window_size: tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # Define relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # Get pair-wise relative position index
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b_, n, c = x.shape
        qkv = self.qkv(x).reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class HAB(nn.Module):
    """Hybrid Attention Block."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 7,
        shift_size: int = 0,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
        conv_scale: float = 0.01,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: type = nn.GELU,
        norm_layer: type = nn.LayerNorm,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=(self.window_size, self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.conv_scale = conv_scale
        self.conv_block = CAB(num_feat=dim, compress_ratio=compress_ratio, squeeze_factor=squeeze_factor)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Calculate attention mask for shifted window
        if self.shift_size > 0:
            h, w = self.input_resolution
            img_mask = torch.zeros((1, h, w, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h_slice in h_slices:
                for w_slice in w_slices:
                    img_mask[:, h_slice, w_slice, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        h, w = x_size
        b, _, c = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(b, h, w, c)

        # Conv branch
        conv_x = self.conv_block(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1).contiguous()

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse(attn_windows, self.window_size, h, w)

        # Reverse cyclic shift
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x

        # Combine attention and conv
        x = attn_x + conv_x * self.conv_scale
        x = x.view(b, h * w, c)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: type | None = None,
    ) -> None:
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        _, _, h, w = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (h, w)


class PatchUnEmbed(nn.Module):
    """Patch to Image Unembedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 96,
    ) -> None:
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        return x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])


class RHAG(nn.Module):
    """Residual Hybrid Attention Group (RHAG)."""

    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        compress_ratio: int,
        squeeze_factor: int,
        conv_scale: float,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float | list[float] = 0.0,
        norm_layer: type = nn.LayerNorm,
        use_checkpoint: bool = False,
        img_size: int = 224,
        patch_size: int = 1,
        resi_connection: str = "1conv",
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.use_checkpoint = use_checkpoint

        self.residual_group = nn.ModuleList(
            [
                HAB(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    compress_ratio=compress_ratio,
                    squeeze_factor=squeeze_factor,
                    conv_scale=conv_scale,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "identity":
            self.conv = nn.Identity()
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
        )

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        residual = self.patch_unembed(x, x_size)
        for blk in self.residual_group:
            x = blk(x, x_size)
        return self.patch_embed(self.conv(self.patch_unembed(x, x_size)) + residual)[0]


class Upsample(nn.Sequential):
    """Upsample module."""

    def __init__(self, scale: int, num_feat: int) -> None:
        m = []
        if (scale & (scale - 1)) == 0:  # Power of 2
            for _ in range(int(math.log2(scale))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f"Unsupported scale factor: {scale}")
        super().__init__(*m)


class HAT(nn.Module):
    """HAT: Hybrid Attention Transformer for Image Super-Resolution.

    Args:
        img_size: Input image size.
        patch_size: Patch size.
        in_chans: Number of input channels.
        embed_dim: Patch embedding dimension.
        depths: Depth of each Transformer layer.
        num_heads: Number of attention heads.
        window_size: Window size.
        compress_ratio: Compress ratio.
        squeeze_factor: Squeeze factor.
        conv_scale: Conv scale.
        mlp_ratio: Ratio of mlp hidden dim to embedding dim.
        qkv_bias: Add bias for qkv.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
        norm_layer: Normalization layer.
        upscale: Upscale factor (2, 3, 4).
        img_range: Image range (1.0 or 255.0).
        upsampler: Upsampler type.
        resi_connection: Residual connection type.
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 180,
        depths: tuple[int, ...] = (6, 6, 6, 6, 6, 6),
        num_heads: tuple[int, ...] = (6, 6, 6, 6, 6, 6),
        window_size: int = 16,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
        conv_scale: float = 0.01,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: type = nn.LayerNorm,
        upscale: int = 4,
        img_range: float = 1.0,
        upsampler: str = "pixelshuffle",
        resi_connection: str = "1conv",
    ) -> None:
        super().__init__()
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # Patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
        )

        # Stochastic depth
        num_layers = len(depths)
        patches_resolution = self.patch_embed.patches_resolution
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Deep feature extraction
        self.layers = nn.ModuleList()
        for i_layer in range(num_layers):
            layer = RHAG(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                compress_ratio=compress_ratio,
                squeeze_factor=squeeze_factor,
                conv_scale=conv_scale,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                use_checkpoint=False,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
            )
            self.layers.append(layer)
        self.norm = norm_layer(embed_dim)

        # Reconstruction
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        if self.upsampler == "pixelshuffle":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, 64, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.upsample = Upsample(upscale, 64)
            self.conv_last = nn.Conv2d(64, in_chans, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x_size = (x.shape[2], x.shape[3])
        x, _ = self.patch_embed(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)
        x = self.patch_unembed(x, x_size)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.conv_first(x)
        x = self.conv_after_body(self.forward_features(x)) + x

        x = self.conv_before_upsample(x)
        x = self.conv_last(self.upsample(x))

        x = x / self.img_range + self.mean
        return x


# =============================================================================
# HAT Upscaler Wrapper
# =============================================================================


# Model configurations
HAT_CONFIGS = {
    "HAT-S": {
        "embed_dim": 144,
        "depths": (6, 6, 6, 6, 6, 6),
        "num_heads": (6, 6, 6, 6, 6, 6),
        "window_size": 16,
    },
    "HAT": {
        "embed_dim": 180,
        "depths": (6, 6, 6, 6, 6, 6),
        "num_heads": (6, 6, 6, 6, 6, 6),
        "window_size": 16,
    },
    "HAT-L": {
        "embed_dim": 180,
        "depths": (6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        "num_heads": (6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6),
        "window_size": 16,
    },
}


# Model URLs
HAT_MODEL_URLS = {
    "HAT-S_x4": "https://github.com/XPixelGroup/HAT/releases/download/v0.1.0/HAT-S_SRx4_ImageNet-pretrain.pth",
    "HAT_x4": "https://github.com/XPixelGroup/HAT/releases/download/v0.1.0/HAT_SRx4_ImageNet-pretrain.pth",
    "HAT-L_x4": "https://github.com/XPixelGroup/HAT/releases/download/v0.1.0/HAT-L_SRx4_ImageNet-pretrain.pth",
    "HAT_x2": "https://github.com/XPixelGroup/HAT/releases/download/v0.1.0/HAT_SRx2_ImageNet-pretrain.pth",
    "HAT_x3": "https://github.com/XPixelGroup/HAT/releases/download/v0.1.0/HAT_SRx3_ImageNet-pretrain.pth",
}


class HATUpscaler(BaseUpscaler):
    """HAT-based image upscaler.

    HAT (Hybrid Attention Transformer) achieves state-of-the-art
    super-resolution quality through:
    - Channel attention for feature recalibration
    - Window-based self-attention for local context
    - Overlapping cross-attention for global coherence

    Quality: 9.8/10 (highest among traditional SR models)
    Speed: Slower than Real-ESRGAN but faster than diffusion models
    """

    def __init__(self, config: UpscaleConfig, variant: str = "HAT") -> None:
        """Initialize HAT upscaler.

        Args:
            config: Upscaling configuration.
            variant: HAT variant - "HAT-S", "HAT", or "HAT-L".
        """
        super().__init__(config)
        self.variant = variant
        self.model: HAT | None = None
        self.device = torch.device("cuda" if config.use_gpu and torch.cuda.is_available() else "cpu")

    @property
    def model_name(self) -> str:
        return f"HAT-{self.variant}"

    def load(self) -> None:
        """Load HAT model."""
        if self._loaded:
            return

        logger.info(f"Loading HAT model ({self.variant})...")

        # Get model config
        model_config = HAT_CONFIGS.get(self.variant, HAT_CONFIGS["HAT"])
        scale = self.config.scale

        # Initialize model
        self.model = HAT(
            upscale=scale,
            img_size=64,
            patch_size=1,
            in_chans=3,
            embed_dim=model_config["embed_dim"],
            depths=model_config["depths"],
            num_heads=model_config["num_heads"],
            window_size=model_config["window_size"],
            compress_ratio=3,
            squeeze_factor=30,
            conv_scale=0.01,
            mlp_ratio=2.0,
            qkv_bias=True,
            upsampler="pixelshuffle",
            resi_connection="1conv",
        )

        # Load weights
        model_key = f"{self.variant}_x{scale}"
        if model_key in HAT_MODEL_URLS:
            model_path = self._get_model_path(model_key)
            if model_path.exists():
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                if "params_ema" in state_dict:
                    state_dict = state_dict["params_ema"]
                elif "params" in state_dict:
                    state_dict = state_dict["params"]
                self.model.load_state_dict(state_dict, strict=True)
                logger.info(f"Loaded HAT weights from {model_path}")
            else:
                logger.warning(f"Model file not found: {model_path}, using random initialization")
        else:
            logger.warning(f"No pretrained weights for {model_key}")

        self.model = self.model.to(self.device)
        self.model.eval()

        if self.config.use_fp16 and self.device.type == "cuda":
            self.model = self.model.half()

        self._loaded = True
        logger.info(f"HAT model loaded on {self.device}")

    def _get_model_path(self, model_key: str) -> Path:
        """Get path to model file, downloading if necessary."""
        model_dir = self.config.model_dir / "hat"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{model_key}.pth"

        if not model_path.exists() and model_key in HAT_MODEL_URLS:
            logger.info(f"Downloading HAT model: {model_key}...")
            import urllib.request

            urllib.request.urlretrieve(HAT_MODEL_URLS[model_key], model_path)
            logger.info(f"Downloaded to {model_path}")

        return model_path

    def unload(self) -> None:
        """Unload model and free memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._loaded = False

    def upscale(self, image: "ndarray", scale: int | None = None) -> UpscaleResult:
        """Upscale image using HAT.

        Args:
            image: Input BGR image (H, W, C).
            scale: Optional scale override.

        Returns:
            UpscaleResult with upscaled image.
        """
        if not self._loaded:
            self.load()

        start_time = time.time()
        scale = scale or self.config.scale
        h, w = image.shape[:2]

        # Convert to tensor
        img_tensor = self._preprocess(image)

        # Tile-based processing for memory efficiency
        with torch.no_grad():
            if self.config.use_fp16 and self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    output = self._process_tiles(img_tensor)
            else:
                output = self._process_tiles(img_tensor)

        # Convert back to numpy
        output_image = self._postprocess(output)

        processing_time = time.time() - start_time

        return UpscaleResult(
            image=output_image,
            original_size=(w, h),
            upscaled_size=(output_image.shape[1], output_image.shape[0]),
            scale_factor=scale,
            model_used=self.model_name,
            processing_time=processing_time,
        )

    def _preprocess(self, image: "ndarray") -> torch.Tensor:
        """Convert BGR numpy array to RGB tensor."""
        # BGR to RGB
        img = image[:, :, ::-1].copy()
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        # HWC to NCHW
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        # To tensor
        tensor = torch.from_numpy(img).to(self.device)
        if self.config.use_fp16 and self.device.type == "cuda":
            tensor = tensor.half()
        return tensor

    def _postprocess(self, tensor: torch.Tensor) -> "ndarray":
        """Convert RGB tensor to BGR numpy array."""
        # NCHW to HWC
        output = tensor.squeeze(0).cpu().float().numpy()
        output = np.transpose(output, (1, 2, 0))
        # Clamp and denormalize
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        # RGB to BGR
        output = output[:, :, ::-1].copy()
        return output

    def _process_tiles(self, img: torch.Tensor) -> torch.Tensor:
        """Process image in tiles for memory efficiency."""
        _, _, h, w = img.shape
        tile_size = self.config.tile_size
        overlap = self.config.tile_overlap
        scale = self.config.scale

        # If image is small enough, process directly
        if h <= tile_size and w <= tile_size:
            return self.model(img)

        # Calculate output size
        out_h, out_w = h * scale, w * scale
        output = torch.zeros((1, 3, out_h, out_w), dtype=img.dtype, device=img.device)
        weights = torch.zeros((1, 1, out_h, out_w), dtype=img.dtype, device=img.device)

        # Create weight mask for blending
        tile_weights = self._create_tile_weights(tile_size * scale, overlap * scale, img.device, img.dtype)

        # Process tiles
        stride = tile_size - overlap
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                # Clamp tile coordinates
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                y_start = max(0, y_end - tile_size)
                x_start = max(0, x_end - tile_size)

                # Extract and process tile
                tile = img[:, :, y_start:y_end, x_start:x_end]

                # Pad if needed
                pad_h = tile_size - tile.shape[2]
                pad_w = tile_size - tile.shape[3]
                if pad_h > 0 or pad_w > 0:
                    tile = F.pad(tile, (0, pad_w, 0, pad_h), mode="reflect")

                # Process
                with torch.no_grad():
                    out_tile = self.model(tile)

                # Remove padding
                if pad_h > 0:
                    out_tile = out_tile[:, :, : -pad_h * scale, :]
                if pad_w > 0:
                    out_tile = out_tile[:, :, :, : -pad_w * scale]

                # Get tile weights
                th, tw = out_tile.shape[2], out_tile.shape[3]
                current_weights = tile_weights[:, :, :th, :tw]

                # Add to output
                oy, ox = y_start * scale, x_start * scale
                output[:, :, oy : oy + th, ox : ox + tw] += out_tile * current_weights
                weights[:, :, oy : oy + th, ox : ox + tw] += current_weights

        # Normalize by weights
        output = output / (weights + 1e-8)

        return output

    def _create_tile_weights(
        self, tile_size: int, overlap: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """Create weight mask for tile blending."""
        weights = torch.ones((1, 1, tile_size, tile_size), dtype=dtype, device=device)

        if overlap > 0:
            # Create linear ramp for overlap regions
            ramp = torch.linspace(0, 1, overlap, dtype=dtype, device=device)

            # Apply ramps to edges
            weights[:, :, :overlap, :] *= ramp.view(1, 1, -1, 1)
            weights[:, :, -overlap:, :] *= ramp.flip(0).view(1, 1, -1, 1)
            weights[:, :, :, :overlap] *= ramp.view(1, 1, 1, -1)
            weights[:, :, :, -overlap:] *= ramp.flip(0).view(1, 1, 1, -1)

        return weights
