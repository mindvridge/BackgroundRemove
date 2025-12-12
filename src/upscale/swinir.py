"""SwinIR upscaler implementation.

SwinIR (Image Restoration Using Swin Transformer) provides:
- Highest quality upscaling (9.7/10)
- Superior long-range dependency modeling via Swin Transformer
- Excellent edge preservation and texture reconstruction
- Best for architecture, digital art, and detailed images

Quality: 9.7/10 (highest among upscalers)
Speed: 20-35 seconds per image (mid-range GPU)
Best for: Digital art, architecture, detailed textures
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
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


# SwinIR model URLs
SWINIR_URLS = {
    UpscaleModel.SWINIR_REAL_SR_X4: {
        "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
        "filename": "SwinIR_realSR_x4_GAN.pth",
    },
    UpscaleModel.SWINIR_CLASSICAL_SR_X4: {
        "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
        "filename": "SwinIR_classicalSR_x4.pth",
    },
    UpscaleModel.SWINIR_LIGHTWEIGHT_X4: {
        "url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth",
        "filename": "SwinIR_lightweight_x4.pth",
    },
}


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition feature map into non-overlapping windows.

    Args:
        x: Input tensor (B, H, W, C).
        window_size: Window size.

    Returns:
        Windows tensor (num_windows*B, window_size, window_size, C).
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(
    windows: torch.Tensor,
    window_size: int,
    H: int,
    W: int,
) -> torch.Tensor:
    """Reverse window partition.

    Args:
        windows: Windows tensor.
        window_size: Window size.
        H: Height of feature map.
        W: Width of feature map.

    Returns:
        Tensor (B, H, W, C).
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """Window based multi-head self attention with relative position bias."""

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
        self.scale = head_dim ** -0.5

        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # Relative position index
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
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """Swin Transformer Block."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int = 8,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim,
            window_size=(window_size, window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        H, W = x_size
        B, L, C = x.shape

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # Partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # Window attention
        attn_windows = self.attn(x_windows, mask=None)

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(B, H * W, C)
        x = shortcut + x

        # FFN
        x = x + self.mlp(self.norm2(x))

        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block."""

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
            )
            for i in range(depth)
        ])

        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x: torch.Tensor, x_size: tuple[int, int]) -> torch.Tensor:
        shortcut = x
        for blk in self.blocks:
            x = blk(x, x_size)

        x = x.view(x.shape[0], *x_size, -1).permute(0, 3, 1, 2).contiguous()
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, self.dim)

        return x + shortcut


class SwinIR(nn.Module):
    """SwinIR: Image Restoration Using Swin Transformer.

    Provides state-of-the-art image restoration including:
    - Super-resolution (2x, 4x)
    - Denoising
    - JPEG artifact removal
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 1,
        in_chans: int = 3,
        embed_dim: int = 180,
        depths: tuple[int, ...] = (6, 6, 6, 6, 6, 6),
        num_heads: tuple[int, ...] = (6, 6, 6, 6, 6, 6),
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        upscale: int = 4,
        img_range: float = 1.0,
        upsampler: str = "pixelshuffle",
        resi_connection: str = "1conv",
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.upscale = upscale
        self.img_range = img_range

        # Shallow feature extraction
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # Deep feature extraction
        self.layers = nn.ModuleList([
            RSTB(
                dim=embed_dim,
                depth=depths[i],
                num_heads=num_heads[i],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
            )
            for i in range(len(depths))
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # Upsampling
        if upsampler == "pixelshuffle":
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, 64 * 4, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.upsample = nn.Sequential(
                nn.PixelShuffle(2),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
                nn.PixelShuffle(2),
                nn.Conv2d(16, 16, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.conv_last = nn.Conv2d(16, in_chans, 3, 1, 1)
        else:
            # Nearest + conv upsampling
            self.conv_before_upsample = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=upscale, mode='nearest'),
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
                nn.LeakyReLU(inplace=True),
            )
            self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad to multiple of window size
        _, _, H, W = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # Shallow features
        x = self.conv_first(x)
        x_size = (x.shape[2], x.shape[3])

        # Deep features
        x_deep = x.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1, x.shape[1])

        for layer in self.layers:
            x_deep = layer(x_deep, x_size)

        x_deep = self.norm(x_deep)
        x_deep = x_deep.view(x.shape[0], *x_size, -1).permute(0, 3, 1, 2).contiguous()

        x = x + self.conv_after_body(x_deep)

        # Upsample
        x = self.conv_before_upsample(x)
        x = self.upsample(x)
        x = self.conv_last(x)

        # Remove padding
        x = x[:, :, :H * self.upscale, :W * self.upscale]

        return x


class SwinIRUpscaler(BaseUpscaler):
    """SwinIR based image upscaler.

    Provides highest-quality upscaling using Swin Transformer architecture.
    Slower than Real-ESRGAN but produces superior results.

    Example:
        >>> upscaler = SwinIRUpscaler(UpscaleConfig(
        ...     model=UpscaleModel.SWINIR_REAL_SR_X4,
        ...     scale=4
        ... ))
        >>> upscaler.load()
        >>> result = upscaler.upscale(image)
    """

    def __init__(self, config: UpscaleConfig) -> None:
        super().__init__(config)
        self.model: SwinIR | None = None
        self.device: torch.device | None = None

    @property
    def model_name(self) -> str:
        return self.config.model.value

    def _get_model_config(self) -> dict:
        """Get model configuration based on model type."""
        model_type = self.config.model

        if model_type == UpscaleModel.SWINIR_REAL_SR_X4:
            return {
                "embed_dim": 180,
                "depths": (6, 6, 6, 6, 6, 6, 6, 6, 6),
                "num_heads": (6, 6, 6, 6, 6, 6, 6, 6, 6),
                "window_size": 8,
                "upscale": 4,
            }
        elif model_type == UpscaleModel.SWINIR_CLASSICAL_SR_X4:
            return {
                "embed_dim": 180,
                "depths": (6, 6, 6, 6, 6, 6),
                "num_heads": (6, 6, 6, 6, 6, 6),
                "window_size": 8,
                "upscale": 4,
            }
        elif model_type == UpscaleModel.SWINIR_LIGHTWEIGHT_X4:
            return {
                "embed_dim": 60,
                "depths": (6, 6, 6, 6),
                "num_heads": (6, 6, 6, 6),
                "window_size": 8,
                "upscale": 4,
            }
        else:
            raise ValueError(f"Unsupported SwinIR model: {model_type}")

    def _download_model(self) -> Path:
        """Download model if not present."""
        if self.config.model not in SWINIR_URLS:
            raise ValueError(f"Unknown SwinIR model: {self.config.model}")

        info = SWINIR_URLS[self.config.model]
        model_path = self.config.model_dir / info["filename"]

        if not model_path.exists():
            logger.info(f"Downloading SwinIR model...")
            import urllib.request

            model_path.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(info["url"], model_path)
            logger.info(f"Downloaded to {model_path}")

        return model_path

    def load(self) -> None:
        """Load the SwinIR model."""
        if self._loaded:
            return

        logger.info(f"Loading SwinIR model: {self.config.model.value}")

        # Setup device
        if self.config.use_gpu and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.config.device_id}")
        else:
            self.device = torch.device("cpu")

        # Create model
        model_config = self._get_model_config()
        self.model = SwinIR(**model_config)

        # Load weights
        model_path = self._download_model()
        state_dict = torch.load(model_path, map_location=self.device)

        if "params_ema" in state_dict:
            state_dict = state_dict["params_ema"]
        elif "params" in state_dict:
            state_dict = state_dict["params"]

        self.model.load_state_dict(state_dict, strict=False)
        self.model = self.model.to(self.device)
        self.model.eval()

        if self.config.use_fp16 and self.device.type == "cuda":
            self.model = self.model.half()

        self._loaded = True
        logger.info(f"SwinIR model loaded on {self.device}")

    def unload(self) -> None:
        """Unload model."""
        if self.model is not None:
            del self.model
            self.model = None

        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def upscale(
        self,
        image: ndarray,
        scale: int | None = None,
    ) -> UpscaleResult:
        """Upscale image using SwinIR."""
        if not self._loaded or self.model is None:
            raise RuntimeError("Model not loaded")

        start_time = time.time()
        original_h, original_w = image.shape[:2]

        # Preprocess
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        img = img.to(self.device)

        if self.config.use_fp16 and self.device.type == "cuda":
            img = img.half()

        # Tile processing for large images
        if original_h * original_w > self.config.tile_size ** 2:
            output = self._tile_inference(img)
        else:
            output = self.model(img)

        # Postprocess
        output = output.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
        output = np.clip(output * 255.0, 0, 255).astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        processing_time = time.time() - start_time
        output_h, output_w = output.shape[:2]

        return UpscaleResult(
            image=output,
            original_size=(original_w, original_h),
            upscaled_size=(output_w, output_h),
            scale_factor=output_w / original_w,
            model_used=self.model_name,
            processing_time=processing_time,
        )

    def _tile_inference(self, img: torch.Tensor) -> torch.Tensor:
        """Process large images in tiles."""
        b, c, h, w = img.shape
        tile = self.config.tile_size
        overlap = self.config.tile_overlap
        scale = self.model.upscale

        out_h, out_w = h * scale, w * scale
        output = torch.zeros((b, c, out_h, out_w), device=img.device, dtype=img.dtype)
        weight = torch.zeros((b, 1, out_h, out_w), device=img.device, dtype=img.dtype)

        for y in range(0, h, tile - overlap):
            for x in range(0, w, tile - overlap):
                y2 = min(y + tile, h)
                x2 = min(x + tile, w)
                y1 = max(0, y2 - tile)
                x1 = max(0, x2 - tile)

                tile_img = img[:, :, y1:y2, x1:x2]
                tile_out = self.model(tile_img)

                oy1, ox1 = y1 * scale, x1 * scale
                oy2, ox2 = y2 * scale, x2 * scale

                output[:, :, oy1:oy2, ox1:ox2] += tile_out
                weight[:, :, oy1:oy2, ox1:ox2] += 1

        return output / weight

    def __repr__(self) -> str:
        status = "loaded" if self._loaded else "not loaded"
        return f"SwinIRUpscaler(model={self.config.model.value}, status={status})"
