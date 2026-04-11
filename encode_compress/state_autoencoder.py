from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from state_utils import (  # noqa: F401  (re-exported for backwards compat)
    StateLike,
    _apply_state,
    _path_to_wkv_layer,
    add_states,
    clone_state,
    mean_squared_error_state,
    move_state_to_cpu,
    move_state_to_device,
    stack_states,
    unstack_states,
)


@dataclass
class AutoEncoderConfig:
    dropout: float = 0.0
    expected_wkv_shape: Tuple[int, int, int] = (32, 64, 64)


class CayleyOrthogonal(nn.Module):
    """
    Orthogonal matrix via Cayley transform: R = (I - A)(I + A)^{-1}
    with skew-symmetric A learned from unconstrained parameters P via A = (P - P^T) / 2.
    """

    def __init__(self, dim: int):
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.dim = dim
        self.p_raw = nn.Parameter(torch.zeros(dim, dim))

    def skew_a(self) -> torch.Tensor:
        p = self.p_raw.to(torch.float32)
        return 0.5 * (p - p.mT)

    def matrix_r(self) -> torch.Tensor:
        d = self.dim
        device = self.p_raw.device
        identity = torch.eye(d, device=device, dtype=torch.float32)
        a = self.skew_a()
        i_plus = identity + a
        i_minus = identity - a
        # R = (I - A)(I + A)^{-1}
        inv_factor = torch.linalg.solve(i_plus, identity)
        r = i_minus @ inv_factor
        return r


class WKVSpatialCayleyRotation(nn.Module):
    """
    Per-layer, per-head spatial rotation on WKV (each head = one [H,W] matrix):
    M'[h] = R1[h] @ M[h] @ R2[h]^T, inverse M[h] = R1[h]^T @ M'[h] @ R2[h] (R1[h], R2[h] orthogonal).
    """

    def __init__(self, num_heads: int, h: int, w: int):
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        self.num_heads = num_heads
        self.rot_row = nn.ModuleList([CayleyOrthogonal(h) for _ in range(num_heads)])
        self.rot_col = nn.ModuleList([CayleyOrthogonal(w) for _ in range(num_heads)])

    def _stack_r1(self) -> torch.Tensor:
        return torch.stack([m.matrix_r() for m in self.rot_row], dim=0)

    def _stack_r2(self) -> torch.Tensor:
        return torch.stack([m.matrix_r() for m in self.rot_col], dim=0)

    def rotate_forward(self, wkv: torch.Tensor) -> torch.Tensor:
        r1 = self._stack_r1().to(wkv.dtype)  # [C, H, H]
        r2 = self._stack_r2().to(wkv.dtype)  # [C, W, W]
        if wkv.ndim == 3:
            return torch.einsum("cik,ckl,cjl->cij", r1, wkv, r2)
        if wkv.ndim == 4:
            return torch.einsum("cik,bckl,cjl->bcij", r1, wkv, r2)
        raise ValueError(f"WKV tensor must be 3D or 4D, got shape={tuple(wkv.shape)}")

    def rotate_inverse(self, wkv: torch.Tensor) -> torch.Tensor:
        r1 = self._stack_r1().to(wkv.dtype)  # [C, H, H]
        r2 = self._stack_r2().to(wkv.dtype)  # [C, W, W]
        if wkv.ndim == 3:
            return torch.einsum("cki,ckl,clj->cij", r1, wkv, r2)
        if wkv.ndim == 4:
            return torch.einsum("cki,bckl,clj->bcij", r1, wkv, r2)
        raise ValueError(f"WKV tensor must be 3D or 4D, got shape={tuple(wkv.shape)}")


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError("kernel_size must be 3 or 7")
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.conv(attn)
        return x * self.sigmoid(attn)


class WKVConvAutoEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Encoder: downsample + dilated context + spatial attention
        self.encoder_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2),
                    nn.GELU(),
                    SpatialAttention(kernel_size=7),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                    nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2),
                    nn.GELU(),
                    SpatialAttention(kernel_size=7),
                ),
                nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                    nn.GELU(),
                    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2),
                    nn.GELU(),
                    SpatialAttention(kernel_size=7),
                ),
            ]
        )

        # Bottleneck keeps spatial size at 8x8 while expanding channels to 512.
        self.bottleneck_block = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.GELU(),
            SpatialAttention(kernel_size=7),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )

        # Decoder with PixelShuffle upsampling (8x8 -> 16x16 -> 32x32 -> 64x64).
        self.decoder_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(512, 128 * 4, kernel_size=3, stride=1, padding=1),
                    nn.PixelShuffle(upscale_factor=2),
                    nn.GELU(),
                    SpatialAttention(kernel_size=7),
                ),
                nn.Sequential(
                    nn.Conv2d(128, 64 * 4, kernel_size=3, stride=1, padding=1),
                    nn.PixelShuffle(upscale_factor=2),
                    nn.GELU(),
                    SpatialAttention(kernel_size=7),
                ),
                nn.Sequential(
                    nn.Conv2d(64, in_channels * 4, kernel_size=3, stride=1, padding=1),
                    nn.PixelShuffle(upscale_factor=2),
                    nn.GELU(),
                    SpatialAttention(kernel_size=7),
                ),
            ]
        )
        self.dec_out = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _to_4d(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if x.ndim == 4:
            return x, False
        if x.ndim == 3:
            return x.unsqueeze(0), True
        raise ValueError(f"WKV tensor must be 3D or 4D, got shape={tuple(x.shape)}")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x, squeezed = self._to_4d(x)
        x = x.to(self.encoder_blocks[0][0].weight.dtype)
        z = x
        for block in self.encoder_blocks:
            z = block(z)
        z = self.bottleneck_block(z)
        z = self.dropout(z)
        if squeezed:
            z = z.squeeze(0)
        return z.to(in_dtype)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        in_dtype = z.dtype
        z, squeezed = self._to_4d(z)
        z = z.to(self.decoder_blocks[0][0].weight.dtype)
        x = z
        for block in self.decoder_blocks:
            x = block(x)
        x = self.dec_out(x)
        if squeezed:
            x = x.squeeze(0)
        return x.to(in_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


class StateStructureAutoEncoder(nn.Module):
    def __init__(self, config: AutoEncoderConfig):
        super().__init__()
        self.config = config
        self.modules_by_layer = nn.ModuleDict()
        self.rotations_by_layer = nn.ModuleDict()

    def _ensure_rotation(self, layer_idx: int, num_heads: int, h: int, w: int) -> WKVSpatialCayleyRotation:
        module_key = f"layer_{layer_idx}"
        if module_key in self.rotations_by_layer:
            return self.rotations_by_layer[module_key]
        rot = WKVSpatialCayleyRotation(int(num_heads), int(h), int(w))
        self.rotations_by_layer[module_key] = rot
        return rot

    def _ensure_module(self, layer_idx: int, x: torch.Tensor) -> WKVConvAutoEncoder:
        module_key = f"layer_{layer_idx}"
        if module_key in self.modules_by_layer:
            return self.modules_by_layer[module_key]

        c, h, w = x.shape[-3:]
        expected = tuple(self.config.expected_wkv_shape)
        got = (int(c), int(h), int(w))
        if got != expected:
            raise ValueError(
                f"Unexpected WKV shape at layer {layer_idx}: got {got}, expected {expected}."
            )
        self._ensure_rotation(layer_idx, c, h, w)
        module = WKVConvAutoEncoder(in_channels=c, dropout=self.config.dropout)
        self.modules_by_layer[module_key] = module
        return module

    def _module_from_layer(self, layer_idx: int) -> WKVConvAutoEncoder:
        module_key = f"layer_{layer_idx}"
        if module_key not in self.modules_by_layer:
            raise KeyError(
                f"Layer {layer_idx} is unknown. Build modules with encode_state() or build_from_state() first."
            )
        return self.modules_by_layer[module_key]

    def build_from_state(self, state: StateLike) -> None:
        def _builder(x: torch.Tensor, path: Tuple[int, ...]) -> torch.Tensor:
            layer_idx = _path_to_wkv_layer(path)
            if layer_idx is not None:
                self._ensure_module(layer_idx, x)
            return x

        _apply_state(_builder, state)

    def encode_state(self, state: StateLike) -> StateLike:
        def _encoder(x: torch.Tensor, path: Tuple[int, ...]) -> torch.Tensor:
            layer_idx = _path_to_wkv_layer(path)
            if layer_idx is None:
                return x
            module = self._ensure_module(layer_idx, x)
            rot = self.rotations_by_layer[f"layer_{layer_idx}"]
            x_rot = rot.rotate_forward(x)
            return module.encode(x_rot)

        return _apply_state(_encoder, state)

    def decode_state(self, latent_state: StateLike) -> StateLike:
        def _decoder(z: torch.Tensor, path: Tuple[int, ...]) -> torch.Tensor:
            layer_idx = _path_to_wkv_layer(path)
            if layer_idx is None:
                return z
            module = self._module_from_layer(layer_idx)
            rot = self.rotations_by_layer[f"layer_{layer_idx}"]
            x_rot = module.decode(z)
            return rot.rotate_inverse(x_rot)

        return _apply_state(_decoder, latent_state)

    def forward_state(self, state: StateLike) -> StateLike:
        latent = self.encode_state(state)
        return self.decode_state(latent)


def mean_squared_error_wkv_only(pred: StateLike, target: StateLike) -> Tuple[torch.Tensor, float]:
    losses: List[torch.Tensor] = []

    def collect(lhs: StateLike, rhs: StateLike, path: Tuple[int, ...]) -> None:
        if torch.is_tensor(lhs):
            if _path_to_wkv_layer(path) is not None:
                losses.append(F.mse_loss(lhs, rhs))
            return
        if isinstance(lhs, list):
            for i, (lv, rv) in enumerate(zip(lhs, rhs)):
                collect(lv, rv, path + (i,))
            return
        if isinstance(lhs, tuple):
            for i, (lv, rv) in enumerate(zip(lhs, rhs)):
                collect(lv, rv, path + (i,))
            return
        raise TypeError(f"Unsupported state type: {type(lhs)}")

    collect(pred, target, ())
    if not losses:
        raise ValueError("No WKV tensor leaves found in state.")
    loss = torch.stack(losses).mean()
    return loss, float(loss.detach().cpu().item())


def rmse_plus_mae_wkv_only(pred: StateLike, target: StateLike) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Per-WKV-layer: RMSE = sqrt(MSE), MAE = mean |pred - target|.
    Total loss = mean(RMSE over layers) + mean(MAE over layers).
    """
    rmse_terms: List[torch.Tensor] = []
    mae_terms: List[torch.Tensor] = []

    def collect(lhs: StateLike, rhs: StateLike, path: Tuple[int, ...]) -> None:
        if torch.is_tensor(lhs):
            if _path_to_wkv_layer(path) is not None:
                mse = F.mse_loss(lhs, rhs)
                rmse_terms.append(torch.sqrt(mse + 1e-24))
                mae_terms.append(F.l1_loss(lhs, rhs))
            return
        if isinstance(lhs, list):
            for i, (lv, rv) in enumerate(zip(lhs, rhs)):
                collect(lv, rv, path + (i,))
            return
        if isinstance(lhs, tuple):
            for i, (lv, rv) in enumerate(zip(lhs, rhs)):
                collect(lv, rv, path + (i,))
            return
        raise TypeError(f"Unsupported state type: {type(lhs)}")

    collect(pred, target, ())
    if not rmse_terms:
        raise ValueError("No WKV tensor leaves found in state.")

    rmse_mean = torch.stack(rmse_terms).mean()
    mae_mean = torch.stack(mae_terms).mean()
    loss = rmse_mean + mae_mean
    return loss, {
        "total": float(loss.detach().cpu().item()),
        "rmse": float(rmse_mean.detach().cpu().item()),
        "mae": float(mae_mean.detach().cpu().item()),
    }
