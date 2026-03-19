from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

StateLike = Union[torch.Tensor, List["StateLike"], Tuple["StateLike", ...]]


def _path_to_wkv_layer(path: Tuple[int, ...]) -> Union[int, None]:
    # RWKV7 state layout per layer:
    # idx*3+0: time-mixing shift vector
    # idx*3+1: matrix-valued WKV state
    # idx*3+2: channel-mixing shift vector
    if len(path) != 1:
        return None
    slot_idx = path[0]
    if slot_idx % 3 != 1:
        return None
    return slot_idx // 3


def _apply_state(fn, state: StateLike, path: Tuple[int, ...] = ()) -> StateLike:
    if torch.is_tensor(state):
        return fn(state, path)
    if isinstance(state, list):
        return [_apply_state(fn, x, path + (i,)) for i, x in enumerate(state)]
    if isinstance(state, tuple):
        return tuple(_apply_state(fn, x, path + (i,)) for i, x in enumerate(state))
    raise TypeError(f"Unsupported state type: {type(state)}")


def stack_states(states: Sequence[StateLike]) -> StateLike:
    if not states:
        raise ValueError("stack_states requires at least one state.")
    head = states[0]
    if torch.is_tensor(head):
        return torch.stack(list(states), dim=0)
    if isinstance(head, list):
        return [stack_states([s[i] for s in states]) for i in range(len(head))]
    if isinstance(head, tuple):
        return tuple(stack_states([s[i] for s in states]) for i in range(len(head)))
    raise TypeError(f"Unsupported state type: {type(head)}")


def unstack_states(batched_state: StateLike) -> List[StateLike]:
    if torch.is_tensor(batched_state):
        return list(torch.unbind(batched_state, dim=0))
    if isinstance(batched_state, list):
        parts = [unstack_states(x) for x in batched_state]
        return [[parts[i][j] for i in range(len(parts))] for j in range(len(parts[0]))]
    if isinstance(batched_state, tuple):
        parts = [unstack_states(x) for x in batched_state]
        return [tuple(parts[i][j] for i in range(len(parts))) for j in range(len(parts[0]))]
    raise TypeError(f"Unsupported state type: {type(batched_state)}")


def move_state_to_device(state: StateLike, device: torch.device) -> StateLike:
    return _apply_state(lambda x, _: x.to(device), state)


def move_state_to_cpu(state: StateLike) -> StateLike:
    return _apply_state(lambda x, _: x.detach().cpu(), state)


def clone_state(state: StateLike) -> StateLike:
    return _apply_state(lambda x, _: x.clone(), state)


def add_states(lhs: StateLike, rhs: StateLike) -> StateLike:
    if torch.is_tensor(lhs):
        return lhs + rhs
    if isinstance(lhs, list):
        return [add_states(lv, rv) for lv, rv in zip(lhs, rhs)]
    if isinstance(lhs, tuple):
        return tuple(add_states(lv, rv) for lv, rv in zip(lhs, rhs))
    raise TypeError(f"Unsupported state type: {type(lhs)}")


def mean_squared_error_state(pred: StateLike, target: StateLike) -> Tuple[torch.Tensor, float]:
    losses: List[torch.Tensor] = []

    def collect(lhs: StateLike, rhs: StateLike):
        if torch.is_tensor(lhs):
            losses.append(F.mse_loss(lhs, rhs))
            return
        if isinstance(lhs, list):
            for lv, rv in zip(lhs, rhs):
                collect(lv, rv)
            return
        if isinstance(lhs, tuple):
            for lv, rv in zip(lhs, rhs):
                collect(lv, rv)
            return
        raise TypeError(f"Unsupported state type: {type(lhs)}")

    collect(pred, target)
    if not losses:
        raise ValueError("No tensor leaves found in state.")
    loss = torch.stack(losses).mean()
    return loss, float(loss.detach().cpu().item())


@dataclass
class AutoEncoderConfig:
    dropout: float = 0.0
    expected_wkv_shape: Tuple[int, int, int] = (32, 64, 64)


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
        self.enc1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1)
        self.enc1_dilated = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2)
        self.enc1_spatial = SpatialAttention(kernel_size=7)

        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc2_dilated = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.enc2_spatial = SpatialAttention(kernel_size=7)

        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.enc3_dilated = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=2, dilation=2)
        self.enc3_spatial = SpatialAttention(kernel_size=7)

        # Deeper bottleneck block (keeps latent shape 512x4x4 with more non-linearity)
        self.bottleneck_down = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bottleneck_dilated = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2)
        self.bottleneck_spatial = SpatialAttention(kernel_size=7)
        self.bottleneck_refine1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bottleneck_refine2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        # Decoder with PixelShuffle upsampling.
        # Each stage: conv to (out_channels * 4) -> PixelShuffle(2) -> GELU -> SpatialAttention.
        self.dec1_expand = nn.Conv2d(512, 128 * 4, kernel_size=3, stride=1, padding=1)
        self.dec1_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.dec1_spatial = SpatialAttention(kernel_size=7)

        self.dec2_expand = nn.Conv2d(128, 64 * 4, kernel_size=3, stride=1, padding=1)
        self.dec2_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.dec2_spatial = SpatialAttention(kernel_size=7)

        self.dec3_expand = nn.Conv2d(64, in_channels * 4, kernel_size=3, stride=1, padding=1)
        self.dec3_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.dec3_spatial = SpatialAttention(kernel_size=7)

        self.dec4_expand = nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, stride=1, padding=1)
        self.dec4_shuffle = nn.PixelShuffle(upscale_factor=2)
        self.dec4_spatial = SpatialAttention(kernel_size=7)
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
        x = x.to(self.enc1.weight.dtype)
        z = self.enc1(x)
        z = F.gelu(z)
        z = self.enc1_dilated(z)
        z = F.gelu(z)
        z = self.enc1_spatial(z)
        z = self.enc2(z)
        z = F.gelu(z)
        z = self.enc2_dilated(z)
        z = F.gelu(z)
        z = self.enc2_spatial(z)
        z = self.enc3(z)
        z = F.gelu(z)
        z = self.enc3_dilated(z)
        z = F.gelu(z)
        z = self.enc3_spatial(z)
        z = self.bottleneck_down(z)
        z = F.gelu(z)
        z = self.bottleneck_dilated(z)
        z = F.gelu(z)
        z = self.bottleneck_spatial(z)
        z = self.bottleneck_refine1(z)
        z = F.gelu(z)
        z = self.bottleneck_refine2(z)
        z = F.gelu(z)
        z = self.dropout(z)
        if squeezed:
            z = z.squeeze(0)
        return z.to(in_dtype)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        in_dtype = z.dtype
        z, squeezed = self._to_4d(z)
        z = z.to(self.dec1_expand.weight.dtype)
        x = self.dec1_expand(z)
        x = self.dec1_shuffle(x)
        x = F.gelu(x)
        x = self.dec1_spatial(x)
        x = self.dec2_expand(x)
        x = self.dec2_shuffle(x)
        x = F.gelu(x)
        x = self.dec2_spatial(x)
        x = self.dec3_expand(x)
        x = self.dec3_shuffle(x)
        x = F.gelu(x)
        x = self.dec3_spatial(x)
        x = self.dec4_expand(x)
        x = self.dec4_shuffle(x)
        x = F.gelu(x)
        x = self.dec4_spatial(x)
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
            return module.encode(x)

        return _apply_state(_encoder, state)

    def decode_state(self, latent_state: StateLike) -> StateLike:
        def _decoder(z: torch.Tensor, path: Tuple[int, ...]) -> torch.Tensor:
            layer_idx = _path_to_wkv_layer(path)
            if layer_idx is None:
                return z
            module = self._module_from_layer(layer_idx)
            return module.decode(z)

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


def _as_batch_flat(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x.reshape(x.shape[0], -1)
    if x.ndim == 3:
        return x.reshape(1, -1)
    raise ValueError(f"WKV tensor must be 3D or 4D, got shape={tuple(x.shape)}")


def _as_batch_head_flat(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        b, c, h, w = x.shape
        return x.reshape(b, c, h * w)
    if x.ndim == 3:
        c, h, w = x.shape
        return x.reshape(1, c, h * w)
    raise ValueError(f"WKV tensor must be 3D or 4D, got shape={tuple(x.shape)}")


def peak_aware_wkv_loss(
    pred: StateLike,
    target: StateLike,
    topk_weight: float = 0.1,
    topk_ratio: float = 0.01,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    if not 0.0 <= topk_ratio <= 1.0:
        raise ValueError(f"topk_ratio must be in [0,1], got {topk_ratio}")

    mse_terms: List[torch.Tensor] = []
    topk_terms: List[torch.Tensor] = []

    def collect(lhs: StateLike, rhs: StateLike, path: Tuple[int, ...]) -> None:
        if torch.is_tensor(lhs):
            if _path_to_wkv_layer(path) is None:
                return
            mse_terms.append(F.mse_loss(lhs, rhs))

            lhs_head_flat = _as_batch_head_flat(lhs)
            rhs_head_flat = _as_batch_head_flat(rhs)
            if topk_ratio > 0:
                numel_per_head = rhs_head_flat.shape[-1]
                k = max(1, int(numel_per_head * topk_ratio))
                topk_idx = torch.topk(rhs_head_flat, k=k, dim=-1).indices
                pred_topk = torch.gather(lhs_head_flat, dim=-1, index=topk_idx)
                target_topk = torch.gather(rhs_head_flat, dim=-1, index=topk_idx)
                topk_terms.append(F.mse_loss(pred_topk, target_topk))
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
    if not mse_terms:
        raise ValueError("No WKV tensor leaves found in state.")

    mse_loss = torch.stack(mse_terms).mean()
    topk_loss = torch.stack(topk_terms).mean() if topk_terms else torch.zeros_like(mse_loss)
    total_loss = mse_loss + (topk_weight * topk_loss)

    return total_loss, {
        "total": float(total_loss.detach().cpu().item()),
        "mse": float(mse_loss.detach().cpu().item()),
        "topk_mse": float(topk_loss.detach().cpu().item()),
    }
