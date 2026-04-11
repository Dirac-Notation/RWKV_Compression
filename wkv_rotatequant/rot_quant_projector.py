from __future__ import annotations

import os
import sys
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from state_utils import (  # noqa: F401  (re-exported for sibling modules)
    StateLike,
    _apply_state,
    _path_to_wkv_layer,
    move_state_to_device,
    stack_states,
)


class STEQuantizer:
    @staticmethod
    def quant_dequant(x: torch.Tensor, bits: int) -> torch.Tensor:
        if bits >= 16:
            return x
        q_levels = (1 << bits) - 1
        if q_levels <= 0:
            return x
        x_fp32 = x.float()
        x_min = x_fp32.amin()
        x_max = x_fp32.amax()
        span = (x_max - x_min).clamp_min(1e-12)
        scale = span / float(q_levels)
        q = torch.round((x_fp32 - x_min) / scale).clamp(0, float(q_levels))
        dq = q * scale + x_min
        dq = dq.to(dtype=x.dtype)
        return x + (dq - x).detach()


class CayleyOrthogonal(nn.Module):
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
        inv_factor = torch.linalg.solve(i_plus, identity)
        r = i_minus @ inv_factor
        return r


class BilateralRotation(nn.Module):
    def __init__(self, num_heads: int, h: int, w: int):
        super().__init__()
        self.rot_left = nn.ModuleList([CayleyOrthogonal(h) for _ in range(num_heads)])
        self.rot_right = nn.ModuleList([CayleyOrthogonal(w) for _ in range(num_heads)])

    def _stack_r1(self, dtype: torch.dtype) -> torch.Tensor:
        return torch.stack([m.matrix_r() for m in self.rot_left], dim=0).to(dtype=dtype)

    def _stack_r2(self, dtype: torch.dtype) -> torch.Tensor:
        return torch.stack([m.matrix_r() for m in self.rot_right], dim=0).to(dtype=dtype)

    def forward_rotate(self, wkv: torch.Tensor) -> torch.Tensor:
        r1 = self._stack_r1(wkv.dtype)
        r2 = self._stack_r2(wkv.dtype)
        if wkv.ndim == 3:
            return torch.einsum("cik,ckl,clj->cij", r1, wkv, r2)
        if wkv.ndim == 4:
            return torch.einsum("cik,bckl,clj->bcij", r1, wkv, r2)
        raise ValueError(f"WKV tensor must be 3D or 4D, got {tuple(wkv.shape)}")

    def inverse_rotate(self, wkv: torch.Tensor) -> torch.Tensor:
        r1 = self._stack_r1(wkv.dtype)
        r2 = self._stack_r2(wkv.dtype)
        if wkv.ndim == 3:
            return torch.einsum("cki,ckl,cjl->cij", r1, wkv, r2)
        if wkv.ndim == 4:
            return torch.einsum("cki,bckl,cjl->bcij", r1, wkv, r2)
        raise ValueError(f"WKV tensor must be 3D or 4D, got {tuple(wkv.shape)}")


class RotationQuantProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.rotations_by_layer = nn.ModuleDict()

    def _ensure_layer_rotation(self, layer_idx: int, x: torch.Tensor) -> BilateralRotation:
        key = f"layer_{layer_idx}"
        if key in self.rotations_by_layer:
            return self.rotations_by_layer[key]
        c, h, w = x.shape[-3:]
        mod = BilateralRotation(int(c), int(h), int(w))
        self.rotations_by_layer[key] = mod
        return mod

    def build_from_state(self, state: StateLike) -> None:
        def _builder(x: torch.Tensor, path: Tuple[int, ...]) -> torch.Tensor:
            layer_idx = _path_to_wkv_layer(path)
            if layer_idx is not None:
                self._ensure_layer_rotation(layer_idx, x)
            return x

        _apply_state(_builder, state)

    def transform_state(self, state: StateLike, bits: int) -> StateLike:
        def _transform(x: torch.Tensor, path: Tuple[int, ...]) -> torch.Tensor:
            layer_idx = _path_to_wkv_layer(path)
            if layer_idx is None:
                return x
            rot = self._ensure_layer_rotation(layer_idx, x)
            x_rot = rot.forward_rotate(x)
            x_q = STEQuantizer.quant_dequant(x_rot, bits=bits)
            x_rec = rot.inverse_rotate(x_q)
            return x_rec

        return _apply_state(_transform, state)


def wkv_only_mse(pred: StateLike, target: StateLike) -> torch.Tensor:
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
    return torch.stack(losses).mean()
