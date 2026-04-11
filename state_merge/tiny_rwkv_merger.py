"""Tiny RWKV-like recurrent model for layer-wise WKV state merging.

A single shared RWKV-4 style cell (time-mixing WKV recurrence + channel-mixing
FFN) walks the RWKV-7 layer stack as if layers were timesteps. At each layer
ell it receives A_ell and B_ell (two pre-filled WKV matrix states) and emits a
merge mask that combines them:

    merged_ell = mask_ell * A_ell + (1 - mask_ell) * B_ell

Because the cell is shared and the heads run in a batched fashion, the total
parameter count is O(d_model**2) — a few tens of thousands of parameters
regardless of how many RWKV layers or heads the base model has. The only
shapes the model needs to know up front are heads_per_layer, H, W, and an
upper bound on num_layers (for the layer embedding table).

The training target is the "ground-truth" merged state obtained by pre-filling
context_A + context_B through the base RWKV model. This makes training a form
of feature-based knowledge distillation: the tiny model learns a cheap
approximation of what the full RWKV forward pass would produce when given the
concatenated context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TinyRWKVMergerConfig:
    d_model: int = 32
    d_ffn: int = 64
    max_layers: int = 128
    use_low_rank_mask: bool = True
    dropout: float = 0.0
    # Optional bias: how much to offset the initial gate. 0.0 → start at 0.5
    # (pure average), positive → start closer to A.
    init_gate_bias: float = 0.0


def _head_stats(x: torch.Tensor) -> torch.Tensor:
    """Compute per-head summary statistics over spatial (last 2) dims.

    x: [..., num_heads, H, W]  →  [..., num_heads, 4]  (mean, std, min, max).
    """
    flat = x.reshape(*x.shape[:-2], -1)
    mean = flat.mean(dim=-1)
    std = flat.std(dim=-1)
    mn = flat.amin(dim=-1)
    mx = flat.amax(dim=-1)
    return torch.stack([mean, std, mn, mx], dim=-1)


class TinyRWKVCell(nn.Module):
    """Shared RWKV-4-style time-mix + channel-mix block.

    The recurrent axis is the RWKV layer index (not the token index), so
    each "timestep" is one layer of the base model.
    """

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model

        # Time mixing (WKV recurrence over layer axis)
        self.ln_tm = nn.LayerNorm(d_model)
        self.time_decay = nn.Parameter(torch.zeros(d_model))
        self.time_first = nn.Parameter(torch.zeros(d_model))
        self.tm_r = nn.Linear(d_model, d_model, bias=False)
        self.tm_k = nn.Linear(d_model, d_model, bias=False)
        self.tm_v = nn.Linear(d_model, d_model, bias=False)
        self.tm_out = nn.Linear(d_model, d_model, bias=False)

        # Channel mixing (RWKV-4 squared-relu FFN)
        self.ln_cm = nn.LayerNorm(d_model)
        self.cm_k = nn.Linear(d_model, d_ffn, bias=False)
        self.cm_v = nn.Linear(d_ffn, d_model, bias=False)
        self.cm_r = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def init_state(
        self,
        batch_shape: tuple,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Allocate WKV recurrence state for shape `(*batch_shape, d_model)`."""
        shape = (*batch_shape, self.d_model)
        return {
            "aa": torch.zeros(shape, device=device, dtype=dtype),
            "bb": torch.zeros(shape, device=device, dtype=dtype),
            "pp": torch.full(shape, -1e30, device=device, dtype=dtype),
        }

    def time_mix_step(
        self, x: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """One WKV recurrence step.

        x: [..., d_model]
        state: dict of [..., d_model] running buffers (aa, bb, pp).
        """
        xn = self.ln_tm(x)
        r = torch.sigmoid(self.tm_r(xn))
        k = self.tm_k(xn)
        v = self.tm_v(xn)

        aa, bb, pp = state["aa"], state["bb"], state["pp"]

        # RWKV-4 numerically stable recurrence.
        ww = self.time_first + k
        p = torch.maximum(pp, ww)
        e1 = torch.exp(pp - p)
        e2 = torch.exp(ww - p)
        wkv = (e1 * aa + e2 * v) / (e1 * bb + e2).clamp_min(1e-8)

        ww2 = pp - torch.exp(self.time_decay)
        p2 = torch.maximum(ww2, k)
        e1 = torch.exp(ww2 - p2)
        e2 = torch.exp(k - p2)
        new_state = {
            "aa": e1 * aa + e2 * v,
            "bb": e1 * bb + e2,
            "pp": p2,
        }

        out = r * self.tm_out(wkv)
        return out, new_state

    def channel_mix_step(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.ln_cm(x)
        k = torch.square(torch.relu(self.cm_k(xn)))  # squared-relu
        v = self.cm_v(k)
        r = torch.sigmoid(self.cm_r(xn))
        return r * v

    def forward(
        self, x: torch.Tensor, state: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        tm_out, new_state = self.time_mix_step(x, state)
        x = x + self.dropout(tm_out)
        x = x + self.dropout(self.channel_mix_step(x))
        return x, new_state


class TinyRWKVStateMerger(nn.Module):
    """Tiny RWKV-like sequential state merger.

    Forward input: `x` of shape `[B, 2*N, H, W]` where
        N = num_layers * heads_per_layer
    and the channel layout is `[A_all_layer_major, B_all_layer_major]` —
    i.e. `x[:, :N]` holds every head of every layer of state A (layer-major),
    and `x[:, N:]` holds the same for state B.

    Forward output: dict with
        "mixed"      : [B, N, H, W] merged state (same layout).
        "mask_mean"  : scalar, mean of produced per-element masks (for logging).
        "gate_mean"  : scalar, mean of coarse per-head gates (for logging).

    This matches the `DynamicStateMixer` interface used by `HeadwiseStateMixer`
    and the SQuAD eval scripts, so it is a drop-in replacement.
    """

    def __init__(
        self,
        num_layers: int,
        heads_per_layer: int,
        height: int,
        width: int,
        config: Optional[TinyRWKVMergerConfig] = None,
    ):
        super().__init__()
        if num_layers < 1 or heads_per_layer < 1 or height < 1 or width < 1:
            raise ValueError(
                f"invalid shape: num_layers={num_layers}, heads_per_layer={heads_per_layer}, "
                f"h={height}, w={width}"
            )
        config = config or TinyRWKVMergerConfig()
        if num_layers > config.max_layers:
            raise ValueError(
                f"num_layers={num_layers} exceeds max_layers={config.max_layers}; "
                "raise TinyRWKVMergerConfig.max_layers."
            )

        self.config = config
        self.num_layers = num_layers
        self.heads_per_layer = heads_per_layer
        self.height = height
        self.width = width
        self.num_groups = num_layers * heads_per_layer

        d = config.d_model
        # Stats dim: 4 (A stats) + 4 (B stats) + 1 (diff mean) + 1 (diff std) + 1 (cosine) + 1 (l2 diff)
        self.stats_dim = 12
        self.in_proj = nn.Linear(self.stats_dim, d, bias=True)
        self.layer_emb = nn.Embedding(config.max_layers, d)

        self.cell = TinyRWKVCell(d, config.d_ffn, dropout=config.dropout)

        # Output heads. Coarse per-head scalar gate is always present; optional
        # low-rank additive spatial refinement gives per-element masks.
        self.gate_head = nn.Linear(d, 1)
        self.use_low_rank_mask = bool(config.use_low_rank_mask)
        if self.use_low_rank_mask:
            self.row_head = nn.Linear(d, height)
            self.col_head = nn.Linear(d, width)

        self._init_output_heads(config.init_gate_bias)

    def _init_output_heads(self, init_gate_bias: float) -> None:
        """Initialise output heads so the merger starts as a pure average."""
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, init_gate_bias)
        if self.use_low_rank_mask:
            nn.init.zeros_(self.row_head.weight)
            nn.init.zeros_(self.row_head.bias)
            nn.init.zeros_(self.col_head.weight)
            nn.init.zeros_(self.col_head.bias)

    @staticmethod
    def _extract_stats(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """a, b: [B, H_per, H, W] → stats [B, H_per, 12]."""
        a_stats = _head_stats(a)  # [B, H_per, 4]
        b_stats = _head_stats(b)  # [B, H_per, 4]

        diff = a - b
        diff_flat = diff.reshape(*diff.shape[:-2], -1)
        d_mean = diff_flat.mean(dim=-1, keepdim=True)
        d_std = diff_flat.std(dim=-1, keepdim=True)
        d_l2 = diff_flat.norm(dim=-1, keepdim=True)

        a_flat = a.reshape(*a.shape[:-2], -1)
        b_flat = b.reshape(*b.shape[:-2], -1)
        cos = F.cosine_similarity(a_flat, b_flat, dim=-1).unsqueeze(-1)

        return torch.cat([a_stats, b_stats, d_mean, d_std, d_l2, cos], dim=-1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D [B, 2N, H, W], got {tuple(x.shape)}")
        if x.size(2) != self.height or x.size(3) != self.width:
            raise ValueError(
                f"Spatial mismatch: got {(x.size(2), x.size(3))}, "
                f"expected {(self.height, self.width)}"
            )
        if x.size(1) != 2 * self.num_groups:
            raise ValueError(
                f"Channel mismatch: got {x.size(1)}, expected 2*{self.num_groups}"
            )

        bsz = x.size(0)
        h, w = self.height, self.width

        # Split A / B (layer-major) and reshape to per-layer view.
        a_all = x[:, : self.num_groups].view(
            bsz, self.num_layers, self.heads_per_layer, h, w
        )
        b_all = x[:, self.num_groups :].view(
            bsz, self.num_layers, self.heads_per_layer, h, w
        )

        device = x.device
        dtype = x.dtype
        state = self.cell.init_state(
            (bsz, self.heads_per_layer), device=device, dtype=dtype
        )

        mixed_layers = []
        gate_means: list[torch.Tensor] = []
        mask_means: list[torch.Tensor] = []

        for ell in range(self.num_layers):
            a_l = a_all[:, ell]  # [B, H_per, H, W]
            b_l = b_all[:, ell]

            stats = self._extract_stats(a_l, b_l)  # [B, H_per, 12]
            layer_idx = torch.tensor(ell, device=device)
            feat = self.in_proj(stats) + self.layer_emb(layer_idx)
            feat, state = self.cell(feat, state)  # [B, H_per, d]

            gate_logit = self.gate_head(feat)  # [B, H_per, 1]
            if self.use_low_rank_mask:
                row_bias = self.row_head(feat)  # [B, H_per, H]
                col_bias = self.col_head(feat)  # [B, H_per, W]
                mask_logit = (
                    gate_logit.unsqueeze(-1)  # [B, H_per, 1, 1]
                    + row_bias.unsqueeze(-1)  # [B, H_per, H, 1]
                    + col_bias.unsqueeze(-2)  # [B, H_per, 1, W]
                )  # [B, H_per, H, W]
            else:
                mask_logit = gate_logit.unsqueeze(-1).expand(-1, -1, h, w)

            mask = torch.sigmoid(mask_logit)
            merged = mask * a_l + (1.0 - mask) * b_l
            mixed_layers.append(merged)
            gate_means.append(torch.sigmoid(gate_logit).mean())
            mask_means.append(mask.mean())

        # [B, num_layers, H_per, H, W] → [B, num_groups, H, W]
        mixed = torch.stack(mixed_layers, dim=1).reshape(bsz, self.num_groups, h, w)

        return {
            "mixed": mixed,
            "gate_mean": torch.stack(gate_means).mean(),
            "mask_mean": torch.stack(mask_means).mean(),
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
