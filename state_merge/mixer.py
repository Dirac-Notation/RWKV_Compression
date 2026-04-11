import math
import os
import sys
from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from state_utils import _is_wkv_path, move_state_to_device  # noqa: F401


# ---------------------------------------------------------------------------
# Per-head summary statistics
# ---------------------------------------------------------------------------

def _head_stats_pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-head joint stats of two WKV matrices.

    a, b: [B, H_per, H, W]  →  [B, H_per, 12]
    Layout: 4 stats (mean/std/min/max) for A, 4 for B, then
    diff_mean / diff_std / diff_l2 / cosine(A, B).
    """
    a_flat = a.reshape(*a.shape[:-2], -1)
    b_flat = b.reshape(*b.shape[:-2], -1)

    a_mean, a_std = a_flat.mean(-1), a_flat.std(-1)
    a_min, a_max = a_flat.amin(-1), a_flat.amax(-1)
    b_mean, b_std = b_flat.mean(-1), b_flat.std(-1)
    b_min, b_max = b_flat.amin(-1), b_flat.amax(-1)

    diff = a_flat - b_flat
    d_mean = diff.mean(-1)
    d_std = diff.std(-1)
    d_l2 = diff.norm(dim=-1)
    cos = F.cosine_similarity(a_flat, b_flat, dim=-1)

    return torch.stack(
        [a_mean, a_std, a_min, a_max, b_mean, b_std, b_min, b_max, d_mean, d_std, d_l2, cos],
        dim=-1,
    )


# ---------------------------------------------------------------------------
# MatrixMergerBlock — replaces the old _LayerMixerBlock
# ---------------------------------------------------------------------------

class MatrixMergerBlock(nn.Module):
    """One per-layer merger block, matrix-aware (NOT convolutional).

    Operates on a single layer's two WKV matrices `(A_ell, B_ell)` of shape
    `[B, H_per, H, W]` and emits a merged tensor of the same shape. The
    architecture treats H, W as feature dimensions of a matrix-valued state,
    not as spatial axes of an image — there are no convolutions on the H/W
    plane (those would impose a spurious locality prior).

    Per-head feature pipeline:
        1. Compute 12-dim per-head joint stats of (A, B).
        2. Project to `d_model`, add a per-layer embedding (set externally).
        3. Cross-head Multi-Head Self-Attention (heads talk to each other).
        4. Channel-mixing FFN.

    Output decoder (per-element merge mask via low-rank factors):
        mask_logit[h, i, j] = scalar_gate[h] + row_factor[h, i] + col_factor[h, j]
        mask = sigmoid(mask_logit) ∈ [0, 1]^{H_per × H × W}
        merged = mask * A + (1 - mask) * B + delta
    where `delta = scale * u_row · u_col^T` is a tiny low-rank residual that
    lets the model escape the convex-combination corner case (init scale=0).

    All output heads are zero-init so the merger starts as a pure average.
    """

    def __init__(
        self,
        heads_per_layer: int,
        height: int,
        width: int,
        d_model: int = 32,
        n_attn_heads: int = 2,
        d_ffn: int | None = None,
        delta_rank: int = 1,
    ):
        super().__init__()
        if heads_per_layer < 1:
            raise ValueError(f"heads_per_layer must be >= 1, got {heads_per_layer}")
        if d_model % n_attn_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_attn_heads={n_attn_heads}"
            )
        if delta_rank < 1:
            raise ValueError(f"delta_rank must be >= 1, got {delta_rank}")
        self.heads_per_layer = heads_per_layer
        self.height = height
        self.width = width
        self.d_model = d_model
        self.delta_rank = delta_rank
        d_ffn = d_ffn or 2 * d_model
        self.stats_dim = 12

        self.in_proj = nn.Linear(self.stats_dim, d_model)

        # Cross-head transformer block (heads as the "sequence").
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_attn_heads, batch_first=True, bias=True
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Linear(d_ffn, d_model),
        )

        # Mask output (low-rank per-element via row + col factors).
        self.gate_head = nn.Linear(d_model, 1)
        self.row_head = nn.Linear(d_model, height)
        self.col_head = nn.Linear(d_model, width)

        # Non-convex residual: rank-K decomposition delta = sum_k u_k · v_k^T.
        self.delta_row_head = nn.Linear(d_model, height * delta_rank)
        self.delta_col_head = nn.Linear(d_model, width * delta_rank)
        self.delta_scale = nn.Parameter(torch.zeros(()))

        self._init_output_heads()

    def _init_output_heads(self) -> None:
        # Mask path: zero everything so init mask = sigmoid(0) = 0.5 → pure avg.
        for head in (self.gate_head, self.row_head, self.col_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        # Delta path: zero weight + non-zero bias on each factor, and zero scale.
        # The product `delta_row · delta_col^T` is then a non-zero constant matrix
        # (so `∂loss/∂delta_scale` is non-zero from step 0), but `delta_scale=0`
        # keeps the actual `delta` exactly zero at init. Once scale moves, the
        # row/col weights start receiving non-zero gradients too.
        nn.init.zeros_(self.delta_row_head.weight)
        nn.init.constant_(self.delta_row_head.bias, 0.5)
        nn.init.zeros_(self.delta_col_head.weight)
        nn.init.constant_(self.delta_col_head.bias, 0.5)
        # delta_scale already zero-init via the constructor

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        layer_emb: torch.Tensor | None = None,
    ) -> dict:
        """a, b: [B, H_per, H, W]; layer_emb: [d_model] or None."""
        if a.shape != b.shape:
            raise ValueError(f"a/b shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
        if a.dim() != 4:
            raise ValueError(f"Expected 4D [B, H_per, H, W], got {tuple(a.shape)}")
        if a.size(2) != self.height or a.size(3) != self.width:
            raise ValueError(
                f"Spatial mismatch: got {(a.size(2), a.size(3))}, "
                f"expected {(self.height, self.width)}"
            )
        if a.size(1) != self.heads_per_layer:
            raise ValueError(
                f"Head count mismatch: got {a.size(1)}, expected {self.heads_per_layer}"
            )

        bsz = a.size(0)
        h, w = self.height, self.width

        # 1) Per-head joint stats → d_model features.
        stats = _head_stats_pair(a, b)  # [B, H_per, 12]
        feat = self.in_proj(stats)  # [B, H_per, d_model]
        if layer_emb is not None:
            feat = feat + layer_emb  # broadcast [d_model] over [B, H_per, d_model]

        # 2) Cross-head attention (heads as the seq dim — permutation-equivariant).
        attn_in = self.attn_norm(feat)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        feat = feat + attn_out

        # 3) Channel-mixing FFN.
        feat = feat + self.ffn(self.ffn_norm(feat))  # [B, H_per, d_model]

        # 4) Decode mask + residual.
        gate = self.gate_head(feat)  # [B, H_per, 1]
        row = self.row_head(feat)  # [B, H_per, H]
        col = self.col_head(feat)  # [B, H_per, W]
        mask_logit = (
            gate.unsqueeze(-1)  # [B, H_per, 1, 1]
            + row.unsqueeze(-1)  # [B, H_per, H, 1]
            + col.unsqueeze(-2)  # [B, H_per, 1, W]
        )  # [B, H_per, H, W]
        mask = torch.sigmoid(mask_logit)

        # Rank-K residual delta = scale * sum_k row_k · col_k^T.
        bsz_local, hp_local = feat.size(0), feat.size(1)
        d_row = self.delta_row_head(feat).view(bsz_local, hp_local, self.delta_rank, h)
        d_col = self.delta_col_head(feat).view(bsz_local, hp_local, self.delta_rank, w)
        delta = self.delta_scale * torch.einsum("bhki,bhkj->bhij", d_row, d_col)  # [B, H_per, H, W]

        merged = mask * a + (1.0 - mask) * b + delta

        return {
            "mixed": merged,
            "mask": mask,
            "delta": delta,
            "feat": feat,
        }


# Backwards-compat alias — older imports / pickles refer to `_LayerMixerBlock`.
_LayerMixerBlock = MatrixMergerBlock


class DynamicStateMixer(nn.Module):
    """Per-layer matrix-aware state merger (replaces the conv-based design).

    Forward input: `x` of shape `[B, 2*N, H, W]` where
        N = num_layers * heads_per_layer
    and the channel layout is `[A_all_layer_major, B_all_layer_major]`.

    For each RWKV layer ell, the mixer
        1. slices `(A_ell, B_ell)` out of `x`,
        2. fetches a learnable layer embedding,
        3. runs them through a *shared* `MatrixMergerBlock`,
    and stacks the per-layer outputs back into `[B, N, H, W]`.

    Compared to the previous convolutional design this:
      - No longer pretends WKV matrices are images (no Conv2d on H, W).
      - Lets every head talk to every other head in the same layer via
        attention (the old design had no cross-head info flow at all).
      - Uses a *shared* per-layer block, distinguished only by a layer
        embedding, which collapses parameter count from O(L · d²) to O(d²).
      - Produces per-element masks via low-rank row+col factors instead of
        the old per-head scalar gate.
      - Has an additive low-rank residual `delta` so the merged value is no
        longer constrained to live element-wise between A and B.
      - Drops the single-step `prev_AB` recurrence (which was too weak to
        capture depth-dependent merging strategies); see `tiny_rwkv_merger`
        for the recurrent variant.

    The legacy `prev_AB` carry-over and per-layer block buggy channel
    interleave that the old conv design had are gone — there is no
    `torch.cat([A, B, prev], dim=1)` step that needed careful interpretation,
    so the corresponding view bug cannot recur.
    """

    def __init__(
        self,
        num_layers: int,
        heads_per_layer: int,
        height: int,
        width: int,
        d_model: int = 32,
        n_attn_heads: int = 2,
        d_ffn: int | None = None,
        delta_rank: int = 1,
        max_layers: int = 128,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if heads_per_layer < 1:
            raise ValueError(f"heads_per_layer must be >= 1, got {heads_per_layer}")
        if height < 1 or width < 1:
            raise ValueError(f"height/width must be >= 1, got {(height, width)}")
        if num_layers > max_layers:
            raise ValueError(
                f"num_layers={num_layers} exceeds max_layers={max_layers}; "
                "raise the constructor arg."
            )

        self.num_layers = num_layers
        self.heads_per_layer = heads_per_layer
        self.height = height
        self.width = width
        self.num_groups = num_layers * heads_per_layer

        self.layer_emb = nn.Embedding(max_layers, d_model)
        self.block = MatrixMergerBlock(
            heads_per_layer=heads_per_layer,
            height=height,
            width=width,
            d_model=d_model,
            n_attn_heads=n_attn_heads,
            d_ffn=d_ffn,
            delta_rank=delta_rank,
        )

    def forward(self, x: torch.Tensor) -> dict:
        if x.dim() != 4:
            raise ValueError(f"Expected 4D [B, 2*N, H, W], got {tuple(x.shape)}")
        n_total = self.num_groups
        if x.size(1) != 2 * n_total:
            raise ValueError(f"Expected channel size 2*N={2 * n_total}, got {x.size(1)}")
        if x.size(2) != self.height or x.size(3) != self.width:
            raise ValueError(
                f"Expected spatial size {(self.height, self.width)}, got {(x.size(2), x.size(3))}"
            )

        bsz = x.size(0)
        h, w = self.height, self.width

        # Layer-major split.
        a_all = x[:, :n_total].view(bsz, self.num_layers, self.heads_per_layer, h, w)
        b_all = x[:, n_total:].view(bsz, self.num_layers, self.heads_per_layer, h, w)

        mixed_layers: List[torch.Tensor] = []
        mask_means: List[torch.Tensor] = []
        for ell in range(self.num_layers):
            a_l = a_all[:, ell]
            b_l = b_all[:, ell]
            le = self.layer_emb(torch.tensor(ell, device=x.device))
            out = self.block(a_l, b_l, layer_emb=le)
            mixed_layers.append(out["mixed"])
            mask_means.append(out["mask"].mean())

        mixed = torch.stack(mixed_layers, dim=1).view(bsz, n_total, h, w)
        return {
            "mixed": mixed,
            "mask_mean": torch.stack(mask_means).mean(),
        }


# Backward compatibility for older imports/scripts.
PointwiseKernelMixer = DynamicStateMixer


def _count_layers_heads_spatial(state: Any) -> tuple[int, int, int, int]:
    layer_specs: List[tuple[int, int, int]] = []

    def walk(x: Any, path: List[int]):
        if torch.is_tensor(x):
            if _is_wkv_path(path):
                if x.ndim < 2:
                    raise ValueError(f"WKV tensor must be >=2D, got shape={tuple(x.shape)} at path={path}")
                heads = int(math.prod(x.shape[:-2]))
                h, w = int(x.shape[-2]), int(x.shape[-1])
                layer_specs.append((heads, h, w))
            return
        if isinstance(x, (list, tuple)):
            for i, item in enumerate(x):
                walk(item, path + [i])
            return
        raise TypeError(f"Unsupported state type: {type(x)}")

    walk(state, [])
    if not layer_specs:
        raise ValueError("State has no WKV tensor leaves.")
    heads0, h0, w0 = layer_specs[0]
    for i, (heads, h, w) in enumerate(layer_specs):
        if heads != heads0 or h != h0 or w != w0:
            raise ValueError(
                f"Inconsistent WKV layout across layers: layer0={layer_specs[0]} vs layer{i}={(heads, h, w)}"
            )
    return len(layer_specs), heads0, h0, w0


def count_layers_heads_from_state(state: Any) -> tuple[int, int, int, int]:
    """Return (num_layers, heads_per_layer, height, width) from an RWKV state tree."""
    return _count_layers_heads_spatial(state)


class HeadwiseStateMixer(nn.Module):
    """Tree-aware wrapper around a per-layer state merger.

    The inner merger class (`mixer_cls`) must accept
    `(num_layers, heads_per_layer, height, width, **mixer_kwargs)` in its
    constructor and expose `forward(x: [B, 2*N, H, W]) -> dict` with a
    `"mixed"` key of shape `[B, N, H, W]`. The layout of the channel
    dimension is `[A_all_layer_major, B_all_layer_major]`.

    Defaults to `DynamicStateMixer` (the convolutional merger) but any
    compatible module works — see `tiny_rwkv_merger.TinyRWKVStateMerger`
    for the tiny RWKV-like variant.
    """

    def __init__(
        self,
        mse_weight: float = 1.0,
        l1_weight: float = 0.2,
        cosine_weight: float = 0.2,
        rec_tol: float = 1e-3,
        mixer_cls: type[nn.Module] | None = None,
        mixer_kwargs: dict | None = None,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.cosine_weight = cosine_weight
        self.rec_tol = rec_tol
        self._num_units = 0
        self._num_layers = 0
        self._heads_per_layer = 0
        self._mixer: nn.Module | None = None
        self._built = False
        self._mixer_cls: type[nn.Module] = mixer_cls or DynamicStateMixer
        self._mixer_kwargs: dict = dict(mixer_kwargs or {})

    @property
    def num_groups(self) -> int:
        return self._num_units

    @property
    def inner_mixer(self) -> nn.Module | None:
        return self._mixer

    def build_from_state(self, state: Any):
        if self._built:
            return
        num_layers, heads, h, w = _count_layers_heads_spatial(state)
        self._num_layers = num_layers
        self._heads_per_layer = heads
        self._num_units = num_layers * heads
        if self._num_units < 1:
            raise ValueError("State has no valid tensor leaves.")
        self._mixer = self._mixer_cls(num_layers, heads, h, w, **self._mixer_kwargs)
        self._built = True

    def _tensor_loss(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        mse = F.mse_loss(pred, tgt)
        l1 = F.l1_loss(pred, tgt)
        pred_flat = pred.reshape(1, -1)
        tgt_flat = tgt.reshape(1, -1)
        cos = 1.0 - F.cosine_similarity(pred_flat, tgt_flat, dim=1).mean()
        return self.mse_weight * mse + self.l1_weight * l1 + self.cosine_weight * cos

    def _scatter_mixed(self, lhs: Any, mixed_flat: torch.Tensor, offset: List[int], path: List[int]) -> Any:
        if torch.is_tensor(lhs):
            if _is_wkv_path(path):
                if lhs.ndim < 2:
                    raise ValueError(f"WKV tensor must be >=2D, got shape={tuple(lhs.shape)} at path={path}")
                n = int(math.prod(lhs.shape[:-2]))
                chunk = mixed_flat[offset[0] : offset[0] + n].view(lhs.shape)
                offset[0] += n
                return chunk.to(dtype=lhs.dtype)
            return lhs
        if isinstance(lhs, tuple):
            return tuple(
                self._scatter_mixed(lhs[i], mixed_flat, offset, path + [i]) for i in range(len(lhs))
            )
        if isinstance(lhs, list):
            return [
                self._scatter_mixed(lhs[i], mixed_flat, offset, path + [i]) for i in range(len(lhs))
            ]
        raise TypeError(f"Unsupported state type: {type(lhs)}")

    def forward(self, lhs: Any, rhs: Any, target: Any | None = None):
        if not self._built:
            self.build_from_state(lhs)
        assert self._mixer is not None
        dev = next(self._mixer.parameters()).device
        if torch.is_tensor(lhs):
            dev = lhs.device
        elif isinstance(lhs, (list, tuple)):
            for item in lhs:
                if torch.is_tensor(item):
                    dev = item.device
                    break
        self._mixer.to(dev)

        chunks_l: List[torch.Tensor] = []
        chunks_r: List[torch.Tensor] = []
        chunks_t: List[torch.Tensor] = []

        def collect(l: Any, r: Any, t: Any, path: List[int]):
            if torch.is_tensor(l):
                if _is_wkv_path(path):
                    if l.ndim < 2:
                        raise ValueError(f"WKV tensor must be >=2D, got shape={tuple(l.shape)} at path={path}")
                    n = int(math.prod(l.shape[:-2]))
                    h, w = l.shape[-2], l.shape[-1]
                    chunks_l.append(l.reshape(n, h, w).float())
                    chunks_r.append(r.reshape(n, h, w).float())
                    if t is not None:
                        chunks_t.append(t.reshape(n, h, w).float())
                return
            if isinstance(l, (list, tuple)):
                for i in range(len(l)):
                    tv = t[i] if isinstance(t, (list, tuple)) else t
                    collect(l[i], r[i], tv, path + [i])
                return
            raise TypeError(f"Unsupported state type: {type(l)}")

        collect(lhs, rhs, target, [])
        if not chunks_l:
            raise RuntimeError("No WKV tensor leaves found in state (expected slot idx % 3 == 1).")

        lhs_flat = torch.cat(chunks_l, dim=0)
        rhs_flat = torch.cat(chunks_r, dim=0)
        n, h, w = lhs_flat.shape
        if n != self._num_units:
            raise ValueError(f"Unit count mismatch: expected {self._num_units}, got {n}")
        tgt_flat = torch.cat(chunks_t, dim=0) if chunks_t else None

        # Layer-major A then layer-major B: x[:, :n] is A, x[:, n:] is B.
        # (The previous `torch.stack([lhs, rhs], dim=1).reshape(1, 2n, h, w)` call
        # interleaved A/B per head, which broke the xa/xb split inside the inner
        # mixer — the merger was being trained on a scrambled channel layout.)
        x = torch.cat([lhs_flat, rhs_flat], dim=0).unsqueeze(0).to(dev)
        out = self._mixer(x)
        mixed_flat = out["mixed"].squeeze(0)

        offset = [0]
        mixed_tree = self._scatter_mixed(lhs, mixed_flat.cpu(), offset, [])
        if offset[0] != n:
            raise RuntimeError(f"Scatter offset mismatch: {offset[0]} vs {n}")
        if target is None:
            return {"mixed": mixed_tree}
        if tgt_flat is None:
            return {"mixed": mixed_tree}

        tgt_flat = tgt_flat.to(dev)
        total_loss = torch.tensor(0.0, device=dev)
        rec_count = 0
        for i in range(n):
            total_loss = total_loss + self._tensor_loss(mixed_flat[i], tgt_flat[i])
            rec_count += int((mixed_flat[i] - tgt_flat[i]).abs().mean().item() <= self.rec_tol)
        denom = max(n, 1)
        return {"mixed": mixed_tree, "loss": total_loss / float(denom), "recall": float(rec_count) / float(denom)}
