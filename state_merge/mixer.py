import math
from typing import Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _is_wkv_path(path: List[int]) -> bool:
    # RWKV state layout: layer*3+1 is WKV matrix state.
    return len(path) == 1 and (path[0] % 3 == 1)


class _LayerMixerBlock(nn.Module):
    """
    One layer: inputs are A_ell, B_ell, and previous layer merged AB_{ell-1} (same head layout).
    Output is AB_ell (same shape as A_ell).
    """

    def __init__(self, heads_per_layer: int, height: int, width: int):
        super().__init__()
        if heads_per_layer < 1:
            raise ValueError(f"heads_per_layer must be >= 1, got {heads_per_layer}")
        self.heads_per_layer = heads_per_layer
        self.height = height
        self.width = width
        hidden_dim = max(8, heads_per_layer // 3)
        self.context_mlp = nn.Sequential(
            nn.Linear(heads_per_layer * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, heads_per_layer * 2),
        )
        self.spatial_conv = nn.Conv2d(heads_per_layer * 3, heads_per_layer * 2, kernel_size=1, bias=True)
        nn.init.normal_(self.spatial_conv.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.spatial_conv.bias)

    def forward(self, x: torch.Tensor):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D [B, 3*H, h, w], got {tuple(x.shape)}")
        if x.size(1) != 3 * self.heads_per_layer:
            raise ValueError(
                f"Expected channel size 3*H={3 * self.heads_per_layer}, got {x.size(1)}"
            )
        if x.size(2) != self.height or x.size(3) != self.width:
            raise ValueError(
                f"Expected spatial size {(self.height, self.width)}, got {(x.size(2), x.size(3))}"
            )

        bsz, _, h, w = x.shape
        xg = x.view(bsz, self.heads_per_layer, 3, h, w)
        global_context = xg.mean(dim=(-2, -1)).reshape(bsz, self.heads_per_layer * 3)
        global_logits = self.context_mlp(global_context).view(bsz, self.heads_per_layer, 2, 1, 1)
        spatial_logits = self.spatial_conv(x).view(bsz, self.heads_per_layer, 2, h, w)
        logits = global_logits + spatial_logits

        weight_b = torch.sigmoid(logits[:, :, 1] - logits[:, :, 0])
        weight_a = 1.0 - weight_b
        coeff = torch.stack([weight_a, weight_b], dim=2)
        xa = xg[:, :, 0]
        xb = xg[:, :, 1]
        mixed = xa + weight_b * (xb - xa)
        return {
            "logits": logits,
            "coeff": coeff,
            "weight_a": weight_a,
            "weight_b": weight_b,
            "mixed": mixed,
        }


class DynamicStateMixer(nn.Module):
    """
    Layer-wise sequential mixer: layer ell uses A_ell, B_ell, and AB_{ell-1}
    (zeros for ell == 0). Each layer has its own parameters.
    """

    def __init__(self, num_layers: int, heads_per_layer: int, height: int, width: int):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if heads_per_layer < 1:
            raise ValueError(f"heads_per_layer must be >= 1, got {heads_per_layer}")
        if height < 1 or width < 1:
            raise ValueError(f"height/width must be >= 1, got {(height, width)}")
        self.num_layers = num_layers
        self.heads_per_layer = heads_per_layer
        self.height = height
        self.width = width
        self.num_groups = num_layers * heads_per_layer
        self.layer_blocks = nn.ModuleList(
            [_LayerMixerBlock(heads_per_layer, height, width) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor):
        if x.dim() != 4:
            raise ValueError(f"Expected 4D [B, 2*N, H, W], got {tuple(x.shape)}")
        n_total = self.num_layers * self.heads_per_layer
        if x.size(1) != 2 * n_total:
            raise ValueError(f"Expected channel size 2*N={2 * n_total}, got {x.size(1)}")
        if x.size(2) != self.height or x.size(3) != self.width:
            raise ValueError(
                f"Expected spatial size {(self.height, self.width)}, got {(x.size(2), x.size(3))}"
            )

        bsz, _, h, w = x.shape
        xa = x[:, :n_total].view(bsz, self.num_layers, self.heads_per_layer, h, w)
        xb = x[:, n_total:].view(bsz, self.num_layers, self.heads_per_layer, h, w)
        prev_ab = torch.zeros(bsz, self.heads_per_layer, h, w, device=x.device, dtype=x.dtype)

        logits_list: List[torch.Tensor] = []
        coeff_list: List[torch.Tensor] = []
        weight_a_list: List[torch.Tensor] = []
        weight_b_list: List[torch.Tensor] = []
        mixed_layers: List[torch.Tensor] = []

        for ell in range(self.num_layers):
            xa_l = xa[:, ell]
            xb_l = xb[:, ell]
            cat = torch.cat([xa_l, xb_l, prev_ab], dim=1)
            out = self.layer_blocks[ell](cat)
            prev_ab = out["mixed"]
            logits_list.append(out["logits"])
            coeff_list.append(out["coeff"])
            weight_a_list.append(out["weight_a"])
            weight_b_list.append(out["weight_b"])
            mixed_layers.append(prev_ab.unsqueeze(1))

        mixed = torch.cat(mixed_layers, dim=1).view(bsz, n_total, h, w)
        logits = torch.stack(logits_list, dim=1)
        coeff = torch.stack(coeff_list, dim=1)
        weight_a = torch.stack(weight_a_list, dim=1)
        weight_b = torch.stack(weight_b_list, dim=1)
        return {
            "logits": logits,
            "coeff": coeff,
            "weight_a": weight_a,
            "weight_b": weight_b,
            "mixed": mixed,
        }


# Backward compatibility for older imports/scripts.
PointwiseKernelMixer = DynamicStateMixer


def move_state_to_device(state: Any, device: torch.device | str):
    if isinstance(state, list):
        return [move_state_to_device(x, device) for x in state]
    if isinstance(state, tuple):
        return tuple(move_state_to_device(x, device) for x in state)
    if torch.is_tensor(state):
        return state.to(device)
    return state


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
    def __init__(
        self,
        mse_weight: float = 1.0,
        l1_weight: float = 0.2,
        cosine_weight: float = 0.2,
        rec_tol: float = 1e-3,
    ):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.cosine_weight = cosine_weight
        self.rec_tol = rec_tol
        self._num_units = 0
        self._num_layers = 0
        self._heads_per_layer = 0
        self._mixer: DynamicStateMixer | None = None
        self._built = False

    @property
    def num_groups(self) -> int:
        return self._num_units

    def build_from_state(self, state: Any):
        if self._built:
            return
        num_layers, heads, h, w = _count_layers_heads_spatial(state)
        self._num_layers = num_layers
        self._heads_per_layer = heads
        self._num_units = num_layers * heads
        if self._num_units < 1:
            raise ValueError("State has no valid tensor leaves.")
        self._mixer = DynamicStateMixer(num_layers, heads, h, w)
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

        x = torch.stack([lhs_flat, rhs_flat], dim=1).reshape(1, 2 * n, h, w).to(dev)
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
