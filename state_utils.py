"""Pure state-tree utilities for RWKV state manipulation.

RWKV states are arbitrarily nested tuples/lists of tensors. This module provides
canonical helpers for walking, copying, and combining such states without any
dependency on the rwkv package itself.

RWKV-7 state layout per layer (top-level list):
    idx*3+0: time-mixing shift vector
    idx*3+1: matrix-valued WKV state      (the "WKV tensor" everything cares about)
    idx*3+2: channel-mixing shift vector
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

StateLike = Union[torch.Tensor, List["StateLike"], Tuple["StateLike", ...]]


# ---------------------------------------------------------------------------
# WKV layer path helpers
# ---------------------------------------------------------------------------

def _path_to_wkv_layer(path: Tuple[int, ...]) -> Union[int, None]:
    """Return layer index if `path` points at a WKV matrix state, else None."""
    if len(path) != 1:
        return None
    slot_idx = path[0]
    if slot_idx % 3 != 1:
        return None
    return slot_idx // 3


def _is_wkv_path(path: Sequence[int]) -> bool:
    """Lightweight boolean alias for `_path_to_wkv_layer(path) is not None`."""
    return len(path) == 1 and (path[0] % 3 == 1)


# ---------------------------------------------------------------------------
# Tree walking
# ---------------------------------------------------------------------------

def _apply_state(fn, state: StateLike, path: Tuple[int, ...] = ()) -> StateLike:
    """Apply `fn(tensor, path)` at every tensor leaf, preserving structure."""
    if torch.is_tensor(state):
        return fn(state, path)
    if isinstance(state, list):
        return [_apply_state(fn, x, path + (i,)) for i, x in enumerate(state)]
    if isinstance(state, tuple):
        return tuple(_apply_state(fn, x, path + (i,)) for i, x in enumerate(state))
    raise TypeError(f"Unsupported state type: {type(state)}")


# ---------------------------------------------------------------------------
# Basic ops
# ---------------------------------------------------------------------------

def clone_state(state: StateLike) -> StateLike:
    return _apply_state(lambda x, _: x.clone(), state)


def move_state_to_device(state: StateLike, device: Union[torch.device, str]) -> StateLike:
    return _apply_state(lambda x, _: x.to(device), state)


def move_state_to_cpu(state: StateLike) -> StateLike:
    return _apply_state(lambda x, _: x.detach().cpu(), state)


# ---------------------------------------------------------------------------
# Stacking / unstacking
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Arithmetic over structured states
# ---------------------------------------------------------------------------

def add_states(lhs: StateLike, rhs: StateLike) -> StateLike:
    if torch.is_tensor(lhs):
        return lhs + rhs
    if isinstance(lhs, list):
        return [add_states(lv, rv) for lv, rv in zip(lhs, rhs)]
    if isinstance(lhs, tuple):
        return tuple(add_states(lv, rv) for lv, rv in zip(lhs, rhs))
    raise TypeError(f"Unsupported state type: {type(lhs)}")


def scale_state(state: StateLike, factor: float) -> StateLike:
    return _apply_state(lambda x, _: x * factor, state)


def mean_states(states: Sequence[StateLike]) -> StateLike:
    """Elementwise average of a sequence of states (uses `add_states` + `scale_state`)."""
    if not states:
        raise ValueError("mean_states requires at least one state.")
    merged = clone_state(states[0])
    for other in states[1:]:
        merged = add_states(merged, other)
    return scale_state(merged, 1.0 / len(states))


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def mean_squared_error_state(pred: StateLike, target: StateLike) -> Tuple[torch.Tensor, float]:
    """MSE averaged over every tensor leaf in a state tree."""
    losses: List[torch.Tensor] = []

    def collect(lhs: StateLike, rhs: StateLike):
        if torch.is_tensor(lhs):
            losses.append(F.mse_loss(lhs, rhs))
            return
        if isinstance(lhs, (list, tuple)):
            for lv, rv in zip(lhs, rhs):
                collect(lv, rv)
            return
        raise TypeError(f"Unsupported state type: {type(lhs)}")

    collect(pred, target)
    if not losses:
        raise ValueError("No tensor leaves found in state.")
    loss = torch.stack(losses).mean()
    return loss, float(loss.detach().cpu().item())
