"""Shared truncated-SVD helpers used by multiple WKV-state compression experiments."""

from __future__ import annotations

from typing import Tuple

import torch


def rank_from_threshold(singular_values: torch.Tensor, threshold: float) -> int:
    """Smallest rank k such that sum(s[:k]) >= threshold * sum(s). Returns >= 1."""
    total = singular_values.sum()
    if total <= 0:
        return 1
    cumsum = torch.cumsum(singular_values, dim=0)
    target = float(threshold) * float(total.item())
    k = int(torch.searchsorted(cumsum, torch.tensor(target, device=cumsum.device), right=False).item()) + 1
    return max(1, min(k, singular_values.shape[0]))


def truncated_svd_reconstruct_by_threshold(
    matrix: torch.Tensor, threshold: float
) -> Tuple[torch.Tensor, float]:
    """Reconstruct `matrix` using the smallest rank whose cumulative singular value
    share reaches `threshold`. Returns (reconstruction, compression_ratio = 2k/d)."""
    orig_dtype = matrix.dtype
    work = matrix.float()
    u, s, vh = torch.linalg.svd(work, full_matrices=False)
    k = rank_from_threshold(s, threshold)
    u_k = u[:, :k]
    s_k = s[:k]
    vh_k = vh[:k, :]
    recon = (u_k * s_k.unsqueeze(0)) @ vh_k
    d = int(s.shape[0])
    compression = (2.0 * float(k)) / float(d)
    return recon.to(dtype=orig_dtype), compression
