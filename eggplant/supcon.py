"""Supervised contrastive loss (Khosla et al., NeurIPS 2020), 'SupCon L_out' formulation.

Written from the paper; structure follows the reference implementation at
https://github.com/HobbitLong/SupContrast (BSD-2-Clause, Copyright (c) 2020 Yonglong Tian).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """features: [B, V, D] projected embeddings (V views per source image); labels: [B].

    Anchors = all B*V views. Positives = every other view with the same label (incl. the other
    view(s) of the anchor image). Self-pairs are excluded from numerator and denominator.
    Computed in float32 with a max-subtracted log-sum-exp for stability."""
    if features.dim() != 3:
        raise ValueError("features must be [batch, views, dim]")
    b, v, _ = features.shape
    if labels.shape[0] != b:
        raise ValueError("labels/features batch mismatch")
    z = F.normalize(features.float(), dim=-1)
    z = z.transpose(0, 1).reshape(v * b, -1)          # view-major: rows [view0 imgs..., view1 imgs...]
    y = labels.repeat(v)                               # matches the view-major ordering
    logits = z @ z.T / temperature
    self_mask = torch.eye(v * b, dtype=torch.bool, device=z.device)
    logits = logits.masked_fill(self_mask, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    pos = (y[:, None] == y[None, :]) & ~self_mask
    n_pos = pos.sum(1)
    if (n_pos == 0).any():  # impossible with V>=2, kept as a guard
        raise RuntimeError("anchor without positives")
    mean_log_prob_pos = log_prob.masked_fill(~pos, 0.0).sum(1) / n_pos
    return -mean_log_prob_pos.mean()
