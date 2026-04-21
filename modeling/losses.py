"""Loss functions for graph-based factor ranking."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def _as_2d(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(0) if x.ndim == 1 else x


def rank_ic_loss(scores: torch.Tensor, returns: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    scores = _as_2d(scores)
    returns = _as_2d(returns)
    mask = _as_2d(mask) if mask is not None else None

    ic_losses = []
    for i in range(scores.shape[0]):
        score_i = scores[i]
        return_i = returns[i]

        if mask is not None:
            valid_mask = mask[i] > 0
            if not valid_mask.any():
                ic_losses.append(torch.zeros((), device=scores.device))
                continue
            score_i = score_i[valid_mask]
            return_i = return_i[valid_mask]

        score_i = (score_i - score_i.mean()) / (score_i.std() + 1e-8)
        return_i = (return_i - return_i.mean()) / (return_i.std() + 1e-8)
        ic = torch.dot(score_i, return_i) / (score_i.shape[0] - 1 + 1e-8)
        ic_losses.append(-ic)

    return torch.stack(ic_losses).mean()


def weighted_pairwise_rank_loss(
    scores: torch.Tensor,
    returns: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    margin: float = 1.0,
) -> torch.Tensor:
    scores = _as_2d(scores)
    returns = _as_2d(returns)
    mask = _as_2d(mask) if mask is not None else None

    losses = []
    for i in range(scores.shape[0]):
        score_i = scores[i]
        return_i = returns[i]

        if mask is not None:
            valid_mask = mask[i] > 0
            if valid_mask.sum() < 2:
                losses.append(torch.zeros((), device=scores.device))
                continue
            score_i = score_i[valid_mask]
            return_i = return_i[valid_mask]

        score_diff = score_i.unsqueeze(0) - score_i.unsqueeze(1)
        return_diff = return_i.unsqueeze(0) - return_i.unsqueeze(1)
        valid_pairs = return_diff != 0
        if not valid_pairs.any():
            losses.append(torch.zeros((), device=scores.device))
            continue

        target = (return_diff > 0).float()
        loss = F.relu(margin - score_diff * (2 * target - 1))
        losses.append(loss[valid_pairs].mean())

    return torch.stack(losses).mean()


def graph_regularizer(
    relations: list,
    prev_latent_adj: Optional[torch.Tensor] = None,
    lambda_sparse: float = 1e-3,
    lambda_temporal: float = 1e-2,
) -> torch.Tensor:
    if not relations:
        return torch.zeros(())

    device = relations[0].adj.device
    reg_loss = torch.zeros((), device=device)

    for rel in relations:
        reg_loss = reg_loss + lambda_sparse * rel.adj.abs().mean()

    latent = next((rel for rel in relations if getattr(rel, "name", "") == "latent"), None)
    if latent is not None and prev_latent_adj is not None and prev_latent_adj.shape == latent.adj.shape:
        reg_loss = reg_loss + lambda_temporal * F.mse_loss(latent.adj, prev_latent_adj)

    return reg_loss


def combined_loss(
    scores: torch.Tensor,
    returns: torch.Tensor,
    relations: Optional[list] = None,
    mask: Optional[torch.Tensor] = None,
    w_rank: float = 1.0,
    w_ic: float = 0.5,
    w_reg: float = 1e-3,
) -> torch.Tensor:
    loss = torch.zeros((), device=scores.device)

    if w_rank > 0:
        loss = loss + w_rank * weighted_pairwise_rank_loss(scores, returns, mask)
    if w_ic > 0:
        loss = loss + w_ic * rank_ic_loss(scores, returns, mask)
    if w_reg > 0 and relations is not None:
        loss = loss + w_reg * graph_regularizer(relations)

    return loss
