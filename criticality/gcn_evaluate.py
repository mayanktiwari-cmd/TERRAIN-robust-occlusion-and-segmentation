"""
gcn_evaluate.py — held-out evaluation and validation for RoadGCN.

Metrics are per-city relative: Coverage@K, AUC, and rank correlation,
computed on the masked (held-out) nodes.
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from typing import Dict

import config


def rank_gatekeepers(model: torch.nn.Module, x: torch.Tensor, edge_index: torch.Tensor, percentile: float = None) -> torch.Tensor:
    """Tensor indices of the city's top-K% nodes by predicted criticality."""
    if percentile is None:
        percentile = config.PSEUDO_LABEL_PERCENTILE
    device = next(model.parameters()).device
    x, edge_index = x.to(device), edge_index.to(device)
    with torch.no_grad():
        probs = model.predict(x, edge_index)
    keep = max(int(round(x.size(0) * (1 - percentile / 100))), 1)
    return probs.argsort(descending=True)[:keep]


def _topk_labels(values: np.ndarray, percentile: float) -> np.ndarray:
    """Binary labels: 1 for the top (100 - percentile)% of `values`."""
    keep = max(int(round(len(values) * (1 - percentile / 100))), 1)
    labels = np.zeros(len(values), dtype=np.int64)
    labels[np.argsort(values)[-keep:]] = 1
    return labels


def _coverage_at_k(probs: np.ndarray, labels: np.ndarray, percentile: float) -> float:
    keep = max(int(round(len(probs) * (1 - percentile / 100))), 1)
    pred = set(np.argsort(probs)[-keep:])
    pos = set(np.flatnonzero(labels == 1))
    return len(pred & pos) / max(len(pos), 1)


def _auc(probs: np.ndarray, labels: np.ndarray) -> float:
    pos = labels == 1
    if pos.sum() == 0 or (~pos).sum() == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(probs)) + 1.0
    n_pos, n_neg = pos.sum(), len(probs) - pos.sum()
    return (ranks[pos].mean() - (n_pos + 1) / 2) / n_neg


def evaluate_gcn(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
    continuous: np.ndarray = None,
    percentile: float = None,
) -> Dict[str, float]:
    """Coverage@K, AUC, Spearman on the masked (held-out) nodes.

    Args:
        y: binary top-K% labels. If `continuous` is given, binary labels are
           derived from the top-K% of `continuous` (matching the betweenness
           pseudo-label source) and Spearman is computed against `continuous`
           (the honest ranking signal).
        continuous: per-node continuous target (e.g. log1p betweenness).
    """
    pct = percentile or config.PSEUDO_LABEL_PERCENTILE
    device = next(model.parameters()).device
    x, edge_index = x.to(device), edge_index.to(device)
    model.eval()
    with torch.no_grad():
        probs = model.predict(x, edge_index).cpu().numpy()
    m = mask.cpu().numpy().astype(bool)
    probs = probs[m]

    if continuous is not None:
        cont = np.asarray(continuous, dtype=np.float64)[m]
        labels = _topk_labels(cont, pct)
        rank_corr = float(spearmanr(probs, cont).statistic)
    else:
        labels = y.cpu().numpy()[m].astype(np.int64)
        rank_corr = float(spearmanr(probs, labels).statistic)

    return {
        "coverage_at_k": _coverage_at_k(probs, labels, pct),
        "auc": _auc(probs, labels),
        "spearman": rank_corr,
    }


def validate_gcn(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    val_mask: torch.Tensor,
    target_mode: str = None,
) -> float:
    """Loss (MSE regression or BCE binary) on the masked (held-out) nodes."""
    target_mode = target_mode or config.GCN_TARGET
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
        if target_mode == "regression":
            return F.mse_loss(logits[val_mask], y[val_mask]).item()
        return F.binary_cross_entropy_with_logits(logits[val_mask], y[val_mask]).item()
