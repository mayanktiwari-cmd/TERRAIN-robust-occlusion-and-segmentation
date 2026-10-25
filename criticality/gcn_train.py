"""
gcn_train.py — multi-city inductive GCN training.

Trains on a list of city graphs (split by city, not by node). The default
target is a continuous, per-graph standardized log1p(betweenness) regression
("regression"); the binary top-K% gatekeeper classification ("binary") is kept
for backward compatibility. Loss is mean-per-city so big cities (jaipur ~141k
nodes) do not drown small ones (chandigarh ~12k). A single-graph path is kept
for the dashboard convenience.
"""

import os

import networkx as nx
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple

import config
from criticality.centrality import detect_gatekeepers
from criticality.gcn_model import RoadGCN
from criticality.graph_adapter import graph_to_tensor


def generate_pseudo_labels(G: nx.Graph, centrality: Dict[int, float], percentile: float = None) -> torch.Tensor:
    """Binary labels aligned to G.nodes(): 1 = top (100 - percentile)% gatekeepers."""
    if percentile is None:
        percentile = config.PSEUDO_LABEL_PERCENTILE
    gatekeepers = set(detect_gatekeepers(centrality, percentile))
    return torch.tensor([1 if n in gatekeepers else 0 for n in G.nodes()], dtype=torch.float32)


def _city_loss(logits: torch.Tensor, y: torch.Tensor, target_mode: str) -> torch.Tensor:
    """Per-city loss, mean-normalized by node count (equal city weight)."""
    if target_mode == "regression":
        return F.mse_loss(logits, y, reduction="mean")
    return F.binary_cross_entropy_with_logits(logits, y, reduction="mean")


def _run_epochs(model: RoadGCN, batches: List[Tuple], epochs: int, lr: float,
                device: torch.device, target_mode: str = None) -> float:
    """Shared train loop over a list of (x, edge_index, y) city batches."""
    target_mode = target_mode or config.GCN_TARGET
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    final_loss = float("nan")
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        total = torch.tensor(0.0, device=device)
        for x, edge_index, y in batches:
            x, edge_index, y = x.to(device), edge_index.to(device), y.to(device)
            total = total + _city_loss(model(x, edge_index), y, target_mode)
        total.backward()
        opt.step()
        sched.step()
        final_loss = total.item()
    return final_loss


def train_gcn(
    G: nx.Graph,
    centrality: Dict[int, float],
    epochs: int = None,
    lr: float = None,
    device: torch.device = None,
    target_mode: str = None,
) -> Tuple[object, torch.Tensor]:
    """Single-graph convenience path: returns (trained RoadGCN, final loss)."""
    epochs = epochs or config.GCN_EPOCHS
    lr = lr or config.GCN_LR
    device = device or torch.device("cpu")
    target_mode = target_mode or config.GCN_TARGET

    x, edge_index, y, _ = graph_to_tensor(G, centrality, target=target_mode)
    model = RoadGCN().to(device)
    loss = _run_epochs(model, [(x, edge_index, y)], epochs, lr, device, target_mode)
    return model, torch.tensor(loss)


def train_gcn_multicity(
    cities: Dict[str, Tuple[nx.Graph, Dict[int, float]]],
    train_cities: List[str],
    epochs: int = None,
    lr: float = None,
    device: torch.device = None,
    save_path: str = None,
    target_mode: str = None,
) -> Tuple[object, torch.Tensor]:
    """Multi-city inductive training; loss = mean-per-city over training cities.

    Args:
        cities: {name: (G, centrality)}.
        train_cities: city names to train on (split by city, not by node).
        save_path: optional checkpoint path (defaults to MODEL_SAVE_PATH).
        target_mode: "regression" (default) or "binary".

    Returns:
        (trained RoadGCN, final loss).
    """
    epochs = epochs or config.GCN_EPOCHS
    lr = lr or config.GCN_LR
    device = device or torch.device("cpu")
    target_mode = target_mode or config.GCN_TARGET

    batches = [
        graph_to_tensor(G, centrality, target=target_mode)[:3]
        for name, (G, centrality) in cities.items()
        if name in train_cities
    ]
    if not batches:
        raise ValueError("no training cities selected")

    model = RoadGCN().to(device)
    loss = _run_epochs(model, batches, epochs, lr, device, target_mode)

    if save_path is not None:
        save_path = save_path or config.MODEL_SAVE_PATH
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
    return model, torch.tensor(loss)
