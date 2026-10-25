"""
graph_adapter.py — convert a road graph (networkx) to GCN tensors.

Deterministic node order is the GCN contract: row i of x corresponds to
node_ids[i], and edge_index indices refer to that same ordering.
"""

import networkx as nx
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import config
from criticality.centrality import detect_gatekeepers, extract_features


def zscore_normalize(features: Dict[int, Tuple[float, ...]]) -> Dict[int, Tuple[float, ...]]:
    """Per-graph z-score so the GCN only sees relative structure.

    std == 0 (uniform graph) is left at 0.0 instead of NaN.
    """
    n = len(features)
    if n == 0:
        return {}

    feat = np.array(list(features.values()), dtype=np.float64)
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    std[std == 0.0] = 1.0
    feat = (feat - mean) / std

    ids = list(features.keys())
    return {ids[i]: tuple(float(v) for v in feat[i]) for i in range(n)}


def graph_to_tensor(
    G: nx.Graph,
    centrality: Optional[Dict[int, float]] = None,
    target: str = "binary",
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], List[int]]:
    """Build GCN tensors from a road graph.

    Args:
        G: healed road graph.
        centrality: betweenness dict (train-time). None at inference -> y = None.
        target: "binary" -> top-K% gatekeeper 0/1 labels; "regression" ->
            per-graph standardized log1p(betweenness) continuous target.

    Returns:
        (x, edge_index, y, node_ids): features (n, width), adjacency (2, 2E),
        labels (n,) or None, and node id order aligned to tensor rows.
    """
    node_ids = list(G.nodes())
    n = len(node_ids)
    pos = {nid: i for i, nid in enumerate(node_ids)}

    features = zscore_normalize(extract_features(G))
    feat_vals = list(features.values())
    width = len(feat_vals[0]) if feat_vals else config.GCN_INPUT_FEATURES
    x = torch.tensor([[features[nid][j] for j in range(width)] for nid in node_ids], dtype=torch.float32)

    src, dst = [], []
    for u, v in G.edges():
        src += [pos[u], pos[v]]
        dst += [pos[v], pos[u]]
    edge_index = torch.tensor([src, dst], dtype=torch.long) if src else torch.empty((2, 0), dtype=torch.long)

    y = None
    if centrality is not None:
        vals = np.array([centrality.get(nid, 0.0) for nid in node_ids], dtype=np.float64)
        if target == "regression":
            logv = np.log1p(vals)
            std = float(logv.std())
            if std < 1e-12:
                std = 1.0
            y = torch.tensor((logv - logv.mean()) / std, dtype=torch.float32)
        else:
            gatekeepers = set(detect_gatekeepers(centrality))
            y = torch.tensor([1 if nid in gatekeepers else 0 for nid in node_ids], dtype=torch.float32)

    return x, edge_index, y, node_ids
