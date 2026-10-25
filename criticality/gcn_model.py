"""
gcn_model.py — RoadGCN: 2-layer GCN

Self-loops + symmetric normalization
S_ij = A_ij / sqrt(d_i * d_j), two propagate rounds, per-node logits.
"""

import torch
import torch.nn as nn

import config


class RoadGCN(nn.Module):
    def __init__(self, input_dim: int = None, hidden_dim: int = None):
        super().__init__()
        if input_dim is None:
            input_dim = config.GCN_INPUT_FEATURES
        if hidden_dim is None:
            hidden_dim = config.GCN_HIDDEN
        self.W0 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W1 = nn.Linear(hidden_dim, 1, bias=False)
        self.dropout = nn.Dropout(config.GCN_DROPOUT)

    @staticmethod
    def _normalize(edge_index: torch.Tensor, n: int) -> torch.Tensor:
        """Per-edge norm weights for S = D^-1/2 (A + I) D^-1/2.

        edge_index must already include self-loops.
        """
        src, dst = edge_index
        deg = torch.zeros(n, dtype=torch.float32, device=edge_index.device)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        return deg_inv_sqrt[src] * deg_inv_sqrt[dst]

    def _propagate(self, x: torch.Tensor, edge_index: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        """Aggregate weighted neighbor features into each node."""
        src, dst = edge_index
        msg = x[src] * norm.unsqueeze(-1)
        out = torch.zeros(x.size(0), x.size(1), dtype=x.dtype, device=x.device)
        out.index_add_(0, dst, msg)
        return out

    def _edge_index_with_loops(self, edge_index: torch.Tensor, n: int) -> torch.Tensor:
        loops = torch.arange(n, dtype=torch.long, device=edge_index.device).repeat(2, 1)
        return torch.cat([edge_index, loops], dim=1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Raw logits per node, shape (n,)."""
        n = x.size(0)
        edge_index = self._edge_index_with_loops(edge_index, n)
        norm = self._normalize(edge_index, n)

        h = torch.relu(self._propagate(x, edge_index, norm) @ self.W0.weight.t())
        h = self.dropout(h)
        z = self._propagate(h, edge_index, norm) @ self.W1.weight.t()
        return z.squeeze(-1)

    def predict(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Gatekeeper probabilities per node, shape (n,)."""
        return torch.sigmoid(self.forward(x, edge_index))
