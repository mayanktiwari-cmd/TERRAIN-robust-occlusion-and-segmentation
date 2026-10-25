"""Multi-city inductive GCN dev harness.

Trains on a few archetype cities (via gcn_train) and evaluates on held-out
ones (via gcn_evaluate). Run: python experiments/dev_multicity.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from criticality.centrality import compute_centrality
from criticality.gcn_evaluate import evaluate_gcn, rank_gatekeepers
from criticality.gcn_train import train_gcn_multicity
from criticality.graph_adapter import graph_to_tensor
from experiments.synthetic_graphs import grid_city, maze_city, organic_city, sprawl_city


def main():
    cities = {
        "grid": grid_city(),
        "organic": organic_city(),
        "maze": maze_city(),
        "sprawl": sprawl_city(),
    }
    train_names = ["grid", "organic", "maze"]
    holdout = ["sprawl"]

    city_data = {}
    for name, G in cities.items():
        centrality = compute_centrality(G, k=config.BETWEENNESS_K)
        x, edge_index, y, node_ids = graph_to_tensor(G, centrality)
        city_data[name] = (centrality, x, edge_index, y, node_ids)
        print(f"{name:8s} n={len(G)} m={G.number_of_edges()} gatekeepers={int(y.sum())}")

    model, loss = train_gcn_multicity(
        {name: (G, compute_centrality(G, k=config.BETWEENNESS_K)) for name, G in cities.items()},
        train_names,
    )
    print(f"final training loss: {loss.item():.4f}")

    model.eval()
    for name in holdout:
        centrality, x, edge_index, y, node_ids = city_data[name]
        mask = torch.ones_like(y, dtype=torch.bool)
        metrics = evaluate_gcn(model, x, edge_index, y, mask)
        ranked = [node_ids[i] for i in rank_gatekeepers(model, x, edge_index).tolist()]
        print(f"{name:8s} coverage={metrics['coverage_at_k']:.3f}  auc={metrics['auc']:.3f}  spearman={metrics['spearman']:.3f}")
        print(f"{'':8s} top-K: {ranked}")


if __name__ == "__main__":
    main()
