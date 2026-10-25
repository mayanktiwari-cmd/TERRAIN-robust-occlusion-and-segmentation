"""Dev smoke test: synthetic barbell graph -> adapter -> RoadGCN train loop.

Two cliques joined by a single bridge: nodes 4, 5 should be the gatekeepers.
Run: python experiments/dev_gcn.py
"""

import sys
from pathlib import Path

import networkx as nx
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from criticality.centrality import compute_centrality
from criticality.graph_adapter import graph_to_tensor
from criticality.gcn_model import RoadGCN

G = nx.barbell_graph(5, 0)
centrality = compute_centrality(G, k=None)
x, edge_index, y, node_ids = graph_to_tensor(G, centrality)

print("shapes:", tuple(x.shape), tuple(edge_index.shape), tuple(y.shape))
print("gatekeepers:", [node_ids[i] for i in range(len(y)) if y[i] == 1])

model = RoadGCN()
opt = torch.optim.Adam(model.parameters(), lr=0.01)

for step in range(400):
    opt.zero_grad()
    loss = F.binary_cross_entropy_with_logits(model(x, edge_index), y)
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(f"step {step:3d}  loss {loss.item():.4f}")

probs = model.predict(x, edge_index)
k = int((len(node_ids) * 0.2 + 0.5))
ranked = [node_ids[i] for i in probs.argsort(descending=True)[:k]]
print("predicted gatekeepers:", ranked)
