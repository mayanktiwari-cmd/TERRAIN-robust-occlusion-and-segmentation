import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while _ROOT and not os.path.exists(os.path.join(_ROOT, "config.py")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
import random
import numpy as np
import networkx as nx
from graph.skeletonize import skeletonize_mask
from graph.build_graph import build_graph
from graph.heal import heal_graph

random.seed(42)
np.random.seed(42)

H = W = 256
m = np.zeros((H, W), dtype=np.uint8)
m[50, 50:140] = 1
m[90, 60:90] = 1
m[150:210, 120] = 1
m[160, 120:180] = 1
m[50:110, 180] = 1

sk, nodes, edges = skeletonize_mask(m)
print("skeleton_px", int(sk.sum()), "nodes", len(nodes), "edges", len(edges))

G = build_graph(sk, nodes, edges)
print("G", G.number_of_nodes(), "nodes", G.number_of_edges(), "edges",
      "comps", nx.number_connected_components(G))

# provoke disconnected fragments by drop edges then heal
Gh = heal_graph(G, max_distance=60, lambda_weight=0.3)
print("healed comps", nx.number_connected_components(Gh),
      "healed_edges", sum(1 for u, v, d in Gh.edges(data=True) if d.get("healing", False)))