import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while _ROOT and not os.path.exists(os.path.join(_ROOT, "config.py")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
import numpy as np
import networkx as nx
from graph.skeletonize import skeletonize_mask
from graph.build_graph import build_graph
from graph.heal import heal_graph
from skimage.morphology import dilation, disk

MASK = os.path.join(_HERE, "graph-tests", "data", "bengaluru_mask_window.npy")
mask = np.load(MASK)

# Roads are 1-px thin from rasterization; dilate to realistic ~6px width
road = (mask > 0).astype(np.uint8)
road = dilation(road, disk(3))
print("road px", int(road.sum()), "frac", float(road.mean()))

sk, nodes, edges = skeletonize_mask(road)
print("skeleton_px", int(sk.sum()), "nodes", len(nodes), "edges", len(edges))

G = build_graph(sk, nodes, edges)
print("G nodes", G.number_of_nodes(), "edges", G.number_of_edges(),
      "components(BEFORE)", nx.number_connected_components(G))

comps = list(nx.connected_components(G))
lcc_before = max((len(c) for c in comps), default=0) / max(1, G.number_of_nodes())
print("LCC BEFORE (fraction of nodes)", round(lcc_before, 4))

Gh = heal_graph(G, max_distance=200.0, lambda_weight=0.3)
print("components(AFTER)", nx.number_connected_components(Gh))
print("healed_edges", sum(1 for u, v, d in Gh.edges(data=True) if d.get("healing", False)))

comps_h = list(nx.connected_components(Gh))
lcc_after = max((len(c) for c in comps_h), default=0) / max(1, Gh.number_of_nodes())
print("LCC AFTER (fraction of nodes)", round(lcc_after, 4))