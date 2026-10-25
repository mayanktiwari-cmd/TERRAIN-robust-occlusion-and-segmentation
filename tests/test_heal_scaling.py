import sys, os
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while _ROOT and not os.path.exists(os.path.join(_ROOT, "config.py")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
import random
import numpy as np
import networkx as nx
from skimage.morphology import dilation, disk
import config
from graph.skeletonize import skeletonize_mask
from graph.build_graph import build_graph
from graph.heal import heal_graph

random.seed(42)
np.random.seed(42)

MASK = os.path.join(_HERE, "graph-tests", "data", "bengaluru_mask_window.npy")
assert os.path.exists(MASK), "run tests/test_data.py first to generate the mask"


def build_real_graph():
    mask = np.load(MASK)
    road = dilation((mask > 0).astype(np.uint8), disk(3))
    sk, nodes, edges = skeletonize_mask(road)
    return build_graph(sk, nodes, edges)


def healed_gap_meters(G):
    worst = 0.0
    for u, v, d in G.edges(data=True):
        if d.get("healing", False):
            worst = max(worst, float(d.get("length", 0.0)))
    return worst


def main():
    G = build_real_graph()
    n = G.number_of_nodes()

    comps_before = nx.number_connected_components(G)
    Gh = heal_graph(G)  # config-driven: HEAL_MAX_GAP_M, HEAL_RECOVER_ISOLATED=True
    cc = [c for c in nx.connected_components(Gh) if c]
    lcc = max((len(c) for c in cc), default=0)
    lcc_frac = lcc / max(1, n)
    healed_edges = sum(1 for u, v, d in Gh.edges(data=True) if d.get("healing", False))

    print("== CONFIG ==")
    print("HEAL_MAX_GAP_M =", config.HEAL_MAX_GAP_M)
    print("HEAL_RELAXED_GATE =", config.HEAL_RELAXED_GATE)
    print("HEAL_RECOVER_ISOLATED =", config.HEAL_RECOVER_ISOLATED)
    print("BETWEENNESS_K =", config.BETWEENNESS_K)

    print("\n== HEALING ==")
    print(f"comps_before={comps_before}")
    print(f"comps_after={len(cc)}  healed_edges={healed_edges}")
    print(f"LCC (frac of nodes) = {lcc_frac:.4f}")

    # 1. Max-gap cap respected on every healed edge
    worst = healed_gap_meters(Gh)
    print(f"max healed gap = {worst:.1f} <= cap {config.HEAL_MAX_GAP_M}? {worst <= config.HEAL_MAX_GAP_M}")
    assert worst <= config.HEAL_MAX_GAP_M, "a healed edge exceeded the hard cap"

    # 2. LCC target
    assert lcc_frac >= 0.99, f"LCC too low: {lcc_frac:.4f}"
    print("LCC >= 0.99 : PASS")

    

    # 3. Betweenness k=2000 on healed graph (networkx read of config value)
    k = min(config.BETWEENNESS_K, Gh.number_of_nodes() - 1)
    cent = nx.betweenness_centrality(Gh, k=k, weight="length", normalized=True)
    vals = sorted(cent.values(), reverse=True)
    print(f"\n== BETWEENNESS k={config.BETWEENNESS_K} ==")
    print(f"max_centrality = {vals[0]:.4f}")
    print(f"mean_centrality = {np.mean(vals):.4f}")
    print("top10 critical nodes:", sorted(cent, key=cent.get, reverse=True)[:10])

    # 4. k=500 vs k=2000 stability
    k500 = nx.betweenness_centrality(Gh, k=500, weight="length", normalized=True)
    k2000 = nx.betweenness_centrality(Gh, k=2000, weight="length", normalized=True)
    top500 = set(sorted(k500, key=k500.get, reverse=True)[:10])
    top2000 = set(sorted(k2000, key=k2000.get, reverse=True)[:10])
    overlap = len(top500 & top2000) / 10.0
    print(f"top-10 overlap k=500 vs k=2000: {overlap:.0%}")
    assert overlap >= 0.5, "k=500 and k=2000 gatekeepers diverge too much (seed sensitivity)"

    print("\nALL heal-scaling checks PASSED")


if __name__ == "__main__":
    main()