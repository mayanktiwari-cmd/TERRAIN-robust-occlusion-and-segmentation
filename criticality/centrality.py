"""
centrality.py — feature extraction + betweenness + gatekeeper detection.

- Betweenness is TRAIN-TIME ONLY (pseudo-label source); never run at inference.
- Cheap local features (degree centrality, clustering) are the inference inputs.
"""

import networkx as nx
from typing import Dict, List, Tuple

import config


def _edge_weight(G: nx.Graph) -> str:
    """Edge attribute used as cost for betweenness, if present."""
    for _, _, data in G.edges(data=True):
        if "length" in data:
            return "length"
    return None


def compute_centrality(G: nx.Graph, k: int = None) -> Dict[int, float]:
    """Betweenness per node is the train-time ground truth (pseudo-labels).

    Uses edge 'length' (meters) as cost when present; falls back to unweighted.

    Args:
        G: healed road graph (networkx.Graph).
        k: Brandes pivot count; None or k >= n = exact. Defaults to BETWEENNESS_K.

    Returns:
        Dict[node id -> betweenness centrality, normalized].
    """
    if G.number_of_nodes() == 0:
        return {}

    if k is None:
        k = config.BETWEENNESS_K
    if k is not None and k >= G.number_of_nodes():
        k = None

    return nx.betweenness_centrality(G, k=k, weight=_edge_weight(G), normalized=True)


def detect_gatekeepers(centrality: Dict[int, float], percentile: float = None) -> List[int]:
    """Top (100 - percentile)% by centrality = gatekeepers.

    Per-city RELATIVE cut — always the city's own top slice, never an absolute
    score boundary.

    Args:
        centrality: output of compute_centrality.
        percentile: rank cutoff; defaults to PSEUDO_LABEL_PERCENTILE (80 -> top 20%).

    Returns:
        Gatekeeper node ids, highest centrality first.
    """
    if percentile is None:
        percentile = config.PSEUDO_LABEL_PERCENTILE
    if not centrality:
        return []

    keep_frac = max(1.0 - percentile / 100.0, 0.0)
    k = max(int(round(len(centrality) * keep_frac)), 1)
    return sorted(centrality, key=centrality.get, reverse=True)[:k]


def extract_features(G: nx.Graph) -> Dict[int, Tuple[float, ...]]:
    """Cheap scale-free node features — the only inference-time inputs.

    All are O(n)-O(n*m)-free, i.e. no global shortest-path computation, so they
    stay cheap at inference. Per-graph z-scored by the adapter before the GCN.

    Feature 1 — degree centrality: dense hubs (comparable across city sizes).
    Feature 2 — local clustering: bridges at LOW degree (sparse neighborhoods).
    Feature 3 — k-core number: the hierarchical backbone (arterials sit in the
                highest k-cores of road networks; strong betweenness correlate).
    Feature 4 — 2-hop degree (sum of neighbors' degrees): hub-ness proxy.
    Feature 5 — mean neighbor degree: assortativity (high-degree nodes tend to
                connect to other high-degree nodes on main corridors).
    """
    # k-core disallows self-loops; strip them on a copy so all features stay aligned.
    G = G.copy()
    self_loops = list(nx.selfloop_edges(G))
    if self_loops:
        G.remove_edges_from(self_loops)

    deg = dict(G.degree())
    deg_c = nx.degree_centrality(G)
    cluster = nx.clustering(G)
    kcore = nx.core_number(G)

    nb_deg_sum: Dict[int, float] = {}
    for u, v in G.edges():
        nb_deg_sum[u] = nb_deg_sum.get(u, 0.0) + deg[v]
        nb_deg_sum[v] = nb_deg_sum.get(v, 0.0) + deg[u]

    return {
        n: (
            float(deg_c.get(n, 0.0)),
            float(cluster.get(n, 0.0)),
            float(kcore.get(n, 0)),
            float(nb_deg_sum.get(n, 0.0)),
            float(nb_deg_sum.get(n, 0.0) / deg[n]) if deg[n] > 0 else 0.0,
        )
        for n in G.nodes()
    }

