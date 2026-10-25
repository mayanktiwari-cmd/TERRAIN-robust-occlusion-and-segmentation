"""
resilience.py — node-removal cascade and the resilience index.

Simulates removing the most critical nodes one by one and measures how the
network degrades (giant-component fraction + global efficiency). The
resilience index is the area under the degradation curve: higher = more
robust to losing critical nodes.
"""

import networkx as nx
from typing import Dict, List, Tuple

import config
from criticality.centrality import detect_gatekeepers


def _giant_component_fraction(G: nx.Graph, original_n: int) -> float:
    if G.number_of_nodes() == 0 or original_n == 0:
        return 0.0
    largest = max(nx.connected_components(G), key=len)
    return len(largest) / original_n


def _global_efficiency(G: nx.Graph):
    """Average inverse shortest path; None above the node cap (O(n^3))."""
    n = G.number_of_nodes()
    if n <= 1 or n > config.RESILIENCE_MAX_EFFICIENCY_NODES:
        return None
    return nx.global_efficiency(G)


def simulate_node_removal(G: nx.Graph, centrality: Dict[int, float], top_k: int = 10) -> List[Dict]:
    """Remove the top_k most central nodes in order, logging degradation.

    Args:
        G: road graph.
        centrality: betweenness dict (ranked removal order).
        top_k: how many nodes to remove.

    Returns:
        List of per-step logs: {node, centrality, component_fraction, efficiency}.
    """
    original_n = G.number_of_nodes()
    work = G.copy()
    ordered = sorted(centrality, key=centrality.get, reverse=True)[:max(top_k, 0)]

    log = []
    for node in ordered:
        if node not in work:
            continue
        work.remove_node(node)
        log.append({
            "node": node,
            "centrality": centrality[node],
            "component_fraction": _giant_component_fraction(work, original_n),
            "efficiency": _global_efficiency(work),
        })
    return log


def compute_resilience_index(G: nx.Graph, centrality: Dict[int, float], top_k: int = None) -> Tuple[float, List[Dict]]:
    """Resilience = normalized area under the giant-component curve (0-1).

    Args:
        G: road graph.
        centrality: betweenness dict.
        top_k: nodes to remove; default = number of gatekeepers (top 20%).

    Returns:
        (resilience_index, removal_log).
    """
    if G.number_of_nodes() == 0 or not centrality:
        return 1.0, []

    if top_k is None:
        top_k = len(detect_gatekeepers(centrality))
    top_k = min(max(top_k, 0), G.number_of_nodes())

    log = simulate_node_removal(G, centrality, top_k)
    if not log:
        return 1.0, []

    values = [1.0] + [entry["component_fraction"] for entry in log]
    area = sum((values[i] + values[i + 1]) / 2.0 for i in range(top_k))
    return area / top_k, log
