"""
TERRAIN — graph/visualize.py

Plotting helpers for the graph pipeline: raw (pre-heal) vs healed comparison
and a dedicated healing-result figure.

Public API (exported from graph/__init__.py):
    plot_graph_comparison(G_raw, G_healed, save=True)
    plot_healing_result(G_healed, save=True)
"""

import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional
import config


def _node_xy(G: nx.Graph, node) -> tuple:
    """Return (x, y) plot coords for a node, preferring 'x'/'y' attrs."""
    data = G.nodes[node]
    if "x" in data and "y" in data:
        return data["x"], data["y"]
    return data.get("pos", (0, 0))


def _plot_edges(ax, G: nx.Graph, color: str, lw: float = 0.5, alpha: float = 0.7):
    for u, v in G.edges():
        x1, y1 = _node_xy(G, u)
        x2, y2 = _node_xy(G, v)
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=alpha, zorder=1)


def _plot_nodes(ax, G: nx.Graph, s: float = 4, color: str = "black"):
    if not G.nodes:
        return
    xs = [_node_xy(G, n)[0] for n in G.nodes()]
    ys = [_node_xy(G, n)[1] for n in G.nodes()]
    ax.scatter(xs, ys, s=s, color=color, zorder=2)


def plot_graph_comparison(G_raw: nx.Graph, G_healed: nx.Graph, save: bool = True, save_path: Optional[str] = None) -> None:
    """Side-by-side raw vs healed road graph."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].set_title(f"Raw (components={len(list(nx.connected_components(G_raw)))})")
    _plot_edges(axes[0], G_raw, "gray")
    _plot_nodes(axes[0], G_raw, s=3)
    axes[1].set_title(f"Healed (components={len(list(nx.connected_components(G_healed)))})")
    _plot_edges(axes[1], G_healed, "dodgerblue")
    _plot_nodes(axes[1], G_healed, s=3)
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_axis_off()
    plt.tight_layout()
    if save:
        path = save_path or f"{config.OUTPUT_DIR}healing_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_healing_result(G_healed: nx.Graph, save: bool = True, save_path: Optional[str] = None) -> None:
    """Plot healed graph with healed edges highlighted."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title(f"Healed road graph (components={len(list(nx.connected_components(G_healed)))})")
    healed_edges = [(u, v) for u, v, d in G_healed.edges(data=True) if d.get("healing", False)]

    # Draw original edges first, healed edges on top.
    for u, v, d in G_healed.edges(data=True):
        if d.get("healing", False):
            continue
        x1, y1 = _node_xy(G_healed, u)
        x2, y2 = _node_xy(G_healed, v)
        ax.plot([x1, x2], [y1, y2], color="dodgerblue", linewidth=0.5, alpha=0.7, zorder=1)

    for u, v in healed_edges:
        x1, y1 = _node_xy(G_healed, u)
        x2, y2 = _node_xy(G_healed, v)
        ax.plot([x1, x2], [y1, y2], color="red", linewidth=1.6, alpha=0.9, zorder=2)

    _plot_nodes(ax, G_healed, s=3)
    ax.set_aspect("equal")
    ax.set_axis_off()
    plt.tight_layout()
    if save:
        path = save_path or f"{config.OUTPUT_DIR}healing_result.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
