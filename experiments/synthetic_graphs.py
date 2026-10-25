"""Synthetic city graphs covering street-pattern archetypes.

Each graph gets node 'pos' (2D) and edge 'length' (meters, euclidean) so
weighted betweenness behaves like real road graphs. Node ids are 0..n-1.
"""

import networkx as nx
import numpy as np


def _finalize(G):
    G = nx.convert_node_labels_to_integers(G)
    for u, v in G.edges():
        pu, pv = G.nodes[u]["pos"], G.nodes[v]["pos"]
        G[u][v]["length"] = float(np.hypot(pu[0] - pv[0], pu[1] - pv[1]))
    return G


def grid_city(h=6, w=6):
    """Planned grid — regular degrees, low clustering variance."""
    G = nx.grid_2d_graph(h, w)
    for n in G.nodes:
        G.nodes[n]["pos"] = (float(n[0]), float(n[1]))
    return _finalize(G)


def organic_city(n=60, radius=0.22, seed=0):
    """Organic/unplanned — high clustering, winding low-degree streets."""
    pos = {i: (float(x), float(y)) for i, (x, y) in enumerate(np.random.default_rng(seed).random((n, 2)))}
    G = nx.random_geometric_graph(n, radius, seed=seed, pos=pos)
    for i in pos:
        G.nodes[i]["pos"] = pos[i]
    return _finalize(G)


def maze_city(h=10, w=10, seed=0):
    """Maze/constrained — random spanning tree: few junctions, many dead ends."""
    G = nx.grid_2d_graph(h, w)
    rng = np.random.default_rng(seed)
    for u, v in G.edges():
        G[u][v]["w"] = float(rng.random())
    tree = nx.random_spanning_tree(G, weight="w")
    for n in tree.nodes:
        tree.nodes[n]["pos"] = (float(n[0]), float(n[1]))
    return _finalize(tree)


def sprawl_city(grid=4, suburbs=12, seed=0):
    """Sprawl — dense core + ring road + radiating leaf suburbs."""
    rng = np.random.default_rng(seed)
    G = nx.grid_2d_graph(grid, grid)
    pos = {n: (float(n[0]), float(n[1])) for n in G.nodes}

    cx, cy = (grid - 1) / 2, (grid - 1) / 2
    boundary = [n for n in G.nodes if n[0] in (0, grid - 1) or n[1] in (0, grid - 1)]
    boundary.sort(key=lambda n: np.arctan2(n[1] - cy, n[0] - cx))
    for i in range(len(boundary)):
        u, v = boundary[i], boundary[(i + 1) % len(boundary)]
        if not G.has_edge(u, v):
            G.add_edge(u, v)

    core = list(G.nodes)
    for i in range(suburbs):
        leaf = (grid + i, -1 - int(rng.integers(0, 2)))
        parent = core[int(rng.integers(0, len(core)))]
        G.add_edge(leaf, parent)
        pos[leaf] = (float(leaf[0]), float(leaf[1]))

    for n in G.nodes:
        G.nodes[n]["pos"] = pos[n]
    return _finalize(G)
