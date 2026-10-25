"""
TERRAIN — graph/build_graph.py

Build a NetworkX graph from the skeleton nodes/chains produced by
skeletonize_mask. Node attributes are written so graph/heal.py can consume
them directly:
    - 'x', 'y'      : pixel column / row (heal._node_coords fallback)
    - 'pixel'       : (row, col) source pixel
    - 'pos'         : (x, y) == (col, row)
Edges carry:
    - 'length'      : Euclidean pixel distance of the chain
    - 'path'        : ordered list of pixel coords along the chain

If an affine transform is supplied, node positions (x, y) are converted to
projected map coordinates via rasterio-style `transform * (x, y)` and
`georef=True` is stored so heal.py uses haversine distances.

Public API (exported from graph/__init__.py):
    build_graph(skeleton, nodes, edges, transform=None) -> nx.Graph
"""

import networkx as nx
import numpy as np
from typing import List, Tuple, Optional
import config


def _node_index(r: int, c: int, width: int) -> int:
    """Deterministic integer key for a node from its (row, col)."""
    return int(r) * width + int(c)


def build_graph(
    skeleton: np.ndarray,
    nodes: List[tuple],
    edges: List[tuple],
    transform=None,
) -> nx.Graph:
    """
    Assemble a NetworkX graph from skeletonise output.

    Args:
        skeleton: uint8 [H, W] predicted skeleton (used only for width).
        nodes   : list of (row, col) junctions/endpoints.
        edges   : list of ((r1,c1),(r2,c2)) chains.
        transform: optional rasterio-like affine used to geolocate pixels.
                   Node attr (x, y) becomes projected lon/lat and 'geoset'=True.
    """
    G = nx.Graph()
    if skeleton is None:
        skeleton = np.zeros((1, 1), dtype=np.uint8)
    height, width = skeleton.shape[:2]

    use_geo = transform is not None

    for (r, c) in nodes:
        nid = _node_index(r, c, width)
        if take_geo := use_geo:
            x, y = transform * (c, r)  # (col, row) -> projected coords
            G.add_node(nid, pos=(x, y), pixel=(r, c), x=x, y=y, geoset=True)
        else:
            G.add_node(nid, pos=(c, r), pixel=(r, c), x=c, y=r, geoset=False)

    for (a, b) in edges:
        (r1, c1), (r2, c2) = a, b
        na, nb = _node_index(r1, c1, width), _node_index(r2, c2, width)
        if na not in G or nb not in G:
            continue
        length = float(np.hypot(c2 - c1, r2 - r1))
        path = [(r1, c1), (r2, c2)]
        G.add_edge(na, nb, length=length, path=path)

    return G