"""
TERRAIN — graph/skeletonize.py

Convert a predicted (or ground-truth) binary road mask into a 1-pixel
skeleton and extract the raw nodes (junctions + endpoints) and the pixel
chains that connect them. This is the entry point of the graph pipeline.

It consumes the *output of the segmentation model* (not OSM), so the whole
TERRAIN pipeline is end-to-end.

Public API (exported from graph/__init__.py):
    skeletonize_mask(mask) -> (skeleton, nodes, edges)
        mask      : np.ndarray [H, W], binary (0/1 or bool) road mask
        skeleton  : np.ndarray [H, W] uint8 skeleton of the mask
        nodes     : list[tuple[int, int]] of (row, col) at junctions/endpoints
        edges     : list[tuple[(r,c), (r,c)]] pixel-chain segments joining nodes
"""

import numpy as np
from skimage.morphology import skeletonize as _sk_skeletonize
from collections import deque
from typing import Tuple, List, Dict, Set


def _threshold_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Binarize a float (prediction) or int (already binary) mask."""
    mask = np.asarray(mask)
    if mask.dtype != np.bool_:
        if mask.dtype.kind == "f":
            binary = (mask > threshold).astype(np.uint8)
        else:
            binary = (mask > 0).astype(np.uint8)
    else:
        binary = mask.astype(np.uint8)
    return binary


def _is_skeleton_pixel(skeleton: np.ndarray, r: int, c: int) -> bool:
    return 0 <= r < skeleton.shape[0] and 0 <= c < skeleton.shape[1] and skeleton[r, c] == 1


def _neighbor_count(skeleton: np.ndarray, r: int, c: int) -> int:
    """8-connected neighbour count of the skeleton pixel at (r, c)."""
    count = 0
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            if _is_skeleton_pixel(skeleton, r + dr, c + dc):
                count += 1
    return count


def _neighbor_list(skeleton: np.ndarray, r: int, c: int) -> List[tuple]:
    """Return the 8-connected skeleton neighbours of (r, c)."""
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if _is_skeleton_pixel(skeleton, nr, nc):
                out.append((nr, nc))
    return out


def skeletonize_mask(mask: np.ndarray) -> Tuple[np.ndarray, List[tuple], List[tuple]]:
    """
    Skeletonize a binary road mask and decompose it into nodes and chains.

    Nodes are skeleton pixels whose 8-neighbourhood degree is NOT 2 (i.e.
    junctions with 3+ neighbours and dead-end endpoints with 1 neighbour).
    Edges are the maximal chains of skeleton pixels traced between nodes.

    Returns:
        skeleton : uint8 [H, W] binary skeleton
        nodes    : list of (row, col) for every junction / endpoint
        edges    : list of ( (r1,c1), (r2,c2) ) chain endpoints
    """
    binary = _threshold_mask(mask)
    if binary.sum() == 0:
        return np.zeros_like(binary), [], []

    skeleton = _sk_skeletonize(binary).astype(np.uint8)

    # -- Collect node pixels ----------------------------------------------
    rows, cols = np.where(skeleton == 1)
    node_set: Set[tuple] = set()
    for r, c in zip(rows, cols):
        deg = _neighbor_count(skeleton, r, c)
        if deg != 2:  # junction (deg>=3) or endpoint (deg==1)
            node_set.add((r, c))

    nodes = sorted(node_set)

    # -- Trace pixel chains between nodes ---------------------------------
    # Visit every skeleton pixel once. Starting from each node, walk along
    # each outgoing branch until we hit another node; record the endpoints.
    visited: Set[tuple] = set()
    edges: List[tuple] = []
    branch_stack: List[tuple] = []  # (start_node, current_pixel)

    for start in nodes:
        for nbr in _neighbor_list(skeleton, start[0], start[1]):
            branch_stack.append((start, nbr))

    while branch_stack:
        start, cur = branch_stack.pop()
        if cur in visited:
            continue

        # Trace the chain from `start` -> ... until the next node.
        chain = [start, cur]
        prev = start
        visited.add(cur)
        reached = cur

        while True:
            following = [n for n in _neighbor_list(skeleton, cur[0], cur[1]) if n != prev]
            following = [n for n in following if n not in chain]
            if not following:
                break
            nxt = following[0]
            chain.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt
            reached = cur
            if cur in node_set:
                break

        if reached in node_set and (start, reached) not in edges and (reached, start) not in edges:
            edges.append((start, reached))

    return skeleton, nodes, edges
