import math
from collections import deque

import networkx as nx
import numpy as np
from scipy.spatial import KDTree

import config


def _haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in meters between two lat/lon points."""
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _node_coords(G, node):
    """Return (lat, lon) when georeferenced, else (x, y) as a fallback key."""
    data = G.nodes[node]
    if "lat" in data and "lng" in data:
        return data["lat"], data["lng"]
    return data.get("x", 0.0), data.get("y", 0.0)


def _node_distance_m(G, a, b):
    """Distance in meters between two nodes (haversine when georeferenced)."""
    la, loa = _node_coords(G, a)
    lb, lob = _node_coords(G, b)
    if "lat" in G.nodes[a] and "lat" in G.nodes[b]:
        return _haversine_m(la, loa, lb, lob)
    # Fallback: raw pixel distance (kept for compatibility with non-georeferenced G).
    return math.hypot(la - lb, loa - lob)


def _edge_direction(G, u, v):
    """Unit vector (lat, lon) pointing from u -> v."""
    lat_u, lon_u = _node_coords(G, u)
    lat_v, lon_v = _node_coords(G, v)
    dy, dx = lat_v - lat_u, lon_v - lon_u
    norm = math.hypot(dy, dx)
    if norm < 1e-12:
        return 0.0, 0.0
    return dy / norm, dx / norm


def _angle_deg(v1, v2):
    """Angle in degrees between two vectors."""
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cos_a = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos_a))


class _DSU:
    """Union-find over fragment indices for cycle-safe greedy merging."""

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def _planarize_healed_edges(G_healed):
    """
    Insert a real junction node wherever a healed bridge chord crosses another
    road's interior, splitting both edges so every visual crossing becomes a
    genuine connected intersection (restores planarity of the healed graph).
    """
    healed_edges = [
        (u, v, data)
        for u, v, data in G_healed.edges(data=True)
        if data.get("healing", False)
    ]
    if not healed_edges:
        return G_healed

    pos = {n: G_healed.nodes[n]["pos"] for n in G_healed.nodes()}

    def _pos(n):
        if n in pos:
            return pos[n]
        return G_healed.nodes[n]["pos"]

    def _bbox_overlap(a, b, c, d):
        return max(min(a[0], b[0]), min(c[0], d[0])) <= min(max(a[0], b[0]), max(c[0], d[0])) and \
               max(min(a[1], b[1]), min(c[1], d[1])) <= min(max(a[1], b[1]), max(c[1], d[1]))

    def _seg_intersect(p1, p2, q1, q2, tol=1e-9):
        x1, y1 = p1; x2, y2 = p2
        x3, y3 = q1; x4, y4 = q2
        den = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
        if abs(den) < tol:
            return None
        t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / den
        u = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / den
        if 0 < t < 1 and 0 < u < 1:
            return t, u
        return None

    # Precompute every edge's endpoints and bbox once.
    all_edges = list(G_healed.edges(data=True))
    ep = [(a, b, _pos(a), _pos(b)) for a, b, d in all_edges]

    # Collect crossings, bbox-rejected for speed. Record the crossing point on
    # BOTH involved edges so a single shared junction node links them.
    split = {}      # edge_pair -> key -> (px,py)
    shared = {}     # (rounded px,py) -> new node id (one junction per point)
    for a, b, d in healed_edges:
        pa, pb = _pos(a), _pos(b)
        for c, e, pc, pe in ep:
            if len({a, b, c, e}) < 4:
                continue
            if not _bbox_overlap(pa, pb, pc, pe):
                continue
            hit = _seg_intersect(pa, pb, pc, pe)
            if hit is None:
                continue
            t, _ = hit
            key = (round(pa[0] + t * (pb[0] - pa[0]), 3),
                   round(pa[1] + t * (pb[1] - pa[1]), 3))
            split.setdefault((a, b), set()).add(key)
            split.setdefault((c, e), set()).add(key)

    # Materialize each affected edge as a chain through shared junction nodes.
    def _junction_id(key):
        if key not in shared:
            shared[key] = "J%d_%d" % (int(key[0] * 1000), int(key[1] * 1000))
        return shared[key]

    for (u, v) in list(split.keys()):
        if not G_healed.has_edge(u, v) and not G_healed.has_edge(v, u):
            continue
        if not split[u, v]:
            continue
        pu, pv = _pos(u), _pos(v)

        def _proj(p):
            return (p[0] - pu[0]) * (pv[0] - pu[0]) + (p[1] - pu[1]) * (pv[1] - pu[1])

        keys = sorted(split[u, v], key=_proj)

        # Ensure every junction node exists (created once, shared across edges).
        for k in keys:
            if k not in shared:
                shared[k] = _junction_id(k)
            nid = shared[k]
            if nid not in G_healed:
                G_healed.add_node(nid, pos=k, pixel=None, x=k[0], y=k[1])
                pos[nid] = k

        if G_healed.has_edge(u, v):
            G_healed.remove_edge(u, v)
        else:
            G_healed.remove_edge(v, u)

        chain = [u] + [shared[k] for k in keys] + [v]
        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            la, lb = _pos(a), _pos(b)
            seg = math.hypot(lb[0] - la[0], lb[1] - la[1])
            G_healed.add_edge(
                a, b, length=seg, chord=seg,
                path=[(la[1], la[0]), (lb[1], lb[0])],
                healing=True,
            )

    return G_healed


def _kink_simplify(G):
    """Collapse degree-2 nodes whose two incident edges are near-collinear."""
    changed = True
    while changed:
        changed = False
        for node in list(G.nodes()):
            if G.degree(node) != 2:
                continue
            a, c = list(G.neighbors(node))
            v_in = _edge_direction(G, a, node)
            v_out = _edge_direction(G, node, c)
            if _angle_deg(v_in, v_out) < config.KINK_TURN_DEG:
                # Preserve total geometry/attributes by splicing a--c.
                len_a = G[a][node].get("length", 0.0)
                len_c = G[node][c].get("length", 0.0)
                path = G[a][node].get("path", []) + G[node][c].get("path", [])
                G.remove_node(node)
                G.add_edge(a, c, length=len_a + len_c, path=path)
                changed = True
                break
    return G


def heal_graph(
    G: nx.Graph,
    max_distance: float = None,
    lambda_weight: float = None,
    max_gap_m: float = None,
    recover_isolated: bool = None,
) -> nx.Graph:
    """
    Heal a fragmented road graph by bridging occlusion gaps between shards.

    Pipeline:
      1. Fragment detection via connected components.
      2. Candidate endpoint-pair gaps found with a KD-Tree (nearest endpoints).
      3. Distance gate (haversine meters) with a HARD per-edge cap = max_gap_m.
      4. Dual-endpoint angular gate (both turn angles <= MAX_ANGLE_HARD).
      5. Greedy bridge by ascending cost = length + lambda*(turn_a+turn_b),
         merging fragments cycle-safely (DSU).
      6. Optional relaxed second pass (gate removed) to recover straight joins.
      7. Optional isolated-node recovery: bridge degree-0 dangling pixels to the
         nearest road component within max_gap_m (raises LCC / cuts fragment count).
      8. Optional kink simplification of near-collinear degree-2 nodes.
      9. All original node/edge attrs are preserved.

    Args:
        G: raw, fragmented road networkx graph with node attrs
            (x, y) and optionally (lat, lng) plus edge 'length' (m) and 'path'.
        max_distance: gap gate for Pass-1 candidate search (default config.MAX_HEAL_DISTANCE).
        lambda_weight: angular penalty weight (default config.LAMBDA_ANGULAR).
        max_gap_m: HARD cap on every healed edge (default config.HEAL_MAX_GAP_M).
        recover_isolated: bridge stranded degree-0 pixels to nearest component
            (default config.HEAL_RECOVER_ISOLATED).

    Returns:
        healed nx.Graph copy.
    """
    max_distance = config.MAX_HEAL_DISTANCE if max_distance is None else max_distance
    lambda_weight = config.LAMBDA_ANGULAR if lambda_weight is None else lambda_weight
    if max_gap_m is None:
        max_gap_m = getattr(config, "HEAL_MAX_GAP_M", config.MAX_HEAL_DISTANCE)
    max_angle_hard = config.MAX_ANGLE_HARD
    relaxed = getattr(config, "HEAL_RELAXED_GATE", True)
    recover_isolated = getattr(config, "HEAL_RECOVER_ISOLATED", False) if recover_isolated is None else recover_isolated

    G_healed = G.copy()

    def _merge_pass(gate_angle):
        fragments = list(nx.connected_components(G_healed))
        node_to_comp = {}
        for ci, comp in enumerate(fragments):
            for node in comp:
                node_to_comp[node] = ci

        endpoints = [n for n in G_healed.nodes() if G_healed.degree(n) == 1]
        if len(endpoints) < 2:
            return 0

        coords = np.array(
            [_node_coords(G_healed, n) for n in endpoints], dtype=np.float64
        )
        tree = KDTree(coords)

        candidates = []
        k = min(10, len(endpoints))
        for i, ep in enumerate(endpoints):
            ep_comp = node_to_comp[ep]
            distances, indices = tree.query(coords[i], k=k)
            distances, indices = distances[1:], indices[1:]  # drop self
            for d, j in zip(distances, indices):
                neighbor = endpoints[j]
                if node_to_comp[neighbor] == ep_comp:
                    continue

                gap_m = _node_distance_m(G_healed, ep, neighbor)
                if gap_m > max_gap_m:  # HARD per-edge cap
                    continue

                ep_neighbors = list(G_healed.neighbors(ep))
                nb_neighbors = list(G_healed.neighbors(neighbor))
                if not ep_neighbors or not nb_neighbors:
                    continue

                dir_a = _edge_direction(G_healed, ep_neighbors[0], ep)
                dir_b = _edge_direction(G_healed, neighbor, nb_neighbors[0])
                d_ab = _edge_direction(G_healed, ep, neighbor)

                turn_a = _angle_deg(dir_a, d_ab)
                turn_b = _angle_deg(d_ab, dir_b)

                if turn_a > gate_angle or turn_b > gate_angle:
                    continue

                cost = gap_m + lambda_weight * (turn_a + turn_b)
                candidates.append(
                    {
                        "a": ep,
                        "b": neighbor,
                        "ca": ep_comp,
                        "cb": node_to_comp[neighbor],
                        "cost": cost,
                        "gap_m": gap_m,
                        "turn_a": turn_a,
                        "turn_b": turn_b,
                    }
                )

        frag_now = list(nx.connected_components(G_healed))
        comp_id = {node: ci for ci, comp in enumerate(frag_now) for node in comp}
        dsu = _DSU(len(frag_now))
        candidates.sort(key=lambda c: c["cost"])
        added = 0
        for cand in candidates:
            ci_a, ci_b = comp_id[cand["a"]], comp_id[cand["b"]]
            if dsu.find(ci_a) == dsu.find(ci_b):
                continue
            if not dsu.union(ci_a, ci_b):
                continue
            a, b = cand["a"], cand["b"]
            path = [_node_coords(G_healed, a), _node_coords(G_healed, b)]
            G_healed.add_edge(
                a,
                b,
                length=cand["gap_m"],
                chord=cand["gap_m"],
                path=path,
                healing=True,
            )
            added += 1
        return added

    def _recover_isolated_pass():
        if not recover_isolated:
            return 0
        isolated = [n for n in G_healed.nodes() if G_healed.degree(n) == 0]
        if not isolated:
            return 0
        others = [n for n in G_healed.nodes() if G_healed.degree(n) > 0]
        if not others:
            return 0

        iso_coords = np.array(
            [_node_coords(G_healed, n) for n in isolated], dtype=np.float64
        )
        oth_coords = np.array(
            [_node_coords(G_healed, n) for n in others], dtype=np.float64
        )
        tree = KDTree(oth_coords)
        distances, indices = tree.query(iso_coords, k=1)

        comp_of = {}
        for i, comp in enumerate(nx.connected_components(G_healed)):
            for node in comp:
                comp_of[node] = i

        added = 0
        for i, node in enumerate(isolated):
            gap_m = float(distances[i])
            if gap_m > max_gap_m:
                continue
            target = others[indices[i]]
            if comp_of.get(node) == comp_of.get(target):
                continue
            path = [_node_coords(G_healed, node), _node_coords(G_healed, target)]
            G_healed.add_edge(
                node,
                target,
                length=gap_m,
                chord=gap_m,
                path=path,
                healing=True,
                recovery="isolated",
            )
            added += 1
        return added

    added1 = _merge_pass(max_angle_hard)
    added2 = _merge_pass(180.0) if relaxed else 0
    added3 = _recover_isolated_pass()
    added4 = _merge_pass(max_angle_hard)

    if getattr(config, "HEAL_PLANARIZE", True):
        G_healed = _planarize_healed_edges(G_healed)

    if getattr(config, "HEAL_KINK_SIMPLIFY", False):
        G_healed = _kink_simplify(G_healed)

    G_healed.graph["healing_edge_count"] = added1 + added2 + added3 + added4
    return G_healed
