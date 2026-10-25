"""
TERRAIN — tests/harness.py

Generic, map-agnostic runner for the graph pipeline. No graph/ or config.py
code is touched; this only *drives* the public API for any registered map and
any window size, independently, and reports metrics + per-stage wall-clock.

Usage (thin wrappers):
    from harness import sample_window, run_graph, compute_metrics, render_plots, ...
"""

import os
import sys
import time

import numpy as np

# Repo root = first ancestor that contains config.py (robust to this file being
# at tests/ or tests/graph-tests/, or any depth).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while _ROOT and not os.path.exists(os.path.join(_ROOT, "config.py")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)

import geopandas as gpd
import rasterio
from rasterio.windows import Window
from rasterio.features import rasterize
from skimage.morphology import dilation, disk

import config
from graph.skeletonize import skeletonize_mask
from graph.build_graph import build_graph
from graph.heal import heal_graph
import matplotlib
matplotlib.use("Agg")  # headless-safe plot rendering
from graph.visualize import plot_graph_comparison, plot_healing_result


# ─── Map registry (any map can be added here) ────────────────────────────────
PS4 = os.path.join(_ROOT, "PS4")
DATA_DIR = os.path.join(_HERE, "data")          # sampled masks live here
OUT_DIR = os.path.join(_ROOT, "outputs", "graph-output-tests")  # test plots

CITIES = {
    "bengaluru": {
        "tif": os.path.join(PS4, "SENTINEL2", "Bengaluru.tif"),
        "gpkg": os.path.join(PS4, "SENTINEL2", "Bengaluru_mask.gpkg"),
        "layer": "edges",
        "crs": "EPSG:32643",
    },
    "hyderabad": {
        "tif": os.path.join(PS4, "CARTOSAT3", "Hyderabad.tif"),
        "gpkg": os.path.join(PS4, "CARTOSAT3", "Hyderabad_mask.gpkg"),
        "layer": "hyderabad_road_mask__hyderabad",
        "crs": "EPSG:32644",
    },
    "delhi": {
        "tif": os.path.join(PS4, "RESOURCESAT2", "Delhi.tif"),
        "gpkg": os.path.join(PS4, "RESOURCESAT2", "Delhi_mask.gpkg"),
        "layer": "Delhi_road_mask_32643",
        "crs": "EPSG:32643",
    },
}


# ─── Sampling ────────────────────────────────────────────────────────────────
def sample_window(city, win=(512, 512), center=None, out_path=None):
    """
    Rasterize a map's road-mask layer over a window of the underlying image.

    Returns (mask, subtransform, bbox, meta). Window is clipped to raster
    bounds; center defaults to the image center; any center/win is accepted.
    """
    cfg = CITIES[city]
    with rasterio.open(cfg["tif"]) as src:
        transform, crs = src.transform, src.crs
        gw, gh = win
        c0, r0 = center if center is not None else (
            src.width // 2 - gw // 2,
            src.height // 2 - gh // 2,
        )
        c0 = max(0, min(int(c0), src.width - 1))
        r0 = max(0, min(int(r0), src.height - 1))
        w = min(gw, src.width - c0)
        h = min(gh, src.height - r0)
        window = Window(c0, r0, w, h)
        bounds = rasterio.windows.bounds(window, transform)
        subtransform = rasterio.windows.transform(window, transform)

        gdf = gpd.read_file(cfg["gpkg"], bbox=bounds, layer=cfg["layer"])
        mask = np.zeros((h, w), dtype=np.uint8)
        shapes = (
            (geom, 255)
            for geom in gdf.geometry
            if geom is not None and not geom.is_empty
        )
        rasterize(shapes, out_shape=(h, w), transform=subtransform, fill=0, out=mask)

    meta = {
        "city": city,
        "crs": crs,
        "window": (w, h),
        "offset": (c0, r0),
        "bbox": bounds,
        "foreground_px": int((mask > 0).sum()),
        "foreground_frac": float((mask > 0).mean()),
        "n_features": len(gdf),
    }

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, mask)
        meta["saved"] = out_path

    return mask, subtransform, bounds, meta


# ─── Pipeline (timed) ────────────────────────────────────────────────────────
def run_graph(mask, transform=None, max_distance=None, lambda_weight=None,
              max_gap_m=None, recover_isolated=None):
    """Run skeletonize -> build_graph -> heal_graph, timing each stage."""
    timings = {}

    t0 = time.perf_counter()
    road = dilation((mask > 0).astype(np.uint8), disk(3))
    timings["dilate"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    sk, nodes, edges = skeletonize_mask(road)
    timings["skeletonize"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    G = build_graph(sk, nodes, edges, transform=transform)
    timings["build"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    Gh = heal_graph(
        G,
        max_distance=max_distance,
        lambda_weight=lambda_weight,
        max_gap_m=max_gap_m,
        recover_isolated=recover_isolated,
    )
    timings["heal"] = time.perf_counter() - t0

    return G, Gh, timings, {
        "skeleton_px": int(sk.sum()),
        "nodes_raw": len(nodes),
        "edges_raw": len(edges),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
    }


# ─── Metrics ─────────────────────────────────────────────────────────────────
def _lcc_fraction(G):
    nx = _nx()
    comps = list(nx.connected_components(G))
    if not comps or G.number_of_nodes() == 0:
        return 0.0, len(comps)
    lcc = max(len(c) for c in comps)
    return lcc / G.number_of_nodes(), len(comps)


def _nx():
    import networkx as nx
    return nx


def compute_metrics(G, Gh):
    """Component/LCC/healed-gap/crossing metrics (no centrality here)."""
    nx = _nx()
    comps_before = nx.number_connected_components(G)
    lcc_before, _ = _lcc_fraction(G)
    comps_after = nx.number_connected_components(Gh)
    lcc_after, _ = _lcc_fraction(Gh)

    healed = [(u, v, d) for u, v, d in Gh.edges(data=True) if d.get("healing", False)]
    total_healed = len(healed)

    # A "true bridge" connects two ORIGINAL nodes (int ids). Planarization
    # inserts shared junction nodes with string ids "J..."; it also re-tags the
    # split *original* road segments with healing=True, but those are real road
    # geometry, NOT bridges, and are not subject to the gap cap.
    bridges = [(u, v, d) for u, v, d in healed
               if isinstance(u, int) and isinstance(v, int)]
    max_bridge_gap = max((float(d.get("length", 0.0)) for _, _, d in bridges), default=0.0)

    # Planarize effect: healed edges that cross another edge's interior.
    crossings = _count_crossings(Gh)

    return {
        "comps_before": comps_before,
        "comps_after": comps_after,
        "lcc_before": lcc_before,
        "lcc_after": lcc_after,
        "healed_edges": total_healed,
        "healed_bridges": len(bridges),
        "max_healed_gap": max_bridge_gap,
        "cap": config.HEAL_MAX_GAP_M,
        "crossings": crossings,
    }


def _count_crossings(G):
    """Cheap interior-crossing estimate among healed edges (bbox prefiltered)."""
    healed = [
        (u, v, G.nodes[u].get("pos", (0, 0)), G.nodes[v].get("pos", (0, 0)))
        for u, v, d in G.edges(data=True)
        if d.get("healing", False)
    ]
    all_edges = [
        (u, v, G.nodes[u].get("pos", (0, 0)), G.nodes[v].get("pos", (0, 0)))
        for u, v in G.edges()
    ]
    count = 0
    for a, b, pa, pb in healed:
        for c, e, pc, pe in all_edges:
            if len({a, b, c, e}) < 4:
                continue
            if not _bbox(pa, pb, pc, pe):
                continue
            if _seg(pa, pb, pc, pe):
                count += 1
    return count


def _bbox(a, b, c, d):
    return max(min(a[0], b[0]), min(c[0], d[0])) <= min(max(a[0], b[0]), max(c[0], d[0])) and \
           max(min(a[1], b[1]), min(c[1], d[1])) <= min(max(a[1], b[1]), max(c[1], d[1]))


def _seg(p1, p2, q1, q2, tol=1e-9):
    x1, y1 = p1; x2, y2 = p2
    x3, y3 = q1; x4, y4 = q2
    den = (x2 - x1) * (y4 - y3) - (y2 - y1) * (x4 - x3)
    if abs(den) < tol:
        return False
    t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / den
    u = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / den
    return 0 < t < 1 and 0 < u < 1


def compute_centrality(Gh, k=2000):
    """Betweenness (sampled) + k500/k2000 gatekeeper stability."""
    nx = _nx()
    n = Gh.number_of_nodes()
    if n < 3:
        return None
    k_use = min(k, n - 1)
    cent = nx.betweenness_centrality(Gh, k=k_use, weight="length", normalized=True)
    vals = sorted(cent.values(), reverse=True)
    out = {
        "k": k_use,
        "max": vals[0] if vals else 0.0,
        "mean": float(np.mean(vals)) if vals else 0.0,
        "top10": sorted(cent, key=cent.get, reverse=True)[:10],
    }
    if n >= 500:
        k5 = min(500, n - 1)
        k20 = min(2000, n - 1)
        _c5 = nx.betweenness_centrality(Gh, k=k5, weight="length", normalized=True)
        _c20 = nx.betweenness_centrality(Gh, k=k20, weight="length", normalized=True)
        top5 = set(sorted(_c5, key=_c5.get, reverse=True)[:10])
        top20 = set(sorted(_c20, key=_c20.get, reverse=True)[:10])
        out["overlap_k500_k2000"] = len(top5 & top20) / 10.0
    return out


# ─── Report / plots ──────────────────────────────────────────────────────────
def render_plots(G, Gh, tag, out_dir=None):
    out_dir = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    plot_graph_comparison(
        G, Gh, save=True,
        save_path=os.path.join(out_dir, f"healing_comparison_{tag}.png"),
    )
    plot_healing_result(
        Gh, save=True,
        save_path=os.path.join(out_dir, f"healing_result_{tag}.png"),
    )
    return [
        os.path.join(out_dir, f"healing_comparison_{tag}.png"),
        os.path.join(out_dir, f"healing_result_{tag}.png"),
    ]


def print_report(city, win, meta, g_meta, metrics, timings, cent=None):
    print(f"\n{'=' * 62}")
    print(f"TERRAIN graph pipeline — {city.upper()}  window={win}")
    print(f"{'=' * 62}")
    print(f"crs            : {meta['crs']}")
    print(f"raster window  : {meta['window']}  offset={meta['offset']}")
    print(f"bbox           : {meta['bbox']}")
    print(f"mask features  : {meta['n_features']}")
    print(f"foreground px  : {meta['foreground_px']} ({meta['foreground_frac']:.3%})")
    if meta.get("saved"):
        print(f"mask saved     : {meta['saved']}")

    print(f"\n-- Graph --")
    print(f"skeleton px    : {g_meta['skeleton_px']}")
    print(f"nodes          : {g_meta['nodes']}   edges: {g_meta['edges']}")

    print(f"\n-- Healing --")
    print(f"components     : {metrics['comps_before']} -> {metrics['comps_after']}")
    print(f"LCC fraction   : {metrics['lcc_before']:.4f} -> {metrics['lcc_after']:.4f}")
    print(f"healed edges   : {metrics['healed_edges']} (bridges: {metrics['healed_bridges']})")
    print(f"max bridge gap : {metrics['max_healed_gap']:.1f}  cap={metrics['cap']:.0f}  "
          f"{'OK' if metrics['max_healed_gap'] <= metrics['cap'] else 'EXCEEDED'}")
    print(f"healed crossings: {metrics['crossings']}")

    if cent:
        print(f"\n-- Betweenness k={cent['k']} --")
        print(f"max            : {cent['max']:.4f}")
        print(f"mean           : {cent['mean']:.4f}")
        print(f"top10          : {cent['top10']}")
        if "overlap_k500_k2000" in cent:
            print(f"k500/k2000 overlap: {cent['overlap_k500_k2000']:.0%}")

    print(f"\n-- Timing (s) --")
    for k, v in timings.items():
        print(f"  {k:<13}: {v:8.3f}")
    print(f"  {'total':<13}: {sum(timings.values()):8.3f}")


def measure_scaling(city, windows=(256, 512, 1024), geo=True, out_dir=None):
    """Empirical per-stage wall-clock scaling vs window size."""
    out_dir = out_dir or OUT_DIR
    nx = _nx()
    rows = []
    for win in windows:
        mask, subtransform, _, meta = sample_window(city, win=(win, win))
        if (mask > 0).sum() == 0:
            print(f"window {win}: empty mask, skipping")
            continue
        G, Gh, timings, _ = run_graph(
            mask, transform=subtransform if geo else None
        )
        rows.append((win, G.number_of_nodes(), G.number_of_edges(),
                     Gh.number_of_edges(), timings))
        print(f"win {win}: nodes={G.number_of_nodes()} edges={G.number_of_edges()} "
              f"timings={ {k: round(v, 3) for k, v in timings.items()} }")

    _plot_scaling(rows, city, out_dir)
    return rows


def _plot_scaling(rows, city, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    wins = [r[0] for r in rows]
    stages = ["skeletonize", "build", "heal"]
    fig, ax = plt.subplots(figsize=(8, 6))
    for stage in stages:
        ys = [r[4].get(stage, 0.0) for r in rows]
        ax.plot(wins, ys, marker="o", label=stage)
    ax.set_xlabel("window width (px)")
    ax.set_ylabel("time (s)")
    ax.set_title(f"{city.upper()} — per-stage scaling")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, f"complexity_scaling_{city}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("scaling plot:", path)
