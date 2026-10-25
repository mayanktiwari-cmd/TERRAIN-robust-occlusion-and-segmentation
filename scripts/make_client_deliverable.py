"""
make_client_deliverable.py — polished client-facing visuals from the three
committed road-mask mosaics (cartosat3 / sentinel2 / liss4).

For each sensor:
  1. auto-detect the road-densest 512x512 window in its mosaic  (rasterio)
  2. skeletonize -> build_graph -> heal_graph                    (graph/)
  3. betweenness centrality -> gatekeepers -> resilience index  (criticality/)
  4. render a 4-panel storyboard AND a combined 3-row poster    (matplotlib)

Outputs (torch-free, runs in the `terrain-graph` env):
  outputs/client_deliverables/client_<sensor>.png     per-sensor storyboard
  outputs/client_deliverables/client_all_sensors.png  combined poster
  outputs/client_deliverables/client_metrics.json     metrics consumed by the report

Run:
    conda run -n terrain-graph python scripts/make_client_deliverable.py [--window 512]
"""

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while _ROOT and not os.path.exists(os.path.join(_ROOT, "config.py")):
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import matplotlib
matplotlib.use("Agg")

import config
from graph.skeletonize import skeletonize_mask
from graph.build_graph import build_graph
from graph.heal import heal_graph
from criticality.centrality import compute_centrality, detect_gatekeepers
from criticality.resilience import compute_resilience_index

SENSORS = [
    ("cartosat3", os.path.join(_ROOT, "cartosat3_mosaic.tif")),
    ("sentinel2", os.path.join(_ROOT, "sentinel2_mosaic.tif")),
    ("liss4", os.path.join(_ROOT, "liss4_mosaic.tif")),
]
DISPLAY = {
    "cartosat3": "Cartosat-3  ·  Hyderabad (1 m GSD, optical)",
    "sentinel2": "Sentinel-2  ·  Bengaluru (10 m GSD, multispectral)",
    "liss4": "LISS-IV  ·  Delhi (5.8 m GSD, multispectral)",
}


# ─── window I/O ──────────────────────────────────────────────────────────────
def read_window(tif, win, center):
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(tif) as src:
        gw, gh = win
        c0, r0 = center
        c0 = max(0, min(int(c0), src.width - 1))
        r0 = max(0, min(int(r0), src.height - 1))
        w = min(gw, src.width - c0)
        h = min(gh, src.height - r0)
        window = Window(c0, r0, w, h)
        subtransform = rasterio.windows.transform(window, src.transform)
        mask = src.read(1, window=window, out_dtype=np.uint8)
        crs = src.crs
    return (mask > 0).astype(np.uint8), subtransform, crs


def detect_dense_center(path, win=512, probes=49):
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(path) as src:
        gw, gh = src.width, src.height
        if gw <= win or gh <= win:
            return (gw // 2, gh // 2)
        n = int(np.sqrt(probes))
        cols = np.linspace(0, gw - win, n)
        rows = np.linspace(0, gh - win, n)
        best = (0.0, None)
        for c in cols:
            for r in rows:
                c0, r0 = int(c), int(r)
                chunk = src.read(1, window=Window(c0, r0, win, win),
                                 out_dtype=np.uint8)
                frac = float((chunk > 0).mean())
                if frac > best[0]:
                    best = (frac, (c0 + win // 2, r0 + win // 2))
        return best[1] if best[1] else (gw // 2, gh // 2)


# ─── helpers ─────────────────────────────────────────────────────────────────
def _nx():
    import networkx as nx
    return nx


def _lcc_fraction(G):
    nx = _nx()
    comps = list(nx.connected_components(G))
    if not comps or G.number_of_nodes() == 0:
        return 0.0
    return max(len(c) for c in comps) / G.number_of_nodes()


# ─── pipeline ────────────────────────────────────────────────────────────────
def run_pipeline(mask, transform):
    from skimage.morphology import dilation, disk
    nx = _nx()
    road = dilation((mask > 0).astype(np.uint8), disk(3))
    sk, nodes, edges = skeletonize_mask(road)
    G = build_graph(sk, nodes, edges, transform=transform)
    Gh = heal_graph(G, max_gap_m=config.HEAL_MAX_GAP_M)
    cent = compute_centrality(Gh, k=config.BETWEENNESS_K)
    gate = detect_gatekeepers(cent)[:8]
    res, _log = compute_resilience_index(Gh, cent, top_k=10)

    comps_before = nx.number_connected_components(G)
    comps_after = nx.number_connected_components(Gh)
    healed = [(u, v, d) for u, v, d in Gh.edges(data=True) if d.get("healing", False)]
    bridges = [(u, v, d) for u, v, d in healed if all(isinstance(x, int) for x in (u, v))]
    max_gap = float(max((d.get("length", 0.0) for _, _, d in bridges), default=0.0))

    info = {
        "skeleton_px": int(sk.sum()),
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "healed_nodes": Gh.number_of_nodes(),
        "healed_edges": Gh.number_of_edges(),
        "comps_before": int(comps_before),
        "comps_after": int(comps_after),
        "lcc_before": round(_lcc_fraction(G), 4),
        "lcc_after": round(_lcc_fraction(Gh), 4),
        "healed_bridges": int(len(bridges)),
        "max_healed_gap_m": round(max_gap, 1),
        "gap_cap_m": config.HEAL_MAX_GAP_M,
        "n_gatekeepers": int(len(gate)),
        "betweenness_max": round(float(max(cent.values())) if cent else 0.0, 5),
        "resilience": round(float(res), 4),
        "foreground_frac": round(float((mask > 0).mean()), 4),
    }
    return G, Gh, cent, gate, info


# ── plotting helpers ─────────────────────────────────────────────────────────
def _node_px(G, transform):
    import rasterio
    inv = ~transform
    px = {}
    for n, d in G.nodes(data=True):
        p = d.get("pos")
        if p is None:
            continue
        c, r = inv * (p[0], p[1])
        px[n] = (float(c), float(r))
    return px


def _lines(Gh, px, which):
    out = []
    for u, v, d in Gh.edges(data=True):
        is_heal = d.get("healing", False)
        if which == "road" and is_heal:
            continue
        if which == "bridge" and not is_heal:
            continue
        if u in px and v in px:
            out.append([px[u], px[v]])
    return out


def _draw_base(ax, mask):
    ax.set_facecolor("#0b1722")
    ax.imshow((mask > 0), cmap="Greys", alpha=0.22,
              interpolation="nearest", zorder=0)
    return ax


def render_storyboard(tag, display, mask, Gh, transform, gate, info, out_path):
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt

    px = _node_px(Gh, transform)
    road = _lines(Gh, px, "road")
    bridge = _lines(Gh, px, "bridge")
    gk = set(str(n) for n in gate)

    fig = plt.figure(figsize=(16, 10.5))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.06)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow((mask > 0), cmap="gray", interpolation="nearest")
    ax.set_title("1. Extracted road mask\n(Swin-UNet segmentation)",
                 fontsize=13)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    _draw_base(ax, mask)
    if road:
        ax.add_collection(LineCollection(road, colors="#4aa8ff", linewidths=1.0))
    if bridge:
        ax.add_collection(LineCollection(bridge, colors="#ff5a5a", linewidths=1.5))
    ax.set_title("2. Healed road network\ncomponents %d -> %d · LCC %.0f%% -> %.0f%%"
                 % (info["comps_before"], info["comps_after"],
                    info["lcc_before"] * 100, info["lcc_after"] * 100), fontsize=12)
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 0])
    _draw_base(ax, mask)
    if road:
        ax.add_collection(LineCollection(road, colors="#4a8aff", linewidths=1.0))
    if bridge:
        ax.add_collection(LineCollection(bridge, colors="#ffd166", linewidths=2.2))
    ax.set_title("3. Occlusion bridges\n%d healed bridges · max gap %.0f m (cap %d m)"
                 % (info["healed_bridges"], info["max_healed_gap_m"],
                    info["gap_cap_m"]), fontsize=12)
    ax.axis("off")

    ax = fig.add_subplot(gs[1, 1])
    _draw_base(ax, mask)
    if road:
        ax.add_collection(LineCollection(road, colors="#4a8ff0", linewidths=0.8))
    if bridge:
        ax.add_collection(LineCollection(bridge, colors="#ff5a5a", linewidths=1.0))
    for n in Gh.nodes():
        if str(n) in gk and n in px:
            ax.scatter(*px[n], s=60, c="#c2185b", zorder=5,
                       edgecolors="white", linewidths=0.6)
    ax.set_title("4. Critical gatekeepers (top-%d)\nresilience index = %.3f"
                 % (info["n_gatekeepers"], info["resilience"]), fontsize=12)
    ax.axis("off")

    handles = [
        Line2D([0], [0], color="#4aa8ff", lw=2, label="Roads (skeleton)"),
        Line2D([0], [0], color="#ff5a5a", lw=2, label="Healed bridges"),
        Line2D([0], [0], color="#ffd166", lw=2, label="Highest-priority bridges"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#c2185b",
               markersize=9, label="Critical gatekeepers"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=True, fontsize=10)

    fig.suptitle("TERRAIN Pipeline  —  %s\n"
                 "skeletonize -> graph -> topological occlusion healing -> criticality"
                 % display, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def render_poster(rows, out_path):
    from matplotlib.collections import LineCollection
    import matplotlib.pyplot as plt

    n = len(rows)
    fig, axes = plt.subplots(n, 3, figsize=(n * 5.6, n * 4.4))
    if n == 1:
        axes = axes[None, :]
    for i, (tag, mask, Gh, transform, info) in enumerate(rows):
        px = _node_px(Gh, transform)
        road = _lines(Gh, px, "road")
        bridge = _lines(Gh, px, "bridge")
        gk = set()

        ax = axes[i][0]
        ax.imshow((mask > 0), cmap="gray", interpolation="nearest")
        ax.set_title("%s road mask" % tag.upper(), fontsize=12)
        ax.axis("off")

        ax = axes[i][1]
        _draw_base(ax, mask)
        if road:
            ax.add_collection(LineCollection(road, colors="#4aa8ff", linewidths=0.8))
        if bridge:
            ax.add_collection(LineCollection(bridge, colors="#ff5a5a", linewidths=1.4))
        ax.set_title("Healed network\ncomps %d->%d · LCC %.0f%%->%.0f%%"
                     % (info["comps_before"], info["comps_after"],
                        info["lcc_before"] * 100, info["lcc_after"] * 100),
                     fontsize=11)
        ax.axis("off")

        ax = axes[i][2]
        _draw_base(ax, mask)
        if road:
            ax.add_collection(LineCollection(road, colors="#4a8aff", linewidths=0.7))
        if bridge:
            ax.add_collection(LineCollection(bridge, colors="#ff5a5a", linewidths=1.0))
        for nod in Gh.nodes():
            if str(nod) in gk and nod in px:
                ax.scatter(*px[nod], s=42, c="#c2185b", zorder=5,
                           edgecolors="white", linewidths=0.5)
        ax.set_title("Critical gatekeepers\nresilience %.3f · bridges %d"
                     % (info["resilience"], info["healed_bridges"]), fontsize=11)
        ax.axis("off")
        axes[i][0].set_ylabel(tag.upper(), fontsize=15, rotation=0, labelpad=40,
                              va="center", color="#0b3954")

    fig.suptitle("TERRAIN — multi-sensor road network + resilience analysis (3 sensors, gap cap %d m)"
                 % config.HEAL_MAX_GAP_M, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.4)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="TERRAIN client-facing visuals")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--out", default=os.path.join(_ROOT, "outputs", "client_deliverables"))
    ap.add_argument("--sensor", default=None, help="limit to one sensor (e.g. cartosat3)")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    metrics_out = {}
    rows = []

    for tag, tif in SENSORS:
        if args.sensor and args.sensor != tag:
            continue
        if not os.path.exists(tif):
            print("[skip] missing mosaic:", tif)
            continue
        center = detect_dense_center(tif, args.window)
        mask, transform, crs = read_window(tif, (args.window, args.window), center)
        G, Gh, cent, gate, info = run_pipeline(mask, transform)
        info["window"] = args.window
        info["center"] = list(center)
        info["crs"] = str(crs)
        info["display"] = DISPLAY.get(tag, tag)

        story = os.path.join(args.out, "client_%s.png" % tag)
        render_storyboard(tag, info["display"], mask, Gh, transform, gate, info, story)

        rows.append((tag, mask, Gh, transform, info))
        metrics_out[tag] = info
        print("[%s] %s  comps %d->%d  LCC %.2f->%.2f  bridges %d  res %.3f  -> %s"
              % (tag, info["display"], info["comps_before"], info["comps_after"],
                 info["lcc_before"], info["lcc_after"], info["healed_bridges"],
                 info["resilience"], story))

    if rows:
        poster = os.path.join(args.out, "client_all_sensors.png")
        render_poster(rows, poster)
        print("poster:", poster)

    with open(os.path.join(args.out, "client_metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    print("metrics:", os.path.join(args.out, "client_metrics.json"))


if __name__ == "__main__":
    main()