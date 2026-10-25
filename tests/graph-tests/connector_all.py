"""
connector_all.py — combined 3-sensor presentation poster.

Chains each mosaic (cartosat3 / sentinel2 / liss4) through the graph module and
the GCN module (via chain_core.run), re-runs healing with the current config cap,
and merges the three into ONE 3x3 poster:

    input mask | healed network | GCN gatekeepers        (row per sensor)

Writes outputs/graph-output-tests/connector_all_sensors.png.

Run (terrain-torch env):
    conda run -n terrain-torch python tests/graph-tests/connector_all.py
        [--window 512] [--gcn-epochs 120] [--center-override ...]
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while _ROOT and not os.path.exists(os.path.join(_ROOT, "config.py")):
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config
import harness as H
import chain_core
from criticality.gcn_evaluate import rank_gatekeepers
from criticality.graph_adapter import graph_to_tensor

import torch
import numpy as np

SENSORS = [
    ("cartosat3", os.path.join(_ROOT, "cartosat3_mosaic.tif"), (19968, 20736)),
    ("sentinel2", os.path.join(_ROOT, "sentinel2_mosaic.tif"), (2560, 2048)),
    ("liss4", os.path.join(_ROOT, "liss4_mosaic.tif"), (3072, 5888)),
]
TOP_K = 8


def _node_px(G, subtransform):
    inv = ~subtransform
    px = {}
    for n, d in G.nodes(data=True):
        p = d.get("pos")
        if p is None:
            continue
        c, r = inv * (p[0], p[1])
        px[n] = (float(c), float(r))
    return px


def main():
    ap = argparse.ArgumentParser(description="TERRAIN combined 3-sensor poster")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--checkpoint", default=config.MODEL_SAVE_PATH)
    ap.add_argument("--gcn-epochs", type=int, default=120)
    ap.add_argument("--out", default=os.path.join(_ROOT, "outputs", "graph-output-tests"))
    args = ap.parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    fig, axes = plt.subplots(len(SENSORS), 3,              figsize=(len(SENSORS) * 5.6, len(SENSORS) * 4.4))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for i, (tag, tif, center) in enumerate(SENSORS):
        mask, subtransform, crs = chain_core.read_mask_window(tif, (args.window, args.window), center)
        G, Gh, timings, _ = H.run_graph(mask, transform=subtransform,
                                        max_gap_m=config.HEAL_MAX_GAP_M)
        metrics = H.compute_metrics(G, Gh)

        model, model_src = chain_core.make_model(Gh, device, args.checkpoint, args.gcn_epochs)
        x, edge_index, _, node_ids = chain_core.graph_to_tensor(Gh)
        idx = rank_gatekeepers(model, x, edge_index)
        gk_nodes = [node_ids[j] for j in idx.tolist()[:TOP_K]]

        px = _node_px(Gh, subtransform)
        road = [[px[u], px[v]] for u, v, d in Gh.edges(data=True)
                if not d.get("healing", False) and u in px and v in px]
        br = [[px[u], px[v]] for u, v, d in Gh.edges(data=True)
              if d.get("healing", False) and u in px and v in px]
        gk = set(str(n) for n in gk_nodes)

        # col 1: input mask
        ax = axes[i][0]
        ax.imshow((mask > 0), cmap="gray", interpolation="nearest")
        ax.set_title("%s input road mask" % tag.upper(), fontsize=12)
        ax.axis("off")

        # col 2: healed
        ax = axes[i][1]
        ax.imshow((mask > 0), cmap="Greys", alpha=0.22, interpolation="nearest")
        if road:
            ax.add_collection(LineCollection(road, colors="#3b7fdd", linewidths=0.7))
        if br:
            ax.add_collection(LineCollection(br, colors="#e03131", linewidths=1.3))
        ax.set_title("Healed\ncomps %d->%d  LCC %.2f->%.2f" %
                     (metrics["comps_before"], metrics["comps_after"],
                      metrics["lcc_before"], metrics["lcc_after"]), fontsize=11)
        ax.axis("off")

        # col 3: GCN gatekeepers
        ax = axes[i][2]
        ax.imshow((mask > 0), cmap="Greys", alpha=0.22, interpolation="nearest")
        if road:
            ax.add_collection(LineCollection(road, colors=["#3b7fdd"], linewidths=0.6))
        if br:
            ax.add_collection(LineCollection(br, colors=["#e03131"], linewidths=0.9))
        for n in Gh.nodes():
            if n in gk and n in px:
                ax.scatter(*px[n], s=44, c="#c2185b", zorder=5, edgecolors="white", linewidths=0.5)
        ax.set_title("GCN gatekeepers\ntop-%d" % len(gk), fontsize=11)
        ax.axis("off")

        rows.append((tag, crs, metrics, model_src, gk_nodes))
        print("  %-10s %s  comps %d->%d  LCC %.2f->%.2f  bridges %d (cap %d) gcn=%s" %
              (tag, crs, metrics["comps_before"], metrics["comps_after"],
               metrics["lcc_before"], metrics["lcc_after"], metrics["healed_edges"],
               metrics["cap"], model_src))

    for i, (tag, _, _) in enumerate(SENSORS):
        axes[i][0].set_ylabel(tag.upper(), fontsize=15, rotation=0, labelpad=36,
                              va="center", color="#0b3954")

    fig.suptitle("TERRAIN road graph + GCN criticality {3 sensors, hard gap cap %d m}"
                 % config.HEAL_MAX_GAP_M, fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=0.3)
    path = os.path.join(out_dir, "connector_all_sensors.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("poster:", path)
    with open(os.path.join(out_dir, "connector_all_summary.json"), "w") as f:
        import json
        json.dump([{"sensor": r[0], "crs": str(r[1]), "metrics": r[2]}
                   for r in rows], f, indent=2, default=str)


if __name__ == "__main__":
    main()