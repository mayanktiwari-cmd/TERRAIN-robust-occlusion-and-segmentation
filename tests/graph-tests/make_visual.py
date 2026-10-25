"""
make_visual.py — pipeline demonstration storyboard.

Produces a single 4-panel figure per city/window telling the "how it works"
story of TERRAIN:

  [1] Ground truth road mask      [2] Raw fragmented network (before heal)
  [3] Healed network (bridges)    [4] Healed network + GCN gatekeepers

plus a metrics footer (components, LCC, max gap, gatekeepers, resilience).

Requires torch; run in the `terrain-torch` env:
    conda run -n terrain-torch python tests/graph-tests/make_visual.py
      [--city hyderabad] [--window 256] [--center "col,row"] [--top-k 8]
      [--out outputs/graph-output-tests]
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while _ROOT and not os.path.exists(os.path.join(_ROOT, "config.py")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import config
import harness as H

import torch
import chain_core
from criticality.gcn_model import RoadGCN
from criticality.gcn_evaluate import rank_gatekeepers
from criticality.graph_adapter import graph_to_tensor
from criticality.resilience import compute_resilience_index


def _local_px(G, subtransform):
    """geo pos -> local pixel dict for plotting."""
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
    ap = argparse.ArgumentParser(description="TERRAIN pipeline storyboard")
    ap.add_argument("--city", default="hyderabad")
    ap.add_argument("--window", type=int, default=256)
    ap.add_argument("--center", default=None, help='quoted "col,row"')
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--checkpoint", default=config.MODEL_SAVE_PATH)
    ap.add_argument("--gcn-epochs", type=int, default=120)
    ap.add_argument("--out", default=os.path.join(_ROOT, "outputs", "graph-output-tests"))
    args = ap.parse_args()

    center = tuple(int(x) for x in args.center.split(",")) if args.center else None
    os.makedirs(args.out, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    # ---- run pipeline ----
    mask, subtransform, _, meta = H.sample_window(
        args.city, win=(args.window, args.window), center=center)
    if (mask > 0).sum() == 0:
        print(f"[visual] empty mask at center={center}; pick a road window")
        return
    G, Gh, timings, _ = H.run_graph(mask, transform=subtransform,
                                    max_gap_m=config.HEAL_MAX_GAP_M)
    metrics = H.compute_metrics(G, Gh)

    # GCN (checkpoint or train on-the-fly)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _src = chain_core.make_model(Gh, device, args.checkpoint,
                                        epochs=args.gcn_epochs)
    x, edge_index, _, node_ids = graph_to_tensor(Gh, centrality=None)
    idx = rank_gatekeepers(model, x, edge_index)
    gk = [node_ids[i] for i in idx.tolist()[:args.top_k]]
    gk_set = set(str(n) for n in gk)
    from criticality.centrality import compute_centrality
    cent = compute_centrality(Gh, k=config.BETWEENNESS_K)
    res_idx, _ = compute_resilience_index(Gh, cent, top_k=10)

    pxG, pxGh = _local_px(G, subtransform), _local_px(Gh, subtransform)

    # ---- build figure ----
    fig = plt.figure(figsize=(14, 10.5))
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.08)

    # P1: truth mask
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow((mask > 0), cmap="gray_r", interpolation="nearest")
    ax.set_title("1. Ground-truth road mask", fontsize=12)
    ax.axis("off")

    # P2: raw fragmented graph
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow((mask > 0), cmap="Greys", alpha=0.35, interpolation="nearest")
    raw_lines = [[pxG[u], pxG[v]] for u, v in G.edges() if u in pxG and v in pxG]
    if raw_lines:
        ax.add_collection(LineCollection(raw_lines, colors="#3b7fdd", linewidths=1.0))
    ax.set_title(f"2. Raw fragmented network  (components={metrics['comps_before']})",
                 fontsize=12, color="#b00000")
    ax.axis("off")

    # P3: healed graph (roads blue, bridges red)
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow((mask > 0), cmap="Greys", alpha=0.25, interpolation="nearest")
    road = [[pxGh[u], pxGh[v]] for u, v, d in Gh.edges(data=True)
            if not d.get("healing", False) and u in pxGh and v in pxGh]
    br = [[pxGh[u], pxGh[v]] for u, v, d in Gh.edges(data=True)
          if d.get("healing", False) and u in pxGh and v in pxGh]
    if road:
        ax.add_collection(LineCollection(road, colors="#3b7fdd", linewidths=1.0))
    if br:
        ax.add_collection(LineCollection(br, colors="#e03131", linewidths=1.4))
    ax.set_title(f"3. Healed network  (components→{metrics['comps_after']}, "
                 f"LCC {metrics['lcc_before']:.2f}→{metrics['lcc_after']:.2f})",
                 fontsize=12)
    ax.axis("off")

    # P4: healed + gatekeepers
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow((mask > 0), cmap="Greys", alpha=0.25, interpolation="nearest")
    if road:
        ax.add_collection(LineCollection(road, colors="#3b7fdd", linewidths=1.0))
    if br:
        ax.add_collection(LineCollection(br, colors="#e03131", linewidths=1.2))
    for n in Gh.nodes():
        if str(n) in gk_set and n in pxGh:
            ax.scatter(*pxGh[n], s=45, c="#c2185b", zorder=5, edgecolors="white",
                       linewidths=0.5)
    ax.set_title(f"4. GCN gatekeepers (top-{len(gk)})   resilience={res_idx:.3f}",
                 fontsize=12)
    ax.axis("off")

    # footer legend
    handles = [
        Line2D([0], [0], color="#3b7fdd", lw=2, label="Roads"),
        Line2D([0], [0], color="#e03131", lw=2, label="Healed bridges"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#c2185b",
               markersize=8, label="GCN gatekeepers"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True, fontsize=11)

    fig.suptitle(f"TERRAIN pipeline — {args.city.upper()}  "
                 f"{args.window}×{args.window}  @ center {center}\n"
                 f"bridges={metrics['healed_edges']} · max gap "
                 f"{metrics['max_healed_gap']:.1f} m (cap {metrics['cap']:.0f} m) · "
                 f"gatekeepers={len(gk)}",
                 fontsize=13)

    path = os.path.join(args.out, f"visual_pipeline_{args.city}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", path)


if __name__ == "__main__":
    main()
