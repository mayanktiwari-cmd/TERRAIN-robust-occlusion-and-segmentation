"""
chain_core.py — shared pipeline core for the three sensor-specific connectors
(connector_cartosat3 / connector_sentinel2 / connector_liss4).

Each sensor's BINARY road-mask mosaic is chained through the graph module
(skeletonize -> build -> heal) and into the GCN module (graph_to_tensor ->
RoadGCN -> rank_gatekeepers -> resilience). The GCN is loaded from a checkpoint
or trained on the healed graph on the fly. Produces a highlighted visual, a
report JSON and the healed graph .pkl.
"""

import json
import os
import sys
import pickle

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while _ROOT and not os.path.exists(os.path.join(_ROOT, "config.py")):
    _ROOT = os.path.dirname(_ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config
import harness as H

import torch
import rasterio
from rasterio.windows import Window
from criticality.gcn_model import RoadGCN
from criticality.gcn_evaluate import rank_gatekeepers
from criticality.graph_adapter import graph_to_tensor
from criticality.centrality import compute_centrality, detect_gatekeepers
from criticality.resilience import compute_resilience_index


def read_mask_window(tif, win=(256, 256), center=None):
    """Read a road-mask window straight from a mosaic tif (1-band uint8)."""
    with rasterio.open(tif) as src:
        gw, gh = win
        if center is None:
            c0, r0 = src.width // 2 - gw // 2, src.height // 2 - gh // 2
        else:
            c0, r0 = center
        c0 = max(0, min(int(c0), src.width - 1))
        r0 = max(0, min(int(r0), src.height - 1))
        w = min(gw, src.width - c0)
        h = min(gh, src.height - r0)
        window = Window(c0, r0, w, h)
        subtransform = rasterio.windows.transform(window, src.transform)
        mask = src.read(1, window=window)
        crs = src.crs
    return (mask > 0).astype(np.uint8), subtransform, crs


def make_model(Gh, device, checkpoint, epochs=120):
    """Load a GCN checkpoint, or train a fresh RoadGCN on the healed graph."""
    if checkpoint and os.path.exists(checkpoint):
        model = RoadGCN()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        return model.to(device).eval(), "checkpoint"
    from criticality.gcn_train import train_gcn
    centrality = compute_centrality(Gh)
    model, loss = train_gcn(Gh, centrality, epochs=epochs, device=device)
    return model.to(device).eval(), "trained(loss=%.4f)" % float(loss)


def _node_px(G, subtransform):
    """geo pos -> local pixel dict for plotting onto the mask window."""
    inv = ~subtransform
    px = {}
    for n, d in G.nodes(data=True):
        p = d.get("pos")
        if p is None:
            continue
        c, r = inv * (p[0], p[1])
        px[n] = (float(c), float(r))
    return px


def _render(tag, mask, Gh, subtransform, metrics, gk_nodes, resilience_index,
            center, crs, out_dir, top_k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    px = _node_px(Gh, subtransform)
    road = [[px[u], px[v]] for u, v, d in Gh.edges(data=True)
            if not d.get("healing", False) and u in px and v in px]
    br = [[px[u], px[v]] for u, v, d in Gh.edges(data=True)
          if d.get("healing", False) and u in px and v in px]
    gk = set(str(n) for n in gk_nodes[:top_k])

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(1, 2, wspace=0.06)

    ax = fig.add_subplot(gs[0, 0])
    ax.imshow((mask > 0), cmap="Greys", alpha=0.3, interpolation="nearest")
    if road:
        ax.add_collection(LineCollection(road, colors="#3b7fdd", linewidths=0.8))
    if br:
        ax.add_collection(LineCollection(br, colors="#e03131", linewidths=1.3))
    ax.set_title("Healed network\ncomps %d->%d  LCC %.2f->%.2f" %
                 (metrics["comps_before"], metrics["comps_after"],
                  metrics["lcc_before"], metrics["lcc_after"]), fontsize=11)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow((mask > 0), cmap="Greys", alpha=0.3, interpolation="nearest")
    if road:
        ax.add_collection(LineCollection(road, colors="#3b7fdd", linewidths=0.8))
    if br:
        ax.add_collection(LineCollection(br, colors="#e03131", linewidths=1.0))
    for n in Gh.nodes():
        if str(n) in gk and n in px:
            ax.scatter(*px[n], s=46, c="#c2185b", zorder=5, edgecolors="white",
                       linewidths=0.5)
    ax.set_title("GCN gatekeepers (top-%d)\nresilience=%.3f" %
                 (len(gk), resilience_index), fontsize=11)
    ax.axis("off")

    handles = [Line2D([0], [0], color="#3b7fdd", lw=2, label="Roads"),
               Line2D([0], [0], color="#e03131", lw=2, label="Healed bridges"),
               Line2D([0], [0], marker="o", color="w",
                      markerfacecolor="#c2185b", markersize=8,
                      label="GCN gatekeepers")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11)
    fig.suptitle("TERRAIN %s mask -> graph -> GCN   %dx%d @ %s  crs=%s" %
                 (tag.upper(), mask.shape[1], mask.shape[0], center, crs),
                 fontsize=12)
    path = os.path.join(out_dir, "connector_%s.png" % tag)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def run(tif, tag, center, checkpoint, out_dir, window=512, gcn_epochs=120,
        top_k=8, no_plot=False):
    """Full mask -> graph -> GCN chain for one sensor mosaic."""
    os.makedirs(out_dir, exist_ok=True)

    mask, subtransform, crs = read_mask_window(tif, (window, window), center)
    if (mask > 0).sum() == 0:
        print("[%s] empty mask at center=%s; pick a road window" % (tag, center))
        return None

    G, Gh, timings, _ = H.run_graph(mask, transform=subtransform,
                                    max_gap_m=config.HEAL_MAX_GAP_M)
    metrics = H.compute_metrics(G, Gh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_src = make_model(Gh, device, checkpoint, gcn_epochs)

    x, edge_index, _, node_ids = graph_to_tensor(Gh, centrality=None)
    idx = rank_gatekeepers(model, x, edge_index)
    gcn_gate_nodes = [node_ids[i] for i in idx.tolist()]

    centrality = compute_centrality(Gh, k=config.BETWEENNESS_K)
    bc_gate = detect_gatekeepers(centrality)[:top_k]
    try:
        resilience_index, removal_log = compute_resilience_index(Gh, centrality, top_k=10)
    except Exception:
        import traceback
        traceback.print_exc()
        resilience_index, removal_log = float("nan"), []
    top_set = set(str(n) for n in gcn_gate_nodes[:top_k])
    overlap = len(top_set & set(str(n) for n in bc_gate)) / max(1, top_k)

    vis = None
    if not no_plot:
        vis = _render(tag, mask, Gh, subtransform, metrics, gcn_gate_nodes,
                      resilience_index, center, crs, out_dir, top_k)

    graph_pkl = os.path.join(out_dir, "connector_%s_graph.pkl" % tag)
    with open(graph_pkl, "wb") as f:
        pickle.dump(Gh, f)

    report = {
        "tag": tag, "crs": str(crs), "window": list(mask.shape),
        "center": center,
        "graph": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                  "healed_nodes": Gh.number_of_nodes(), "healed_edges": Gh
                  .number_of_edges()},
        "metrics": metrics, "gcn_source": model_src,
        "gcn": {"n_gatekeepers": len(gcn_gate_nodes),
                "top": [str(n) for n in gcn_gate_nodes[:top_k]],
                "baseline_top": [str(n) for n in bc_gate],
                "gcn_vs_betweenness_topK_overlap": round(overlap, 4)},
        "resilience_index": round(float(resilience_index), 5),
        "removal_steps": len(removal_log),
        "timings": {k: round(v, 4) for k, v in timings.items()},
    }
    rpath = os.path.join(out_dir, "connector_%s_report.json" % tag)
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("visual:", vis)
    print("report:", rpath)
    return report