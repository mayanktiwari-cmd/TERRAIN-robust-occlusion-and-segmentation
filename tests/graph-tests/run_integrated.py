

"""
run_integrated.py — integrated TERRAIN simulation: graph heal + GCN criticality.

End-to-end on a real satellite window:
    rasterize .gpkg mask -> skeletonize -> build_graph -> heal_graph
        -> graph_to_tensor -> RoadGCN (terrain_best.pth) -> gatekeepers
        -> betweenness baseline (for comparison) -> resilience index
        -> outputs (mask, graph.pkl, report.json, plots)

Requires torch; run in the `terrain-torch` env:
    conda run -n terrain-torch python tests/graph-tests/run_integrated.py
      [--city hyderabad] [--window 512] [--center "col,row"]
      [--checkpoint outputs/models/terrain_best.pth] [--top-k 5]
"""

import argparse
import json
import os
import sys

import numpy as np

# graph/visualize.py calls plt.show(); under the headless Agg backend that is a
# harmless no-op but emits a warning. Silence it (plots are still saved as PNG).
import warnings
warnings.filterwarnings("ignore",
                        category=UserWarning,
                        module="matplotlib")

# Repo root (for config/graph/criticality imports) + tests/graph-tests for harness.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = _HERE
while _ROOT and not os.path.exists(os.path.join(_ROOT, "config.py")):
    _ROOT = os.path.dirname(_ROOT)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import config
import harness as H

import torch
from criticality.gcn_model import RoadGCN
from criticality.gcn_evaluate import rank_gatekeepers
from criticality.graph_adapter import graph_to_tensor
from criticality.centrality import compute_centrality, detect_gatekeepers
from criticality.resilience import compute_resilience_index
import chain_core


def _save_gatekeeper_plot(Gh, node_ids, idx, gcn_gate_nodes, out_dir, tag, top_k=5):
    """Highlight the top-K GCN gatekeepers on the healed graph."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = {n: (d.get("pos", (0, 0))) for n, d in Gh.nodes(data=True)}
    gate_ids = set(str(x) for x in gcn_gate_nodes[:top_k])

    plt.figure(figsize=(9, 7))
    nx = sys.modules.get("networkx") or __import__("networkx")
    nx.draw_networkx_edges(Gh, pos, edge_color="#b0b0b0", width=1.0)
    nx.draw_networkx_nodes(Gh, pos, node_size=12, node_color="#4c9be8")
    if gate_ids:
        nodes_hi = [n for n in Gh.nodes() if str(n) in gate_ids]
        nx.draw_networkx_nodes(Gh, pos, nodelist=nodes_hi, node_size=90,
                               node_color="crimson")
        nx.draw_networkx_labels(Gh, pos, labels={n: "" for n in nodes_hi})
    plt.title(f"{tag.upper()} — GCN gatekeepers (top-{len(gate_ids)})")
    plt.tight_layout()
    path = os.path.join(out_dir, f"gcn_gatekeepers_{tag}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  plot : {path}")


def main():
    ap = argparse.ArgumentParser(description="Integrated TERRAIN graph-heal + GCN run")
    ap.add_argument("--city", default="hyderabad")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--center", default=None, help='quoted "col,row" in PowerShell')
    ap.add_argument("--checkpoint", default=config.MODEL_SAVE_PATH)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--gcn-epochs", type=int, default=120)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    center = tuple(int(x) for x in args.center.split(",")) if args.center else None

    out_dir = os.path.join(_ROOT, "outputs", "graph-output-tests")
    os.makedirs(out_dir, exist_ok=True)
    mask_path = os.path.join(_HERE, "data", f"{args.city}_graph_gcn_mask.npy")

    # ---- 1. mask (gpkg stand-in for the not-yet-built segmentation) ----
    mask, subtransform, bounds, meta = H.sample_window(
        args.city, win=(args.window, args.window), center=center, out_path=mask_path
    )
    if (mask > 0).sum() == 0:
        print(f"[integrated] empty mask at center={center}; pick a road window")
        return

    # ---- 2. graph pipeline (heal) ----
    G, Gh, timings, g_meta = H.run_graph(mask, transform=subtransform,
                                         max_gap_m=config.HEAL_MAX_GAP_M)
    metrics = H.compute_metrics(G, Gh)

    # ---- 3. GCN inference ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_src = chain_core.make_model(Gh, device, args.checkpoint, args.gcn_epochs)

    # x (n,2) from degree + clustering; y=None at inference; node_ids align to rows.
    x, edge_index, _, node_ids = graph_to_tensor(Gh, centrality=None)
    idx = rank_gatekeepers(model, x, edge_index)
    gcn_gate_nodes = [node_ids[i] for i in idx.tolist()]

    # ---- 4. baseline + resilience ----
    centrality = compute_centrality(Gh, k=config.BETWEENNESS_K)
    bc_gate_nodes = detect_gatekeepers(centrality)[:args.top_k]
    resilience_index, removal_log = compute_resilience_index(Gh, centrality, top_k=10)

    # GCN-vs-betweenness overlap on the top-k
    top_k_set = set(str(x) for x in gcn_gate_nodes[:args.top_k])
    bc_set = set(str(x) for x in bc_gate_nodes)
    overlap = len(top_k_set & bc_set) / max(1, args.top_k)

    # ---- 5. outputs ----
    if not args.no_plot:
        H.render_plots(G, Gh, f"{args.city}_gcn", out_dir=out_dir)
        _save_gatekeeper_plot(Gh, node_ids, idx, gcn_gate_nodes,
                              out_dir=out_dir, tag=args.city)

    gpath = os.path.join(out_dir, f"{args.city}_gcn_graph.pkl")
    with open(gpath, "wb") as f:
        import pickle
        pickle.dump(Gh, f)

    report = {
        "city": args.city,
        "device": str(device),
        "checkpoint": args.checkpoint,
        "window": args.window,
        "center": center,
        "crs": str(meta["crs"]),
        "bbox": list(bounds),
        "foreground_px": meta["foreground_px"],
        "graph": {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                  "healed_nodes": Gh.number_of_nodes(), "healed_edges": Gh.number_of_edges()},
        "metrics": metrics,
        "gcn": {
            "n_gatekeepers": len(gcn_gate_nodes),
            "top": [str(n) for n in gcn_gate_nodes[:args.top_k]],
            "baseline_top": [str(n) for n in bc_gate_nodes],
            "gcn_vs_betweenness_topK_overlap": round(overlap, 4),
        },
        "resilience_index": round(float(resilience_index), 5),
        "removal_steps": len(removal_log),
        "timings": {k: round(v, 4) for k, v in timings.items()},
    }
    rpath = os.path.join(out_dir, f"{args.city}_gcn_report.json")
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 66)
    print(f"TERRAIN integrated sim — {args.city.upper()} window={args.window} center={center}")
    print("=" * 66)
    print(f"device         : {device}")
    print(f"mask fg px     : {meta['foreground_px']}  bbox={bounds}")
    print("-- Graph+Heal --")
    print(f"  nodes/edges   : {G.number_of_nodes()}/{G.number_of_edges()} -> "
          f"{Gh.number_of_nodes()}/{Gh.number_of_edges()}")
    print(f"  components    : {metrics['comps_before']} -> {metrics['comps_after']}"
          f"  LCC {metrics['lcc_before']:.3f} -> {metrics['lcc_after']:.3f}")
    print(f"  max bridge gap: {metrics['max_healed_gap']:.1f} m  "
          f"cap={metrics['cap']:.0f} m  "
          f"{'OK' if metrics['max_healed_gap'] <= metrics['cap'] else 'EXCEEDED'}")
    print("-- GCN gatekeepers --")
    print(f"  top {args.top_k}: {[str(n) for n in gcn_gate_nodes[:args.top_k]]}")
    print(f"  vs betweenness topK overlap: {overlap:.2%}")
    print(f"  resilience index: {resilience_index:.5f}  ({len(removal_log)} removal steps)")
    print("-- artifacts --")
    print(f"  mask : {mask_path}")
    print(f"  graph: {gpath}")
    print(f"  json : {rpath}")


if __name__ == "__main__":
    main()