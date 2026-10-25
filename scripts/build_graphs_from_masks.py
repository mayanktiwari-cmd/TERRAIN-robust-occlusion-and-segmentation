"""
Build road graphs + criticality outputs from predicted road masks.

Consumes the masks produced by scripts/predict_all_masks.py (data/masked/<sensor>/mask/)
and runs the full graph chain per tile:
    mask -> skeletonize_mask -> build_graph(transform) -> heal_graph
    -> centrality -> gatekeepers -> resilience index
Saving for the friend's graph part:
    outputs/graphs/<sensor>/graph_<tile>.graphml   (networkx interchange)
    outputs/graphs/<sensor>/summary.json           (nodes/edges/gatekeepers/resilience)

Usage:
    python scripts/build_graphs_from_masks.py --sensors liss4 sentinel2 --limit 20
    python scripts/build_graphs_from_masks.py --limit 5 --graphml 3
"""

import argparse
import glob
import json
import os
import sys

import networkx as nx
import numpy as np
import rasterio

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402
from graph.skeletonize import skeletonize_mask  # noqa: E402
from graph.build_graph import build_graph  # noqa: E402
from graph.heal import heal_graph  # noqa: E402
from criticality.centrality import compute_centrality, detect_gatekeepers  # noqa: E402
from criticality.resilience import compute_resilience_index  # noqa: E402


def load_mask(mask_path):
    with rasterio.open(mask_path) as ds:
        arr = ds.read(1)
        transform = ds.transform
    return (arr > 0).astype(np.uint8), transform


def build_road_graph(mask, transform):
    skeleton, nodes, edges = skeletonize_mask(mask)
    G = build_graph(skeleton, nodes, edges, transform=transform)
    return heal_graph(G)


def run_criticality(G):
    centrality = compute_centrality(G, k=config.BETWEENNESS_K)
    gatekeepers = detect_gatekeepers(centrality)
    resilience_index, removal_log = compute_resilience_index(G, centrality)
    return centrality, gatekeepers, resilience_index


def _sanitize_graph(G):
    """GraphML-safe copy: scalar/string attrs only (drops tuple pixel/path lists)."""
    out = nx.Graph()
    for n, d in G.nodes(data=True):
        a = {}
        pos = d.get("pos")
        if isinstance(pos, (tuple, list)) and len(pos) == 2:
            a["x"], a["y"] = float(pos[0]), float(pos[1])
        if isinstance(d.get("pixel"), (tuple, list)):
            a["pixel"] = "({}, {})".format(int(d["pixel"][0]), int(d["pixel"][1]))
        for k in ("lat", "lon", "lng"):
            if isinstance(d.get(k), (int, float)):
                a[k] = float(d[k])
        out.add_node(str(n), **a)
    for u, v, d in G.edges(data=True):
        a = {k: d[k] for k in ("length",) if k in d and isinstance(d[k], (int, float))}
        out.add_edge(str(u), str(v), **a)
    return out


def process_sensor(sensor, limit, graphml, out_dir):
    masks = sorted(glob.glob(os.path.join("data/masked", sensor, "mask", "mask_tile_*.tif")))
    if limit > 0:
        masks = masks[:limit]
    if not masks:
        print(f"[skip] {sensor}: no masks")
        return
    sdir = os.path.join(out_dir, sensor)
    os.makedirs(sdir, exist_ok=True)
    summary = []
    for i, mask_path in enumerate(masks):
        mask, transform = load_mask(mask_path)
        G = build_road_graph(mask, transform)
        centrality, gatekeepers, resilience_index = run_criticality(G)
        entry = {
            "mask": os.path.basename(mask_path),
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "n_gatekeepers": len(gatekeepers),
            "resilience_index": round(float(resilience_index), 4),
        }
        summary.append(entry)
        if graphml > 0 and i < graphml:
            base = os.path.splitext(entry["mask"])[0]
            gpath = os.path.join(sdir, f"graph_{base}.graphml")
            nx.write_graphml(_sanitize_graph(G), gpath)
        if (i + 1) % 20 == 0:
            print(f"  {sensor}: {i+1}/{len(masks)}", flush=True)
    with open(os.path.join(sdir, "summary.json"), "w") as f:
        json.dump({"sensor": sensor, "n_masks": len(summary), "graphs": summary},
                  f, indent=2)
    med = np.median([g["n_nodes"] for g in summary]) if summary else 0
    print(f"{sensor}: {len(summary)} graphs | median nodes={med:.0f} | "
          f"resilience p50={np.median([g['resilience_index'] for g in summary] or [0]):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Masks -> road graphs -> criticality.")
    parser.add_argument("--sensors", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--graphml", type=int, default=3,
                        help="how many per-sensor graphs to export as GraphML")
    parser.add_argument("--out", default="outputs/graphs")
    args = parser.parse_args()
    sensors = args.sensors or list(config.SENSORS)
    for s in sensors:
        process_sensor(s, args.limit, args.graphml, args.out)
    print(f"done -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
