"""
connector_sentinel2.py — chain the sentinel2 road-mask mosaic -> graph -> GCN.

Input: sentinel2_mosaic.tif (EPSG:32643, ~10 m/px). Connects the binary mask
through the graph module (skeletonize -> build -> heal) then the GCN module
(graph_to_tensor -> RoadGCN -> gatekeepers -> resilience). Writes
connector_sentinel2.png / _report.json / _graph.pkl.

Run (terrain-torch env):
    conda run -n terrain-torch python tests/graph-tests/connector_sentinel2.py
        [--center "2560,2048"] [--window 512] [--gcn-epochs 120]
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
import chain_core


TIF = os.path.join(_ROOT, "sentinel2_mosaic.tif")
TAG = "sentinel2"
DEFAULT_CENTER = (2560, 2048)  # road-dense window (col,row)


def main():
    ap = argparse.ArgumentParser(description="sentinel2 mask -> graph -> GCN")
    ap.add_argument("--center", default=None, help='quoted "col,row" (road-dense default)')
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--checkpoint", default=config.MODEL_SAVE_PATH)
    ap.add_argument("--gcn-epochs", type=int, default=120)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--out", default=os.path.join(_ROOT, "outputs", "graph-output-tests"))
    args = ap.parse_args()

    center = tuple(int(x) for x in args.center.split(",")) if args.center else DEFAULT_CENTER
    chain_core.run(TIF, TAG, center, args.checkpoint, args.out,
                   window=args.window, gcn_epochs=args.gcn_epochs,
                   top_k=args.top_k, no_plot=args.no_plot)


if __name__ == "__main__":
    main()