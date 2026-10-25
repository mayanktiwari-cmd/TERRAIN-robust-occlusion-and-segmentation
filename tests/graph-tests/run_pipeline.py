"""
TERRAIN — tests/graph-tests/run_pipeline.py

Friendly CLI to run the full graph pipeline on ANY registered map and ANY
window, independently. No source code is modified; it only drives the public
graph/ API via harness.py.

Examples:
    python tests/graph-tests/run_pipeline.py --city hyderabad --window 512
    python tests/graph-tests/run_pipeline.py --city delhi --center "2048,2048"
    python tests/graph-tests/run_pipeline.py --city bengaluru --center "2304,3072"
    python tests/graph-tests/run_pipeline.py --list
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import harness


def main():
    ap = argparse.ArgumentParser(description="TERRAIN graph pipeline runner (any map, any window).")
    ap.add_argument("--city", default="hyderabad", choices=sorted(harness.CITIES),
                    help="registered map (see --list)")
    ap.add_argument("--window", type=int, default=512, help="window width/height in px")
    ap.add_argument("--center", default=None, help="offset 'r,c' (default: image center)")
    ap.add_argument("--no-geo", action="store_true",
                    help="disable georeferencing (fall back to pixel units)")
    ap.add_argument("--skip-centrality", action="store_true", help="skip betweenness")
    ap.add_argument("--no-plot", action="store_true", help="skip plotting")
    ap.add_argument("--save-mask", action="store_true",
                    help="write the sampled mask to tests/graph-tests/data/<city>_mask_window.npy")
    ap.add_argument("--out", default=None,
                    help="output dir for plots (default: outputs/graph-output-tests)")
    ap.add_argument("--list", action="store_true", help="list registered maps and exit")
    args = ap.parse_args()

    if args.list:
        for name, cfg in harness.CITIES.items():
            print(f"{name:<12} crs={cfg['crs']:<12} layer={cfg['layer']}")
        return

    win = (args.window, args.window)
    center = None
    if args.center:
        c, r = args.center.split(",")
        center = (int(c), int(r))

    # 1) Sample
    t0 = time.perf_counter()
    mask_path = os.path.join(harness.DATA_DIR,
                             f"{args.city}_mask_window.npy") if args.save_mask else None
    mask, subtransform, _, meta = harness.sample_window(
        args.city, win=win, center=center, out_path=mask_path
    )
    sample_s = time.perf_counter() - t0
    print(f"sampled in {sample_s:.2f}s  foreground={meta['foreground_px']}px")

    if (mask > 0).sum() == 0:
        print("WARNING: empty road mask in this window; nothing to heal.")
        return

    # 2) Pipeline
    G, Gh, timings, g_meta = harness.run_graph(
        mask, transform=subtransform if not args.no_geo else None
    )

    # 3) Metrics
    metrics = harness.compute_metrics(G, Gh)
    cent = None if args.skip_centrality else harness.compute_centrality(Gh)

    # 4) Report
    harness.print_report(args.city, win, meta, g_meta, metrics, timings, cent)

    # 5) Plots
    if not args.no_plot:
        outs = harness.render_plots(G, Gh, args.city, out_dir=args.out)
        print("\nplots:", outs)


if __name__ == "__main__":
    main()
