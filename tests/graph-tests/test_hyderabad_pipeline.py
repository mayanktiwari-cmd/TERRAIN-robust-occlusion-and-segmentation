"""TERRAIN — tests/graph-tests/test_hyderabad_pipeline.py
Full pipeline on Hyderabad 512x512 (geographic), prints report + saves plots.

Run: python tests/graph-tests/test_hyderabad_pipeline.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness

CITY = "hyderabad"
WIN = (512, 512)

MASK = os.path.join(harness.DATA_DIR, "hyderabad_mask_window.npy")


def main():
    if not os.path.exists(MASK):
        mask, subtransform, _, meta = harness.sample_window(
            CITY, win=WIN, out_path=MASK
        )
    else:
        import numpy as np
        mask = np.load(MASK)
        subtransform = None
        meta = {"crs": harness.CITIES[CITY]["crs"], "window": WIN,
                "offset": (0, 0), "bbox": (0, 0, 0, 0), "n_features": 0,
                "foreground_px": int((mask > 0).sum()),
                "foreground_frac": float((mask > 0).mean())}

    G, Gh, timings, g_meta = harness.run_graph(mask, transform=subtransform)
    metrics = harness.compute_metrics(G, Gh)
    cent = harness.compute_centrality(Gh)
    harness.print_report(CITY, WIN, meta, g_meta, metrics, timings, cent)

    outs = harness.render_plots(G, Gh, CITY)
    print("\nplots:", outs)


if __name__ == "__main__":
    main()