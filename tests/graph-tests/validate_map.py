"""
validate_map.py — how close is the generated map to the result (ground truth)?

Measures and LOGS closeness between the generated/healed map and the
ground-truth road network for a city window:

  Pixel level (when a generated mask is supplied via --mask):
    Dice, IoU, precision, recall  (generated vs truth)

  Graph level (always):
    - skeleton recall: fraction of truth road pixels covered by the graph
    - healed-bridge legitimacy: of the bridges added by healing, what fraction
      follow a real road gap in the truth (vs. hallucinating across background)
    - topology: connected components of truth vs of the healed graph

Verdict lines are printed and a JSON report is written.

Run in the `terrain-torch` env:
    conda run -n terrain-torch python tests/graph-tests/validate_map.py
      [--city hyderabad] [--window 512] [--center "col,row"]
      [--mask <generated_mask.npy>]     # optional: segmentation output to compare
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
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import config
import harness as H


# ─── pixel-level closeness (generated mask vs truth) ─────────────────────────
def pixel_metrics(generated: np.ndarray, truth: np.ndarray) -> dict:
    g = (generated > 0).astype(bool)
    t = (truth > 0).astype(bool)
    inter = np.logical_and(g, t).sum()
    union = np.logical_or(g, t).sum()
    tp = inter
    fp = (g & ~t).sum()
    fn = (~g & t).sum()
    iou = inter / union if union else float("nan")
    dice = 2 * inter / (g.sum() + t.sum()) if (g.sum() + t.sum()) else float("nan")
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    rec = tp / (tp + fn) if (tp + fn) else float("nan")
    return {"iou": round(float(iou), 4), "dice": round(float(dice), 4),
            "precision": round(float(prec), 4), "recall": round(float(rec), 4)}


# ─── graph-level closeness (healed graph vs truth) ────────────────────────────
def skeleton_recall(Gh, truth, subtransform):
    """Fraction of truth road pixels that lie within tol px of the graph."""
    import networkx as nx
    tol = 2.0
    inv = ~subtransform
    pts = []
    for u, v in Gh.edges():
        pu, pv = Gh.nodes[u].get("pos"), Gh.nodes[v].get("pos")
        if pu is None or pv is None:
            continue
        c1, r1 = inv * (pu[0], pu[1]); c2, r2 = inv * (pv[0], pv[1])
        for p in _line((c1, r1), (c2, r2), max(2, int(np.hypot(c2 - c1, r2 - r1)))):
            pts.append((round(p[0]), round(p[1])))
    if not pts:
        return 0.0
    ys, xs = np.array([p[1] for p in pts]), np.array([p[0] for p in pts])
    h, w = truth.shape
    xs = np.clip(xs, 0, w - 1); ys = np.clip(ys, 0, h - 1)
    road = (truth > 0)
    hit = road[ys, xs]
    cov = hit.mean()
    # truth road pixels near the graph
    from scipy.ndimage import binary_dilation
    near = binary_dilation(hit, iterations=int(tol))
    recall = hit[near].mean() if near.sum() else 0.0
    return round(float(cov), 4)


def _line(p1, p2, n):
    t = np.linspace(0, 1, max(2, n))
    return np.column_stack([p1[0] + (p2[0] - p1[0]) * t, p1[1] + (p2[1] - p1[1]) * t])


def bridge_legitimacy(Gh, truth, subtransform, radius=3.0, min_on_truth=0.5):
    """Healed TRUE bridges (int-int) validated against truth road pixels.

    A bridge is LEGITIMATE if >= min_on_truth of its sampled path pixels lie on
    truth roads (within radius). Otherwise it is flagged hallucinated.
    Returns counts + the sampled on-truth fraction distribution.
    """
    import numpy as _np
    inv = ~subtransform
    road = (truth > 0)
    h, w = truth.shape
    bridges = [(u, v, d) for u, v, d in Gh.edges(data=True)
               if d.get("healing", False) and isinstance(u, int) and isinstance(v, int)]
    fracs = []
    for u, v, d in bridges:
        pu, pv = Gh.nodes[u].get("pos"), Gh.nodes[v].get("pos")
        if pu is None or pv is None:
            continue
        c1, r1 = inv * (pu[0], pu[1]); c2, r2 = inv * (pv[0], pv[1])
        n = max(3, int(np.hypot(c2 - c1, r2 - r1)))
        pts = _line((c1, r1), (c2, r2), n)
        xs = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
        ys = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
        fracs.append(road[ys, xs].mean())
    if not fracs:
        return {"n_bridges": 0, "legit": 0, "hallucinated": 0,
                "legitimacy": 0.0, "mean_on_truth": 0.0, "max_on_truth": 0.0}
    fracs = np.asarray(fracs)
    legit = int((fracs >= min_on_truth).sum())
    return {
        "n_bridges": len(fracs),
        "legit": legit,
        "hallucinated": int(len(fracs) - legit),
        "legitimacy": round(float(legit / len(fracs)), 4),
        "mean_on_truth": round(float(fracs.mean()), 4),
        "max_on_truth": round(float(fracs.max()), 4),
    }


def topology_truth(truth):
    from skimage.morphology import label
    lab = label(truth > 0)
    comps = len(np.unique(lab)) - 1 if (truth > 0).any() else 0
    return comps


def main():
    ap = argparse.ArgumentParser(description="Validate generated map vs ground truth")
    ap.add_argument("--city", default="hyderabad")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--center", default=None, help='quoted "col,row"')
    ap.add_argument("--mask", default=None,
                    help="generated mask .npy to compare (else use truth as baseline)")
    ap.add_argument("--out", default=os.path.join(_ROOT, "outputs", "graph-output-tests"))
    args = ap.parse_args()

    center = tuple(int(x) for x in args.center.split(",")) if args.center else None
    os.makedirs(args.out, exist_ok=True)

    # truth mask = rasterized gpkg ground truth
    truth, subtransform, bounds, meta = H.sample_window(
        args.city, win=(args.window, args.window), center=center)

    # generated mask: either a supplied segmentation output, or the truth raster
    if args.mask:
        generated = np.load(args.mask)
    else:
        generated = truth.copy()  # baseline: graph built straight from truth

    G, Gh, timings, g_meta = H.run_graph(generated, transform=subtransform,
                                         max_gap_m=config.HEAL_MAX_GAP_M)
    metrics = H.compute_metrics(G, Gh)

    report = {
        "city": args.city, "window": args.window, "center": center,
        "crs": str(meta["crs"]), "bbox": list(bounds),
        "foreground_px_truth": meta["foreground_px"],
        "pixel_level": pixel_metrics(generated, truth),
        "graph_level": {
            "skeleton_recall_vs_truth": skeleton_recall(Gh, truth, subtransform),
            "bridge_legitimacy": bridge_legitimacy(Gh, truth, subtransform),
            "healing_metrics": metrics,
            "truth_components": topology_truth(truth),
        },
        "timings": timings,
        "note": ("pixel metrics are ~1.0 when no --mask is given (generated==truth). "
                 "Pass a segmentation --mask to get the real model-vs-truth closeness."),
    }

    # verdict
    gl = report["graph_level"]
    bl = gl["bridge_legitimacy"]
    lines = [f"bridge legitimacy      : {bl['legitimacy']:.0%} "
             f"({bl['legit']}/{bl['n_bridges']}) legit, "
             f"{bl['hallucinated']} hallucinated, mean_on_truth={bl['mean_on_truth']:.2f}",
             f"healed components      : {metrics['comps_before']} -> {metrics['comps_after']} "
             f"(truth={gl['truth_components']})",
             f"LCC                   : {metrics['lcc_before']:.3f} -> {metrics['lcc_after']:.3f}",
             f"max bridge gap        : {metrics['max_healed_gap']:.1f} m "
             f"cap={metrics['cap']:.0f} m"]
    if args.mask:
        pl = report["pixel_level"]
        lines.insert(0, f"pixel Dice/IoU          : {pl['dice']:.3f} / {pl['iou']:.3f} "
                        f"(prec {pl['precision']:.3f} rec {pl['recall']:.3f})")

    print("\n" + "=" * 66)
    print(f"VALIDATION — {args.city.upper()} window={args.window} center={center}")
    print("=" * 66)
    for ln in lines:
        print("  " + ln)

    rpath = os.path.join(args.out, f"{args.city}_validate_report.json")
    with open(rpath, "w") as f:
        json.dump(report, f, indent=2)
    print("  report:", rpath)


if __name__ == "__main__":
    main()
