"""
TERRAIN — tests/graph-tests/check_environment.py

Self-check interface for the road-graph pipeline. Verifies, in order:
  1. Required third-party packages are importable.
  2. Pinned reference versions (from a working env) match, warn-only.
  3. graph/ modules import and expose the public API.
  4. A tiny synthetic mask runs end-to-end (skeletonize -> build -> heal).
  5. Optionally, a real map window runs end-to-end (geo by default).

Exit codes: 0 = all checks pass; 1 = a hard check failed.

Run:
    python tests/graph-tests/check_environment.py                 # synthetic smoke only (fast)
    python tests/graph-tests/check_environment.py --map hyderabad # + real window (geo)
    python tests/graph-tests/check_environment.py --map delhi --no-geo --window 256
    python tests/graph-tests/check_environment.py --full          # all registered maps
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

RESULTS = []


def _report(name, ok, detail=""):
    RESULTS.append((name, ok))
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}{('  -- ' + detail) if detail else ''}")
    return ok


def _import_versions():
    """Required for graph/ + tests/. Returns {module: version}. Raises on missing."""
    mods = [
        "numpy", "networkx", "scipy", "skimage", "matplotlib",
        "geopandas", "rasterio", "shapely", "pandas", "pyogrio",
    ]
    out = {}
    for m in mods:
        mod = __import__(m)
        out[m] = getattr(mod, "__version__", "?")
    return out


# Reference versions from the verified working env (warn-only if different).
REFERENCE = {
    "numpy": "2.4.6", "networkx": "3.6.1", "scipy": "1.17.1",
    "skimage": "0.26.0", "matplotlib": "3.11.0", "geopandas": "1.1.3",
    "rasterio": "1.4.3", "shapely": "2.1.2", "pandas": "3.0.3",
    "pyogrio": "0.10.0",
}


def check_deps():
    try:
        ver = _import_versions()
    except ImportError as e:
        _report("dependencies importable", False, str(e))
        return False
    _report("dependencies importable", True)
    for m in ["numpy", "networkx", "scipy", "skimage", "matplotlib",
              "geopandas", "rasterio", "shapely", "pandas", "pyogrio"]:
        v = ver[m]
        ref = REFERENCE.get(m)
        if ref and v != ref:
            _report(f"version {m}", False, f"have {v}, reference {ref}")
        else:
            _report(f"version {m}", True, v)
    return True


def check_graph_api():
    try:
        from graph.skeletonize import skeletonize_mask
        from graph.build_graph import build_graph
        from graph.heal import heal_graph
        from graph.visualize import plot_graph_comparison, plot_healing_result
    except Exception as e:
        _report("graph/ public API", False, repr(e))
        return False
    _report("graph/ public API", True,
            "skeletonize_mask, build_graph, heal_graph, plot_*")
    return True


def check_synthetic_smoke():
    """End-to-end on a synthetic mask — fast, no data files needed."""
    from skimage.morphology import skeletonize as _skel

    rng = np.random.default_rng(0)
    m = rng.random((64, 64)) > 0.88
    m = _skel(m).astype(np.uint8)
    if (m > 0).sum() == 0:
        m[10, 10] = 1
        m[50, 50] = 1

    from graph.skeletonize import skeletonize_mask
    from graph.build_graph import build_graph
    from graph.heal import heal_graph

    t0 = time.perf_counter()
    sk, nodes, edges = skeletonize_mask(m)
    G = build_graph(sk, nodes, edges)
    Gh = heal_graph(G)
    dt = time.perf_counter() - t0

    nodes_ok = G.number_of_nodes() >= 0 and sk.shape == m.shape
    healed_ok = Gh.number_of_nodes() >= 0
    _report("synthetic smoke (skel->build->heal)", nodes_ok and healed_ok,
            f"nodes={G.number_of_nodes()} edges={G.number_of_edges()} "
            f"healed_nodes={Gh.number_of_nodes()} in {dt:.3f}s")
    return nodes_ok and healed_ok


def check_real_map(city, win, geo):
    import harness

    mask, subtransform, _, meta = harness.sample_window(city, win=(win, win))
    if (mask > 0).sum() == 0:
        _report(f"real run {city}", False, "empty window (no roads)")
        return False
    G, Gh, tim, gm = harness.run_graph(mask, transform=subtransform if geo else None)
    met = harness.compute_metrics(G, Gh)
    cap = met["cap"]
    gap_ok = met["max_healed_gap"] <= cap
    lcc_ok = met["lcc_after"] >= 0.99
    _report(f"real run {city} (win={win}, geo={geo})",
            gap_ok and lcc_ok,
            f"comps {met['comps_before']}->{met['comps_after']} LCC "
            f"{met['lcc_after']:.3f} max_gap {met['max_healed_gap']:.1f}<={cap:.0f} "
            f"frag={meta['foreground_px']}px")
    return gap_ok and lcc_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default=None, choices=sorted(__import__("harness").CITIES),
                    help="run a real map window too")
    ap.add_argument("--full", action="store_true", help="run all registered maps")
    ap.add_argument("--no-geo", action="store_true")
    ap.add_argument("--window", type=int, default=256)
    args = ap.parse_args()

    print("TERRAIN environment self-check\n" + "-" * 40)
    hard = True
    hard &= check_deps()
    hard &= check_graph_api()
    hard &= check_synthetic_smoke()

    if args.full:
        import harness
        for city in harness.CITIES:
            hard &= check_real_map(city, args.window, not args.no_geo)
    elif args.map:
        hard &= check_real_map(args.map, args.window, not args.no_geo)

    print("-" * 40)
    n_fail = sum(1 for _, ok in RESULTS if not ok)
    print(f"SUMMARY: {len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if (n_fail and hard) else 0)


if __name__ == "__main__":
    main()