"""TERRAIN — tests/graph-tests/test_complexity.py
Empirically measure per-stage wall-clock scaling across window sizes on a map,
saving outputs/graph-output-tests/complexity_scaling_<city>.png. Mirrors the
asymptotic analysis.

Run: python tests/graph-tests/test_complexity.py --city hyderabad --windows 256,512,1024
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="hyderabad", choices=sorted(harness.CITIES))
    ap.add_argument("--windows", default="256,512,1024")
    ap.add_argument("--no-geo", action="store_true")
    ap.add_argument("--out", default=None,
                    help="output dir for the scaling plot (default: outputs/graph-output-tests)")
    args = ap.parse_args()

    wins = [int(x) for x in args.windows.split(",")]
    harness.measure_scaling(
        args.city, windows=wins, geo=not args.no_geo, out_dir=args.out
    )


if __name__ == "__main__":
    main()