"""TERRAIN — tests/graph-tests/hyderabad_data.py
Sample a 512x512 window of the Hyderabad road mask -> data/hyderabad_mask_window.npy

Run: python tests/graph-tests/hyderabad_data.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness

OUT = os.path.join(harness.DATA_DIR, "hyderabad_mask_window.npy")

if __name__ == "__main__":
    mask, _, _, meta = harness.sample_window("hyderabad", win=(512, 512), out_path=OUT)
    print(meta)
    print("saved:", OUT)