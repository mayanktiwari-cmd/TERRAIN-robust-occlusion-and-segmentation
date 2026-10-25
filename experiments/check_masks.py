"""
Phase-0 mask audit for TERRAIN (run BEFORE more training).

For each sensor, sample a spread of tiles and report:
    - road-pixel fraction (median / p90 / min)
    - estimated physical road half-width (m) via distance transform x GSD
      (a sanity proxy for "over-thick rasterization", e.g. LISS-IV ~42% road)
    - overlay-grid PNGs (image | mask) saved to outputs/mask_audit/

Usage:
    python experiments/check_masks.py                # all sensors
    python experiments/check_masks.py --sensors liss4 --n 24 --out outputs/mask_audit
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config  # noqa: E402

N_SAMPLES = 12


def load_pair(img_path, mask_path):
    import rasterio
    with rasterio.open(img_path) as ds:
        img = ds.read().astype(np.float32)      # (C,H,W)
    with rasterio.open(mask_path) as ds:
        mask = ds.read(1)                        # (H,W) uint16 0/1
    return img, mask > 0


def road_width_meters(mask, gsd):
    """Approx physical mean road half-width (m) via Euclidean distance transform."""
    from scipy import ndimage
    if not mask.any():
        return 0.0
    dist = ndimage.distance_transform_edt(mask)
    half_w_px = dist[mask].mean()
    return float(half_w_px * 2.0 * gsd)          # full width ~ 2x mean half-width


def road_fraction(mask):
    return float(mask.mean())


def audit_sensor(sensor, n, out_dir):
    import rasterio
    mask_dir = os.path.join(config.DATA_DIR, "processed", sensor, "mask")
    img_dir = os.path.join(config.DATA_DIR, "processed", sensor, "image")
    masks = sorted(glob.glob(os.path.join(mask_dir, "tile_*.tif")))
    if not masks:
        print(f"[skip] {sensor}: no tiles")
        return None
    gsd = config.GSD_METERS.get(sensor, 1.0)
    step = max(1, len(masks) // (n * 3))
    chosen = masks[::step][:n]

    fracs, widths, panels = [], [], []
    for mask_path in chosen:
        name = os.path.basename(mask_path)
        img_path = os.path.join(img_dir, name)
        if not os.path.isfile(img_path):
            continue
        img, mask = load_pair(img_path, mask_path)
        fracs.append(road_fraction(mask))
        widths.append(road_width_meters(mask, gsd))
        panels.append((sensor, name, img, mask))

    fracs = np.array(fracs)
    print(f"{sensor}: n={len(fracs)} tiles | road% mean={fracs.mean()*100:.1f} "
          f"p50={np.median(fracs)*100:.1f} p90={np.percentile(fracs,90)*100:.1f} "
          f"min={fracs.min()*100:.3f} | est. road width m (median)={np.median(widths):.1f} "
          f"(GSD={gsd}m)")
    save_grid(panels, out_dir, sensor)
    return {"sensor": sensor, "road_fracs": fracs, "widths": widths}


def save_grid(panels, out_dir, sensor):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"  [warn] matplotlib missing; skipping overlay grid for {sensor}")
        return
    n = len(panels)
    if n == 0:
        return
    cols, rows = 4, int(np.ceil(n / 4))
    fig, axes = plt.subplots(rows, cols * 2, figsize=(4 * 2 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i, (sensor_name, name, img, mask) in enumerate(panels):
        rgb = img[:3]
        lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
        rgb = np.clip((rgb - lo) / max(1e-6, hi - lo), 0, 1)
        rgb = np.transpose(rgb, (1, 2, 0))
        axes[2 * i].imshow(rgb)
        axes[2 * i].set_title(f"{sensor_name} {name[:14]}", fontsize=7)
        axes[2 * i].axis("off")
        axes[2 * i + 1].imshow(mask, cmap="gray_r")
        axes[2 * i + 1].set_title(f"road {mask.mean()*100:.1f}%", fontsize=7)
        axes[2 * i + 1].axis("off")
    for a in axes[2 * n:]:
        a.axis("off")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"mask_audit_{sensor}.png")
    fig.suptitle(f"{sensor} — image | mask (road% + est width m)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    print(f"  grid -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Phase-0 mask audit.")
    parser.add_argument("--sensors", nargs="*", default=None,
                        help="sensor subfolders (default: all in config.SENSORS)")
    parser.add_argument("--n", type=int, default=N_SAMPLES, help="tiles to sample per sensor")
    parser.add_argument("--out", default=os.path.join(config.OUTPUT_DIR, "mask_audit"))
    args = parser.parse_args()

    sensors = args.sensors or config.SENSORS
    print("percentiles (road%): low p50 = sparse roads; suspiciously high mean "
          "(e.g. >35%) + wide est. width => over-thick rasterization")
    for s in sensors:
        audit_sensor(s, args.n, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
