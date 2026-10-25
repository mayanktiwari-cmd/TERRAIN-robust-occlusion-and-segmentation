"""
Inference helpers: load a trained SwinUNet and turn tiles into binary road masks.

- load_model: build SwinUNet and load a checkpoint (tolerates DataParallel 'module.').
- run_segmentation: tile -> (mask_uint8_0_255, prob_float32), with optional
  speckle removal + morphological close.
- save_mask: write the mask as a georeferenced single-band GeoTIFF (same transform
  as the source tile).

Normalization reuses the repo's cached per-sensor percentile stats; the sensor is
inferred from the tile path (data/processed/<sensor>/...) with a per-tile 2-98
percentile fallback when it can't be determined.
"""

import os
import re

import numpy as np
import torch

import config
from segmentation.model import SwinUNet


def load_model(checkpoint_path=None, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinUNet(condition_on_gsd=False)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        sd = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        model.load_state_dict(sd)
        print(f"[inference] loaded ckpt {checkpoint_path}")
    model.to(device).eval()
    return model


def _infer_sensor(path):
    m = re.search(r"/data/processed/(\w+)/", path.replace("\\", "/"))
    return m.group(1) if m else None


def _percentile_norm(arr, in_channels):
    arr = arr.astype(np.float32)
    if arr.shape[0] < in_channels:
        pad = np.full((in_channels - arr.shape[0], arr.shape[1], arr.shape[2]),
                      config.PAD_VALUE, dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    arr = arr[:in_channels]
    lo = np.percentile(arr, 2, axis=(1, 2), keepdims=True)
    hi = np.percentile(arr, 98, axis=(1, 2), keepdims=True)
    return np.clip((arr - lo) / np.maximum(hi - lo, 1e-6), 0, 1)


def _normalize_tile(arr, sensor):
    from segmentation.dataset import _load_norm_stats, _normalize
    if sensor is None:
        return _percentile_norm(arr, config.IN_CHANNELS)
    try:
        stats = _load_norm_stats()
        if sensor in stats:
            return _normalize(arr.astype(np.float32), sensor, stats, config.IN_CHANNELS)
    except Exception:
        pass
    return _percentile_norm(arr, config.IN_CHANNELS)


def _cleanup(mask, min_component_px=None, close_kernel=None):
    from scipy import ndimage
    min_component_px = config.INFER_MIN_COMPONENT_PX if min_component_px is None \
        else min_component_px
    close_kernel = config.INFER_CLOSE_KERNEL if close_kernel is None else close_kernel
    if min_component_px > 0:
        lab, n = ndimage.label(mask)
        if n > 0:
            sizes = ndimage.sum(mask, lab, range(1, n + 1))
            keep = np.isin(lab, np.where(sizes >= min_component_px)[0] + 1)
            mask = mask * keep
    if close_kernel:
        mask = ndimage.binary_closing(mask, structure=np.ones((3, 3))).astype(mask.dtype)
    return mask


def _binarize(prob, base_threshold=None):
    """Prob -> binary mask, with the adaptive anti-flood pass.

    Base threshold = base_threshold or config.INFER_THRESHOLD_START. If the tile's
    predicted road fraction exceeds config.INFER_MAX_ROAD_FRAC, the threshold is
    raised in INFER_THRESHOLD_STEP increments (up to INFER_THRESHOLD_MAX) until the
    fraction drops under the cap. This prunes the "predict road everywhere" flood
    (which pushed LISS-IV to ~77% road) without retraining.
    """
    base = base_threshold if base_threshold is not None \
        else config.INFER_THRESHOLD_START
    if not getattr(config, "INFER_ADAPTIVE_THRESHOLD", False):
        return (prob > base).astype(np.uint8)
    t = base
    while True:
        mask = prob > t
        if mask.mean() <= config.INFER_MAX_ROAD_FRAC or t >= config.INFER_THRESHOLD_MAX:
            return mask.astype(np.uint8)
        t = min(config.INFER_THRESHOLD_MAX, t + config.INFER_THRESHOLD_STEP)


def run_segmentation(tile, checkpoint_path=None, device=None,
                     threshold=None, cleanup=True, model=None):
    """Return (mask_uint8 0/255, prob_float32) for a tile path or (C,H,W) array.

    Pass a prebuilt `model` (from load_model) to avoid rebuilding the network per
    tile — run_inference() in the standalone inference.py does this for full-coverage
    passes over thousands of tiles.
    """
    import rasterio
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    threshold = config.INFER_THRESHOLD if threshold is None else threshold
    model = model if model is not None else load_model(checkpoint_path, device)
    if isinstance(tile, str):
        sensor = _infer_sensor(tile)
        with rasterio.open(tile) as ds:
            arr = ds.read()                  # (C,H,W) uint16
    else:
        arr, sensor = np.asarray(tile), None
    norm = _normalize_tile(arr, sensor)      # (C,H,W) float32 [0,1]
    inp = torch.from_numpy(norm).unsqueeze(0).to(device)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(inp)
        prob = torch.sigmoid(logits.float())[0, 0].cpu().numpy()
    mask = _binarize(prob, threshold)
    if cleanup:
        mask = _cleanup(mask)
    return mask * 255, prob


def infer_mask(tile_path: str, checkpoint_path=None, device=None) -> np.ndarray:
    """Binary (0/255 uint8) road mask for a satellite tile — graph pipeline contract.

    `main.py._build_graph` calls `infer_mask(tile_path)`; this wraps
    run_segmentation so the mask->skeleton->graph chain uses real predictions
    instead of the synthetic fallback.
    """
    mask_uint8, _ = run_segmentation(tile_path, checkpoint_path, device)
    return mask_uint8


def save_mask(mask_uint8, source_tile_path, out_path):
    """Write mask_uint8 (0/255) as a georeferenced single-band GeoTIFF."""
    import rasterio
    with rasterio.open(source_tile_path) as ds:
        profile = ds.profile
    profile.update(dtype="uint8", count=1, compress="deflate", nodata=0)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask_uint8, 1)
    return out_path
