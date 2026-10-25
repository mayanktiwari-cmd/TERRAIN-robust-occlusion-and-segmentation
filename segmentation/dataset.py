"""
Multi-sensor dataset for cross-GSD road segmentation (one model, 3 sensors).

Builds an index over data/processed/<sensor>/{image,mask} for every sensor in
config.SENSORS, applies per-sensor percentile normalization (shifted to [0,1]),
aligns all sensors to config.IN_CHANNELS bands (LISS-IV is 3-band -> zero-fill the
4th/Blue channel), applies multiscale + geometric augmentation, and returns
(img, mask, meta). Opt-in split-by-sensor (hold out one sensor for validation)
matches docs/MULTISCALE_PLAN.md and prevents train/val leakage.

Each sample returns:
    img   : torch.Tensor [C, H, W] float32  (normalized, zero-filled to IN_CHANNELS)
    mask  : torch.Tensor [1, H, W] float32  (binary road mask in [0,1])
    meta  : list of [sensor_id, gsd_meters] for optional resolution conditioning
"""

import json
import os
import glob
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

import config


def _load_road_fracs():
    """Load per-tile road fractions JSON (if present). Returns {} when absent."""
    try:
        with open(config.ROAD_FRACS_PATH, "r") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _load_norm_stats(path=None):
    path = path or config.NORM_STATS_PATH
    with open(path, "r") as f:
        return json.load(f)


def _clamp01(x):
    return np.clip(x, 0.0, 1.0)


class MultiSensorDataset(Dataset):
    def __init__(
        self,
        root: str = None,
        sensors: List[str] = None,
        split: Optional[str] = "train",
        holdout_sensor: Optional[str] = None,
        img_size: int = None,
        augment: bool = True,
        in_channels: int = None,
        norm_stats_path: str = None,
        cache_size: int = None,
    ):
        """split in {"train","val","test","none"}; holdout_sensor controls which
        sensor is treated as the validation city."""
        self.root = root or os.path.join(config.DATA_DIR, "processed")
        self.sensors = sensors or list(config.SENSORS)
        self.split = split
        self.holdout_sensor = holdout_sensor if holdout_sensor is not None else config.VAL_HOLDOUT_SENSOR
        self.img_size = img_size or config.IMG_SIZE
        self.augment = augment
        self.in_channels = in_channels or config.IN_CHANNELS
        self.cache_size = cache_size if cache_size is not None else config.CACHE_SIZE
        self._cache: "OrderedDict" = OrderedDict()

        self.stats = _load_norm_stats(norm_stats_path)

        # Which sensors participate. Two modes:
        #   VAL_HOLDOUT_SENSOR set  -> old behavior (one sensor fully held out).
        #   else + VAL_SPLIT_FRACTION -> every sensor splits per-tile by index
        #                               (last 20% of each sensor = val). This lets
        #                               LISS-IV train too (domain-gap fix).
        split_frac = getattr(config, "VAL_SPLIT_FRACTION", 0.0) or 0.0
        per_sensor_split = (
            split in ("train", "val") and not self.holdout_sensor and split_frac > 0
        )
        if per_sensor_split:
            active = list(self.sensors)
        elif split in ("train",):
            active = [s for s in self.sensors if s != self.holdout_sensor]
        elif split in ("val", "test"):
            active = [s for s in self.sensors if s == self.holdout_sensor]
        else:
            active = list(self.sensors)

        if not active:
            raise ValueError(f"Split '{split}' left no active sensors: {self.sensors}")

        # Optional per-tile road fractions (one-time pass: compute_road_fracs.py).
        self.road_fracs = _load_road_fracs()

        # Build flat index of (sensor, image_path, mask_path).
        self.items: List[Tuple[str, str, str]] = []
        self.sensor_ids = {}   # sensor -> int index

        for i, s in enumerate(active):
            self.sensor_ids[s] = i
            img_dir = os.path.join(self.root, s, "image")
            mask_dir = os.path.join(self.root, s, "mask")
            tiles = sorted(glob.glob(os.path.join(img_dir, "tile_*.tif")))
            pairs = []
            for img_path in tiles:
                name = os.path.basename(img_path)
                mask_path = os.path.join(mask_dir, name)
                if not os.path.isfile(mask_path):
                    continue
                # MIN_ROAD_FRAC: skip near-empty masks (in-memory, non-destructive).
                # JSON keys are "<sensor>/<basename>" (tile_* names collide across
                # sensors, so bare basenames are ambiguous).
                if config.MIN_ROAD_FRAC > 0 and self.road_fracs:
                    frac = self.road_fracs.get(f"{s}/{name}")
                    if frac is not None and frac < config.MIN_ROAD_FRAC:
                        continue
                pairs.append((img_path, mask_path))
            # Per-sensor deterministic split by tile index: train = first (1-frac),
            # val = last frac. Sorted tile order makes the split reproducible.
            if per_sensor_split:
                cut = int(len(pairs) * (1.0 - split_frac))
                if split == "val":
                    pairs = pairs[cut:]
                else:
                    pairs = pairs[:cut]
            self.items.extend((s, ip, mp) for ip, mp in pairs)

        # Per-sensor index lists for balanced sampling.
        self.per_sensor = {s: [] for s in active}
        for idx, (s, _, _) in enumerate(self.items):
            self.per_sensor[s].append(idx)

        # Sparse-road oversampling (ON via SAMPLE_WEIGHTED_BY_ROAD): tiles whose
        # road-pixel fraction < SPARSE_ROAD_FRAC get SPARSE_OVERSAMPLE x sampling
        # weight inside make_sensor_balanced_batches. Fixes the "predict road
        # everywhere" degenerate that crushed sparse-tile IoU (0.033).
        self.sample_weights = {}
        if config.SAMPLE_WEIGHTED_BY_ROAD and self.road_fracs:
            for s in active:
                idxs = self.per_sensor[s]
                if not idxs:
                    continue
                fracs = np.array(
                    [self.road_fracs.get(
                        f"{self.items[i][0]}/{os.path.basename(self.items[i][2])}", np.nan)
                     for i in idxs], dtype=np.float32)
                w = np.where(
                    np.isnan(fracs),
                    1.0,
                    np.where(fracs < config.SPARSE_ROAD_FRAC, config.SPARSE_OVERSAMPLE, 1.0),
                )
                self.sample_weights[s] = w / w.sum()

        if self.augment:
            self.transform = self._build_train_transform()
        else:
            self.transform = A.Compose([A.Resize(self.img_size, self.img_size)])

    # ────────────────────────────────────────────────────────────────
    @staticmethod
    def _build_train_transform():
        scale_lo, scale_hi = config.SCALE_AUG
        return A.Compose([
            A.RandomScale(scale_limit=(scale_lo, scale_hi), p=1.0,
                          interpolation=1, mask_interpolation=0),
            A.Resize(config.IMG_SIZE, config.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sensor, img_path, mask_path = self.items[idx]
        img_raw, mask_raw = self._read_pair(img_path, mask_path)     # uint16
        img = _normalize(img_raw.astype(np.float32), sensor, self.stats, self.in_channels)
        mask = mask_raw.astype(np.float32)[None, :, :]               # [1,H,W]

        # Apply augmentation (synchronized image + mask).
        transformed = self.transform(image=np.transpose(img, (1, 2, 0)), mask=mask[0])
        img_aug = np.transpose(transformed["image"], (2, 0, 1))
        mask_aug = transformed["mask"][None, :, :]

        return (
            torch.from_numpy(img_aug).float(),
            torch.from_numpy(mask_aug).float(),
            self._meta(sensor),
        )

    # ────────────────────────────────────────────────────────────────
    def _read_pair(self, img_path, mask_path):
        """Read a raw (img, mask) pair with an optional LRU decode cache.

        The cache stores *pre-augmentation* uint16 arrays only (~0.6 MB/sample);
        normalization and the multiscale RandomScale still run per-access, so cached
        samples remain fully augmented. Enabled via config.CACHE_SIZE (e.g. ~10000
        on Kaggle, where 4 CPU cores are the bottleneck, not the GPU).
        """
        if self.cache_size <= 0:
            return self._read_pair_uncached(img_path, mask_path)
        key = (img_path, mask_path)
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        val = self._read_pair_uncached(img_path, mask_path)
        if len(self._cache) >= self.cache_size:
            self._cache.popitem(last=False)
        self._cache[key] = val
        return val

    @staticmethod
    def _read_pair_uncached(img_path, mask_path):
        with rasterio.open(img_path) as ds:
            img = ds.read()          # uint16 (C,H,W)
        with rasterio.open(mask_path) as ds:
            mask = ds.read(1)        # uint16 (H,W)
        return img, mask

    def _meta(self, sensor):
        return [int(self.sensor_ids[sensor]), float(config.GSD_METERS.get(sensor, -1.0))]


def read_normalized(img_path, sensor, stats, in_channels):
    """Public wrapper: load a uint16 tile from disk, normalize, zero-fill."""
    with rasterio.open(img_path) as ds:
        arr = ds.read().astype(np.float32)   # (C,H,W)
    return _normalize(arr, sensor, stats, in_channels)


def _normalize(arr, sensor, stats, in_channels):
    """Per-sensor percentile normalization -> [0,1], zero-fill to in_channels.

    Missing bands (LISS-IV is 3-band vs in_channels=4) are padded with
    config.PAD_VALUE *before* normalization so the pad band is normalized
    consistently (it maps to 0 after clamping), not special-cased after.
    """
    pct_lo = np.asarray(stats[sensor]["pct_lo"], dtype=np.float32)
    pct_hi = np.asarray(stats[sensor]["pct_hi"], dtype=np.float32)

    # Pad raw array to in_channels first.
    if arr.shape[0] < in_channels:
        pad = np.full((in_channels - arr.shape[0], arr.shape[1], arr.shape[2]),
                      config.PAD_VALUE, dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=0)
    # Stats should be in_channels-wide (new JSON always is); tolerate older JSON.
    if len(pct_lo) < in_channels:
        n_pad = in_channels - len(pct_lo)
        pct_lo = np.concatenate([pct_lo, np.zeros(n_pad, dtype=np.float32)])
        pct_hi = np.concatenate([pct_hi, np.ones(n_pad, dtype=np.float32)])

    denom = pct_hi - pct_lo
    denom[denom <= 0] = 1.0
    return _clamp01((arr - pct_lo[:, None, None]) / denom[:, None, None])


def make_sensor_balanced_batches(dataset, batch_size, n_batches=None):
    """Yield batches of indices with an even per-sensor split.

    C3 (47k) vs S2 (1.5k) would otherwise drown the epoch ~95% Cartosat-3.
    Each batch holds ~batch_size/n_sensors samples per sensor; small sensors are
    oversampled (with replacement) so every batch stays full and balanced.
    When dataset.sample_weights are set (SAMPLE_WEIGHTED_BY_ROAD), draws within a
    sensor follow those capped weights instead of uniform sampling.
    """
    rng = np.random.default_rng()
    sensors = list(dataset.per_sensor.keys())
    n_sensors = len(sensors)
    per_batch = max(1, batch_size // n_sensors)
    leftovers = batch_size - per_batch * n_sensors
    lens = {s: len(dataset.per_sensor[s]) for s in sensors}
    weights = getattr(dataset, "sample_weights", None) or {}
    if n_batches is None:
        n_batches = max(1, max(int(np.ceil(v / per_batch)) for v in lens.values()))
    batches = []
    for _ in range(n_batches):
        batch = []
        for i, s in enumerate(sensors):
            take = per_batch + (1 if i < leftovers else 0)
            pool = np.asarray(dataset.per_sensor[s], dtype=np.int64)
            p = None
            w = weights.get(s)
            if w is not None and len(w) == len(pool):
                p = w
            batch.extend(rng.choice(pool, size=min(take, len(pool)), replace=True, p=p).tolist())
        rng.shuffle(batch)
        batches.append(batch)
    return batches


# Backward-compatible aliases (segmentation/__init__.py re-exports these names).
DeepGlobeDataset = MultiSensorDataset
DeepGlobeSubset = None