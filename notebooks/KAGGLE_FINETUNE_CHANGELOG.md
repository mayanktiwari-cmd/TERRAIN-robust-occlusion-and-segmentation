# KAGGLE_FINETUNE_CHANGELOG

Corrected `kaggle_finetune.ipynb` (written to repo root `kaggle_finetune.ipynb` and
canonical `notebooks/kaggle_finetune.ipynb` — identical content). Scope: segmentation
training notebook only. Nothing touched in graph-healing, criticality, GCN, or dashboard.

## Cell-by-cell changes (vs the stale draft)

**Intro (markdown)**
- Rewritten: replaced the single-`terrain-tiles` input assumption with the three-input
  reality (`terrain-code`, `terrain-tiles-cartosat3`, `terrain-tiles-liss4-s2`).
- Explicitly notes **Internet = ON** is required (Cell 6 downloads timm ImageNet Swin
  weights from Hugging Face; `SwinUNet` falls back to random init if the download fails).
- GPU resolved to **single T4** for the notebook body (see Cell 7 note below).

**Cell 1 — environment check:** unchanged.

**Cell 2 — deps:** added `segmentation-models-pytorch` to the pip install (Bug 3).

**Cell 3 — code pull:** unchanged (`terrain-code` → `/kaggle/working`).

**Cell 4 — config + data paths (Bug 1 + dataloader task):**
- **Removed `config.CACHE_SIZE = 10000` entirely** (Bug 1).
- Rewritten to unify the two tile inputs: `TILE_INPUTS` maps every sensor to its input;
  `_sensor_root()` probes the archive layouts (`data/processed/`, `processed/`, or bare)
  so it works regardless of how the zip was staged; a symlink farm under
  `/kaggle/working/data/processed/<sensor>` makes the repo's single-root
  `MultiSensorDataset` work unchanged.
- `_find_norm_stats()` probes both tile inputs for the one combined `_norm_stats.json`
  and copies it into the unified dir (keeps `config.NORM_STATS_PATH` a single path —
  compatible with `MultiSensorDataset`).
- `config.EPOCHS = 5` kept as the first-run smoke value (unchanged, per instruction).
- Added explicit `config.BATCH_SIZE = 8`.

**Cell 5 — dataset smoke (Bug 2):**
- **Removed `len(tr._cache)`.** The real attribute is `MultiSensorDataset._cache`, an
  `OrderedDict` that is only populated when `cache_size > 0`; with the Cell-4 override
  gone (config default `CACHE_SIZE = 0`), it stays empty, so the old line was misleading.
  Replaced with genuine internal state: `sensor_ids` map + per-sensor `per_sensor`
  index lengths, plus asserts that sentinel2 is absent from train and present in val.

**Cell 6 — model:** unchanged code. Markdown header now flags Internet = ON.

**Cell 7 — train (GPU inconsistency):**
- Resolved to **single GPU**: `device=torch.device("cuda")`, no DataParallel in the
  executed path. Rationale: `segmentation/train.py` runs single-device loops, and the
  team's own `segmentation/train_kaggle.py` calls `train_model(device="cuda")` with no
  wrapper. `batch_size` now reads `config.BATCH_SIZE`. DataParallel kept only as an
  optional comment (with the `module.`-prefix caveat and Cell-9 strip hint).

**Cell 8 — history/plot:** unchanged.

**Cell 9 — strip `module.` prefix:** unchanged (kept as the DataParallel-cleanup helper).

## Confirmed facts / not changed
- `IMG_SIZE = 256` everywhere (config, dataset asserts 4×256×256); no 512 anywhere.
- `TERRAIN_Loss` weights unchanged (Dice α=0.4, IoU β=0.3, Boundary γ=0.3, Connectivity δ=0.2).
- Phase C `CONDITION_ON_GSD = False` untouched.
- Split verified against the two-input layout: train = cartosat3 (47,153) + liss4 (2,695);
  val = sentinel2 (1,503) only.

## Flagged: repo code change (NOT applied — out of notebook scope)
- `MultiSensorDataset.__init__(root=...)` accepts only a **single root string** and
  builds `os.path.join(root, sensor, "image")`. It has **no native support for multiple
  roots / a per-sensor root dict**. The notebook works around this with the symlink farm,
  so no code change is required for this run.
- Proposed minimal signature change if multi-root support is wanted later:
  `root: Optional[Union[str, Dict[str, str]]]` — a dict keyed by sensor name maps each
  sensor to its own tile root; `img_dir`/`mask_dir` resolve per sensor; `None`/`str`
  keeps current behavior. This is backward compatible and would let the notebook point
  at the two inputs directly without symlinks.

## Transparency note
- Bug 3 asked to add `segmentation-models-pytorch` because it "is imported somewhere in
  the training/model code path." A repo-wide grep found **no** `import segmentation_models_pytorch`
  / `smp` anywhere (the only third-party image lib referenced is `torchvision` in
  `experiments/check_dataset.py` and `requirements.txt`). The package was added to the
  deps cell as instructed (harmless), but no repo code currently imports it.
