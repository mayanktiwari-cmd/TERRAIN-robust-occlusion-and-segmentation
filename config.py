"""
TERRAIN — Topologically Enhanced Road Resilience Analysis & Intelligence Network

All hyperparameters in one place. Every module reads from here.
No hardcoded values anywhere else.
"""

# ─── Paths ───────────────────────────────────────────────────────────
DATA_DIR = "data/"
OUTPUT_DIR = "outputs/"
MODEL_SAVE_PATH = "outputs/models/terrain_best.pth"

# ─── Segmentation ────────────────────────────────────────────────────
IMG_SIZE = 256          # processed tiles are 256x256; official doc mandates 128/256
BATCH_SIZE = 32         # AMP makes this fit on a 16GB T4; GRAD_ACCUM lifts effective batch
EPOCHS = 30             # fine-tune budget (was stuck at 5 — under-training)
LR = 2e-4               # legacy scheduler peak (finetune.py uses DECODER_LR instead)
MIN_LR = 1e-6           # floor for the warmup->cosine schedule
WARMUP_EPOCHS = 3       # legacy linear warmup (finetune.py uses WARMUP_FRACTION + OneCycleLR)
AMP = True              # FP16 autocast + GradScaler (2x speed, ~2x memory headroom)
GRAD_ACCUM = 8          # accumulate before optimizer.step() -> effective batch 32*8=256
GRAD_CLIP_NORM = 1.0    # global grad-norm clipping for stability
ENCODER = "tu-swin_tiny_patch4_window7_224"
ENCODER_WEIGHTS = "imagenet"
NUM_WORKERS = 4
PIN_MEMORY = True
IN_CHANNELS = 4         # all sensors aligned to 4 bands (LISS-IV zero-filled)
PAD_VALUE = 0.0         # constant used to fill LISS-IV's missing 4th band (pre-normalization)

# ─── Fine-tuning (finetune.py) ───────────────────────────────────────
# Domain-gap fix: LISS-IV joins training too, via a per-sensor 80/20 tile split
# (last 20% of each sensor's tiles = val) instead of a full sensor holdout.
TRAIN_SENSORS = ["cartosat3"]
VAL_SPLIT_FRACTION = 0.2      # per-sensor tile-index split; last 20% of each sensor = val
VAL_HOLDOUT_SENSOR = None     # None = per-sensor split; set a name to restore the old holdout
# Encoder (Swin backbone) reloaded from timm ImageNet weights; decoder kept from ckpt.
FINETUNE_INIT_CKPT = "outputs/models/terrain_best.pth"      # starting weights (may be missing)
FINETUNE_SAVE_PATH = "outputs/models/terrain_finetuned_best.pth"
FREEZE_ENCODER_EPOCHS = 5     # train decoder+head only for the first 5 epochs
ENCODER_LR = 2e-5             # encoder parameter-group LR (lower; backbone already strong)
DECODER_LR = 3e-4             # decoder/head parameter-group LR (OneCycleLR max_lr)
WARMUP_FRACTION = 0.1         # OneCycleLR pct_start (warmup = first 10% of training)
FINETUNE_PATIENCE = 8         # early-stopping patience on val IoU

# ─── Multi-Sensor (multiscale learning) ──────────────────────────────
SENSORS = ["cartosat3", "liss4", "sentinel2"]         # dataset folders under data/processed
# Canonical mentor-specified GSDs; geotransform `res` is ground truth where available.
GSD_METERS = {"cartosat3": 1.0, "liss4": 5.8, "sentinel2": 10.0}
# Per-sensor, per-band percentile normalization (uint16 -> [0,1]).
NORM_PERCENTILE = (2, 98)
NORM_STATS_PATH = "data/processed/_norm_stats.json"
# "sensor_balanced" draws equal samples per sensor per batch (C3=47k would otherwise drown the epoch).
SAMPLE_STRATEGY = "sensor_balanced"
# Split by sensor-city: one entire sensor held out for validation (no leakage).
# LISS-IV (5.8m) is the intermediate GSD -> train on cartosat3 (1m) + sentinel2 (10m).
# Now OFF: fine-tuning uses per-sensor 80/20 tile splits (VAL_SPLIT_FRACTION) so
# LISS-IV also trains and the model sees every sensor's resolution during training.
VAL_HOLDOUT_SENSOR = None
# Class imbalance handling. FocalLoss + POS_WEIGHT is the primary lever; tile-level
# sparse oversampling is secondary and ON for fine-tuning.
MIN_ROAD_FRAC = 0.005     # skip tiles with < 0.5% road pixels at index build (in-memory)
SAMPLE_WEIGHTED_BY_ROAD = True   # upweight sparse-road tiles in balanced batches
SPARSE_ROAD_FRAC = 0.05   # a tile is "sparse" below this road-pixel fraction
SPARSE_OVERSAMPLE = 3.0   # sampling weight given to sparse tiles (3x requested)
ROAD_FRACS_PATH = "data/processed/road_fracs.json"  # per-tile road fractions (compute_road_fracs.py)
# Phase C: FiLM conditioning on log10(GSD). Default OFF; enable only if cross-sensor IoU gap is weak.
CONDITION_ON_GSD = False
# albumentations RandomScale range for multiscale robustness.
SCALE_AUG = (0.75, 1.5)
# In-memory decoded-tile LRU cache (in samples). 0 = off (local). On Kaggle set
# ~10000 so the 4 CPU cores stop re-decoding GeoTIFFs every epoch; the multiscale
# RandomScale still runs per-access, only the file decode is cached.
CACHE_SIZE = 0

# ─── Loss Weights ────────────────────────────────────────────────────
# Composite = Dice + Focal + Boundary + Connectivity, weights normalized to sum 1.0.
# Focal carries the pixel-level class-imbalance load (thin roads vs background).
LOSS_TYPE = "focal_combo"
FOCAL_GAMMA = 1.5     # down-weight easy background pixels
FOCAL_ALPHA = 0.25    # relative weight of the positive (road) class
POS_WEIGHT = 3.0      # BCE/Focal positive-class weight (low -> avoids flooding roads)
LOSS_ALPHA = 0.5      # Dice
LOSS_BETA = 0.2       # Focal
LOSS_GAMMA = 0.2      # Skeleton (distance-transform weighted BCE)
LOSS_DELTA = 0.1      # Connectivity (noisy term, deliberately small)

# ─── Inference / Evaluation ─────────────────────────────────────────
INFER_THRESHOLD = 0.45        # base binarization threshold (anti-flood, default 0.45)
BEST_THRESHOLD = 0.45         # binarization threshold used during training validation
INFER_MIN_COMPONENT_PX = 16  # drop predicted blobs smaller than this (speckle removal)
INFER_CLOSE_KERNEL = True    # morphological close to stitch thin-road gaps in masks
INFER_CHECKPOINT = "outputs/models/terrain_best.pth"  # weights used by inference + main.py

# Anti-flood (fixes the "predict road everywhere" degenerate that blew LISS-IV to
# ~77% road). Per tile: if the predicted road fraction exceeds INFER_MAX_ROAD_FRAC,
# raise the threshold in INFER_THRESHOLD_STEP increments until it drops under the cap
# (or hits INFER_THRESHOLD_MAX). No retraining needed. Roads are rarely >50% of a
# 256x256 tile, so capping the per-tile fraction prunes the false-positive flood.
INFER_ADAPTIVE_THRESHOLD = True
INFER_MAX_ROAD_FRAC = 0.25   # per-tile road-fraction cap (was 0.5; fewer flooded roads)
INFER_THRESHOLD_START = 0.45 # base threshold used when the tile is NOT flooding
INFER_THRESHOLD_STEP = 0.05  # how far to shift the threshold per escalation
INFER_THRESHOLD_MAX = 0.70   # never push the threshold past this (was 0.9)

# ─── Graph Healing ───────────────────────────────────────────────────
MAX_HEAL_DISTANCE = 200.0       # meters (gap gate, haversine) — was 0.002 degrees (~200m)
LAMBDA_ANGULAR = 0.3            # smooth angular penalty weight
MAX_ANGLE_HARD = 90             # hard reject above this (degrees)
KINK_TURN_DEG = 10              # near-collinear threshold for kink removal (degrees)
HEAL_KINK_SIMPLIFY = False      # enable optional kink-simplification pass

# Per-healed-edge max gap (meters). Hard cap: no healed bridge may exceed this,
# guaranteeing no absurd cross-city occlusion joins are ever emitted. Lowered
# from 300 -> 200 to keep bridges locally plausible and avoid long awkward joins.
HEAL_MAX_GAP_M = 200.0
# Run a second angular pass with the hard gate relaxed (smooth penalty only)
# to recover straight-ish joins missed by the conservative first pass.
HEAL_RELAXED_GATE = True
# Bridge stranded degree-0 isolated pixels to their nearest road component
# (within HEAL_MAX_GAP_M). This collapses fragment count / raises LCC.
HEAL_RECOVER_ISOLATED = True
# Relative cost weight for recovered isolated-node bridges (vs angular penalty).
HEAL_ISOLATED_WEIGHT = 1.0
# Insert real junction nodes wherever a healed chord crosses another road,
# so every visual crossing becomes a genuine connected intersection.
HEAL_PLANARIZE = True

# ─── Centrality ──────────────────────────────────────────────────────
BETWEENNESS_K = 500             # approximation pivot count (k=2000 is ~4x slower on big cities)
DEFAULT_SPEED_KMH = 30          # fallback for missing OSM speed data

# ─── GCN ─────────────────────────────────────────────────────────────
GCN_HIDDEN = 64
GCN_EPOCHS = 200
GCN_LR = 0.01
GCN_DROPOUT = 0.3
# 5 cheap node features: degree centrality, clustering, k-core, 2-hop degree,
# mean neighbor degree (all O(n)-ish; no global shortest-path at inference).
GCN_INPUT_FEATURES = 5
# "regression" = continuous per-graph standardized log1p(betweenness) target
# (rank at eval/dashboard); "binary" = top-K% gatekeeper classification.
GCN_TARGET = "regression"
PSEUDO_LABEL_PERCENTILE = 90   # top 10% by centrality = gatekeeper (experiment)
GCN_TRAIN_SPLIT = 0.8          # 80/20 train/val

# ─── Resilience ───────────────────────────────────────────────────────
RESILIENCE_MAX_EFFICIENCY_NODES = 2000   # skip global efficiency above this (O(n^3))

# ─── GCN OSM training cities ──────────────────────────────────────────
OSM_TRAIN_CITIES = {
    "chandigarh": "Chandigarh, India",
    "jaipur": "Jaipur, India",
    "varanasi": "Varanasi, India",
    "bhopal": "Bhopal, India",
    "indore": "Indore, India",
    "nagpur": "Nagpur, India",
}
OSM_HOLDOUT_CITIES = {
    "bengaluru": "Bengaluru, India",  # ISRO stand-ins for now
    "delhi": "Delhi, India",
    "hyderabad": "Hyderabad, India",
}

# ─── Dashboard ───────────────────────────────────────────────────────
DASHBOARD_PORT = 8501
MAP_CENTER = [12.9716, 77.5946]  # Bengaluru
MAP_ZOOM = 12
