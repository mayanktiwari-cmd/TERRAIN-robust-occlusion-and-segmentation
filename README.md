# TERRAIN

**T**opologically **E**nhanced **R**oad **R**esilience **A**nalysis & **I**ntelligence **N**etwork

End-to-end geospatial AI pipeline for ISRO's **Bharatiya Antariksh Hackathon 2026 (PS4)**.  
Developed by **Team Nebula Decoders**

Extracts occlusion-robust road networks from satellite imagery and maps urban vulnerability using graph-theoretic resilience analysis.

---

## Pipeline

```
Satellite Tile → Preprocessing → Segmentation (Swin-UNet)
→ Skeletonization → Graph Construction → Topological Healing
→ Criticality Analysis (GCN) → Dashboard (Streamlit + Folium)
```

## Team

| Member | Role |
|---|---|
| **Mayank** (Lead) | Swin-UNet Segmentation (Swin Transformer track) |
| **Sagnik** | GCN (Criticality: centrality, RoadGCN, resilience)  + assist Swin-UNet|
| **Sumit** | Graph Pipeline (skeletonization, graph building, healing) |
| **Ekas** | Integration + Pipeline Runner + Web Dashboard implementation|

## Repository Structure

```
TERRAIN/
├── config.py              # All hyperparameters in one place
├── requirements.txt       # Dependencies
├── main.py                # Pipeline runner (stub — implement during finals)
├── data/                  # Download scripts + preprocessing
├── segmentation/          # Dataset, Swin-UNet model, loss, train/eval/inference
├── graph/                 # Skeletonization, graph building, healing, visualization
├── criticality/           # Criticality (GCN): centrality, RoadGCN, training, resilience index
├── dashboard/             # Streamlit + Folium app
├── tests/graph-tests/     # Graph verification harness, data sampling, runners
├── experiments/           # Jupyter notebooks (populated during finals)
└── outputs/               # Models, masks, graphs, figures
```

## Graph pipeline — implemented & testable

`graph/` (skeletonize → build → multi-pass topological healing → planarization →
visualization) is implemented and covered by a self-check interface under
`tests/graph-tests/`. The runner is **map- and window-agnostic**: registered maps
are `bengaluru`, `hyderabad`, `delhi` (add any map to `CITIES` in
`tests/graph-tests/harness.py`).

### Setup

The pipeline runs in the `terrain-graph` conda env (install once):

```bash
conda create -n terrain-graph python=3.11
conda activate terrain-graph
pip install -r requirements.txt     # graph-pinned stack is annotated in the file
```

> **Always use the `terrain-graph` env.** Running `python tests/...` with the base
> env fails (`ModuleNotFoundError: No module named 'skimage'`, etc.).

### Run it (every command below is from the repo root)

**1. Verify the environment + pipeline** (exit 0 = all pass — no data needed):

```bash
conda run -n terrain-graph python tests/graph-tests/check_environment.py                  # deps + versions + synthetic smoke
conda run -n terrain-graph python tests/graph-tests/check_environment.py --map hyderabad  # + a real map window (geo)
conda run -n terrain-graph python tests/graph-tests/check_environment.py --full            # all registered maps
```

**2. Run the full pipeline on any map / any window** (print report + save plots to `outputs/graph-output-tests/`):

```bash
conda run -n terrain-graph python tests/graph-tests/run_pipeline.py --city delhi --window 512 --center "2048,2048"
conda run -n terrain-graph python tests/graph-tests/run_pipeline.py --city bengaluru --window 512 --center "2304,3072"
conda run -n terrain-graph python tests/graph-tests/run_pipeline.py --city hyderabad --window 512 --center "10240,12288"

conda run -n terrain-graph python tests/graph-tests/run_pipeline.py --list                # list registered maps
```

**PowerShell gotcha:** quote the `--center` value. In PowerShell the comma is an
array separator, so `--center 2048,2048` is split into two args. Write it as
`--center "2048,2048"` / `--center '2048,2048'`. (On Linux/macOS shells, unquoted
`--center 2048,2048` is fine.)

`--center` takes `col,row` (top-left of the window). Omit it to use the image
center. Other flags: `--no-geo` (pixel units instead of geographic metres),
`--skip-centrality` (faster), `--no-plot`, `--save-mask` (writes to
`tests/graph-tests/data/`), `--window N`.

**3. Empirically measure per-stage scaling** (heal is `O(H·E)`, superlinear):

```bash
conda run -n terrain-graph python tests/graph-tests/test_complexity.py --city hyderabad --windows 256,512,1024
```

### Reading the results

- **Topology target:** `components -> 1` and `LCC -> ~1.0` (bridges collapse the
  fragmented road shards into one street network).
- **Gap cap:** `max bridge gap ≤ 200` `HEAL_MAX_GAP_M`. The metric counts **true
  bridges** (original-node→original-node), excluding planarization-split original
  road segments (see `tests/graph-tests/harness.py::compute_metrics`).
- **Criticality preview:** betweenness `k=2000` top-10 gatekeepers + `k=500`
  vs `k=2000` stability.
- **Plots:** `outputs/graph-output-tests/healing_comparison_<city>.png` (raw vs
  healed), `outputs/graph-output-tests/healing_result_<city>.png` (blue = roads,
  red = healed bridges).

### Integrated runs (graph-heal + GCN) — `terrain-torch` env

The GCN step needs `torch`, so these runners use the separate **`terrain-torch`**
env (Python 3.11 + torch + the graph stack). Create it once:

```bash
conda create -n terrain-torch python=3.11
conda activate terrain-torch
pip install torch numpy networkx geopandas rasterio scikit-image matplotlib pyogrio
```

**1. Integrated single-window run** (mask → heal → GCN gatekeepers → resilience;
saves graph `.pkl`, report `.json`, plots):

```bash
conda run -n terrain-torch python tests/graph-tests/run_integrated.py \
  --city hyderabad --window 512 --center "10240,12288"
```

**2. Full-map running** (entire city, multiprocessed tiles → GeoJSON + Leaflet
HTML + PNG + report):

```bash
conda run -n terrain-torch python tests/graph-tests/run_fullmap.py --city hyderabad
# outputs: hyderabad_fullmap.geojson (EPSG:32644), hyderabad_fullmap.html
#          (Leaflet map: blue roads / red bridges / magenta gatekeepers),
#          hyderabad_fullmap.png
```

Open the `.html` in a browser, or the `.geojson` in **QGIS**, to pan/zoom the whole
healed network with layer toggles.

**3. Validate how close the map is to ground truth** (Dice/IoU via `--mask`,
bridge legitimacy, components, LCC, gap):

```bash
conda run -n terrain-torch python tests/graph-tests/validate_map.py \
  --city hyderabad --window 512 --center "10240,12288"
conda run -n terrain-torch python tests/graph-tests/validate_map.py ... --mask <seg_mask.npy>
```

**4. Pipeline demonstration visual** (4-panel storyboard:
truth → raw fragmented → healed → GCN gatekeepers):

```bash
conda run -n terrain-torch python tests/graph-tests/make_visual.py \
  --city hyderabad --window 256 --center "10240,12288"
# -> outputs/graph-output-tests/visual_pipeline_hyderabad.png
```

### Multi-sensor connectors (mask → graph → GCN)

The three segmentation output mosaics are the real inputs, and each has its own
**dedicated connector script**, chaining its mask through the graph module →
GCN:

| Sensor | Mosaic tif | CRS | Connector |
|--------|-----------|-----|-----------|
| cartosat3 | `cartosat3_mosaic.tif` | EPSG:32644 | `connector_cartosat3.py` |
| sentinel2 | `sentinel2_mosaic.tif` | EPSG:32643 | `connector_sentinel2.py` |
| liss4 | `liss4_mosaic.tif` | EPSG:32643 | `connector_liss4.py` |

Each runs the full chain on its own mask: read a road-dense window →
`harness.run_graph` (skeletonize → build → heal) → `graph_to_tensor` →
`RoadGCN` → `rank_gatekeepers` → resilience index, then writes
`connector_<sensor>.png`, `connector_<sensor>_report.json` and
`connector_<sensor>_graph.pkl`. The GCN is trained on the healed graph on the
fly if the `terrain_best.pth` checkpoint is absent.

```bash
conda run -n terrain-torch python tests/graph-tests/connector_cartosat3.py --gcn-epochs 120
conda run -n terrain-torch python tests/graph-tests/connector_sentinel2.py --gcn-epochs 120
conda run -n terrain-torch python tests/graph-tests/connector_liss4.py      --gcn-epochs 120
```

> **Note:** topology targets and gap caps are printed in each report
> (`outputs/graph-output-tests/*_report.json`).

## GCN criticality analysis — implemented & shipped

`criticality/` implements the **Criticality Analysis (GCN)** step of the
pipeline: given a healed road graph, it ranks the most critical nodes
("gatekeepers") and computes a city-wide **resilience index**. The shipped
checkpoint and its held-out evaluation are committed under `outputs/`.

### Methodology

- **Problem framing — binary node classification.** A node is a *gatekeeper*
  if it lies in the top `K%` of betweenness-central nodes. Betweenness
  (Brandes, edge-`length`-weighted) is **train-time only** — it is the
  pseudo-label source and is never computed at inference.
- **Inference features (2, scale-free).** Per-node *degree centrality* (dense
  hubs) and *local clustering coefficient* (bridges at low degree). Both are
  per-graph z-scored (`zscore_normalize` in `graph_adapter.py`) so the model
  only sees relative structure and generalizes across city sizes.
- **Model — `RoadGCN`.** A pure-PyTorch 2-layer GCN (no torch-geometric).
  Self-loops + symmetric normalization `S = D^-1/2 (A + I) D^-1/2`, two
  propagation rounds, ReLU, dropout, sigmoid predict. It is **inductive**: the
  same weights are applied to any city graph regardless of node count.
- **Split by city, not by node.** Trained on 6 OSM cities, held out on 3 ISRO
  city stand-ins (`config.OSM_TRAIN_CITIES` / `OSM_HOLDOUT_CITIES`) — no
  train/val leakage, and the model must generalize cross-city.
- **Evaluation (per-city, masked held-out nodes).** `Coverage@K` (fraction of
  true gatekeepers recovered in the top-K% predicted), `AUC`, and `Spearman`
  rank correlation — see `criticality/gcn_evaluate.py`.
- **Resilience index.** Ordered node-removal cascade on the betweenness
  ranking → area under the giant-component-fraction curve, normalized to
  [0, 1] (higher = more robust to losing critical nodes). The per-step
  `removal_log` feeds the dashboard.

### Module map

| File | Responsibility |
|---|---|
| `criticality/centrality.py` | Betweenness (train), gatekeeper detection (per-city relative percentile cut), feature extraction |
| `criticality/graph_adapter.py` | Deterministic node ordering, per-graph z-scoring, graph → `(x, edge_index, y, node_ids)` tensors |
| `criticality/gcn_model.py` | `RoadGCN` — 2-layer pure-PyTorch GCN |
| `criticality/gcn_train.py` | Pseudo-labels + single- and multi-city inductive training |
| `criticality/gcn_evaluate.py` | `rank_gatekeepers`, `Coverage@K`/`AUC`/`Spearman`, validation BCE |
| `criticality/resilience.py` | Node-removal cascade + resilience index |
| `criticality/terrain-gcn.ipynb` | Canonical Kaggle training notebook (GPU, exports the checkpoint) |

### Configuration (`config.py`)

```python
GCN_HIDDEN = 64        # hidden width
GCN_EPOCHS = 100
GCN_LR = 0.01
GCN_DROPOUT = 0.3
PSEUDO_LABEL_PERCENTILE = 90   # top 10% by centrality = gatekeeper
BETWEENNESS_K = 500            # Brandes pivot count (CPU-feasible)
MODEL_SAVE_PATH = "outputs/models/terrain_best.pth"
```

### Artifacts (committed)

- `outputs/models/terrain_best.pth` — shipped checkpoint. `RoadGCN` weights:
  `W0 (64, 2)`, `W1 (1, 64)` (2 features, hidden 64).
- `outputs/metrics.json` — held-out evaluation on the three ISRO city
  stand-ins, plus the predicted gatekeeper node ids per city.

### Results — held-out cities

| City | Coverage@K | AUC | Spearman |
|---|---|---|---|
| Bengaluru | 0.316 | 0.688 | 0.260 |
| Delhi | 0.278 | 0.641 | 0.196 |
| Hyderabad | 0.289 | 0.663 | 0.227 |

The ranking is most informative at the top: recovering **~3× random** gatekeepers
at the final top-10% label setting (vs ~1.5× random at top-20%). Across all
experiments (model width, feature count, label threshold) the cross-city AUC
sits at a 0.64–0.69 ceiling with Spearman 0.20–0.26 — diminishing returns
beyond the shipped config, so experimentation is closed.

### Training & running

```bash
# Full pipeline on a tile (loads the checkpoint, ranks gatekeepers, resilience index):
python main.py --skip-dashboard

# Local multi-city training over OSM cities (needs internet + osmnx):
python main.py --train

# Heavy GPU training: open criticality/terrain-gcn.ipynb on Kaggle
#   (Accelerator: GPU T4x2, Internet: On, GH_PAT secret for the repo clone)
```

- If `outputs/models/terrain_best.pth` is missing or fails to load, `main.py`
  degrades gracefully to pure betweenness rankings (no crash).
- **CPU feasibility:** betweenness at `k=500` runs all 9 cities in ~70 min;
  `k=2000` is ~4× slower (~4.8 h) for little quality gain. Planned speedups:
  stdlib multiprocessing over Brandes pivots + a pickle disk cache.

## Status

Graph pipeline and criticality (GCN) are implemented and verified (see
`tests/graph-tests/` and `outputs/metrics.json`). Segmentation and the
dashboard integration remain to be completed for finals. Graph test scripts
live under `tests/graph-tests/` (sampled masks in `tests/graph-tests/data/`)
and their figures under `outputs/graph-output-tests/`.

## Notes

- All numerical metrics in this repo (e.g. baseline IoU, F1, healing coverage, centrality values) are from **preliminary experiments using different building blocks** (UNet++/ResNet-34, not the final Swin-UNet + GCN stack). They are **not** regression targets for the final pipeline.
- Dataset downloads (DeepGlobe, SpaceNet SN5, Sentinel-2) require Kaggle/AWS/Copernicus authentication at runtime and are stubbed out.
- The final pipeline will use **Cartosat-3 data** provided during the hackathon — the pipeline is resolution-agnostic by design.
