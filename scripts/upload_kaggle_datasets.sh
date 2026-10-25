#!/usr/bin/env bash
#
# Part 0 of the Kaggle fine-tuning setup — one-time local prep.
#
#  1. Builds the processed-tiles archive (contains processed/... so it maps 1:1
#     to the local layout expected by MultiSensorDataset).
#  2. Builds the tiny code dataset (config.py + segmentation/*.py).
#  3. Uploads both as PRIVATE Kaggle datasets via the CLI.
#
# Prereqs:  pip install kaggle
#           ~/.kaggle/kaggle.json (kaggle.com -> Settings -> API -> Create New Token)
#
# Usage:    KAGGLE_USERNAME=your_kaggle_name bash scripts/upload_kaggle_datasets.sh
#
set -euo pipefail

USER="${KAGGLE_USERNAME:?set KAGGLE_USERNAME to your Kaggle handle}"
STAGE="$(mktemp -d /tmp/terrain_kaggle.XXXXXX)"
trap 'echo "cleaning $STAGE"; rm -rf "$STAGE"' EXIT

echo "== building processed-tiles archive (13-14 GB) =="
mkdir -p "$STAGE/terrain-tiles"
( cd data && zip -r -1 "$STAGE/terrain-tiles/processed.zip" processed \
      -x 'processed/cartosat3/zi2W6ZXN/*' -x 'processed/cartosat3/text.py' )

echo "== building code dataset =="
mkdir -p "$STAGE/terrain-code"
cp config.py segmentation/*.py "$STAGE/terrain-code/"
printf 'timm\nalbumentations\nrasterio\n' > "$STAGE/terrain-code/requirements.txt"

echo "== metadata =="
cat > "$STAGE/terrain-tiles/dataset-metadata.json" <<EOF
{"id": "$USER/terrain-tiles", "title": "TERRAIN Processed Tiles", "licenses": [{"name": "CC0-1.0"}]}
EOF
cat > "$STAGE/terrain-code/dataset-metadata.json" <<EOF
{"id": "$USER/terrain-code", "title": "TERRAIN Segmentation Code", "licenses": [{"name": "CC0-1.0"}]}
EOF

echo "== uploading (private by default) =="
kaggle datasets create -p "$STAGE/terrain-tiles" -q
kaggle datasets create -p "$STAGE/terrain-code" -q

echo
echo "Done. In the notebook add Input -> $USER/terrain-tiles and $USER/terrain-code"
echo "(input folders will be /kaggle/input/terrain-tiles and /kaggle/input/terrain-code)."
