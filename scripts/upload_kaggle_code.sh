#!/usr/bin/env bash
#
# Upload the code dataset with the EXACT slug the notebook expects:
#   /kaggle/input/datasets/$KAGGLE_USERNAME/terrain-code5
# (notebook Cell 6 copies from "terrain-code5" into /kaggle/working).
#
# Prereqs: pip install kaggle  and  ~/.kaggle/kaggle.json
# Usage:   KAGGLE_USERNAME=your_kaggle_handle bash scripts/upload_kaggle_code.sh
set -euo pipefail

USER="${KAGGLE_USERNAME:?set KAGGLE_USERNAME to your Kaggle handle}"
STAGE="$(mktemp -d /tmp/terrain_code5.XXXXXX)"
trap 'rm -rf "$STAGE"' EXIT

mkdir -p "$STAGE/terrain-code5"
cp config.py segmentation/*.py "$STAGE/terrain-code5/"
printf 'timm\nalbumentations\nrasterio\nscipy\n' > "$STAGE/terrain-code5/requirements.txt"

cat > "$STAGE/terrain-code5/dataset-metadata.json" <<EOF
{"id": "$USER/terrain-code5", "title": "TERRAIN Segmentation Code", "licenses": [{"name": "CC0-1.0"}]}
EOF

echo "== uploading as $USER/terrain-code5 (updates existing version if present) =="
kaggle datasets create -p "$STAGE/terrain-code5" -q
echo "OK: mounted at /kaggle/input/datasets/$USER/terrain-code5"
