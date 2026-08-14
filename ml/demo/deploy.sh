#!/usr/bin/env bash
# Assemble a self-contained bundle and push it to a Hugging Face Space.
#
#   ./deploy.sh [namespace/space-name]      (default: yalishanda/dynamics-needed)
#
# The monorepo (ml/demo/) is the source of truth; the Space is just a target.
# Model weights + SoundFont are copied in here, never committed to the repo.
set -euo pipefail

SPACE="${1:-yalishanda/dynamics-needed}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
PROC="$REPO_ROOT/data/processed"
SF_SRC="$REPO_ROOT/sf/big/FluidR3_GM.sf2"
PKG_SRC="$REPO_ROOT/ml/src/drum_dynamics"
BUILD="$HERE/_build"

# Files Engine.load() expects in models/ (name in repo == name it looks for).
MODEL_FILES=(
  lightgbm_model.joblib
  lightgbm_features.json
  head_mdn.pt
  mdn_meta.json
  head_categorical.pt
  transformer_meta.json
)

echo ">> checking sources"
for f in "${MODEL_FILES[@]}"; do
  [[ -f "$PROC/$f" ]] || { echo "missing model file: $PROC/$f" >&2; exit 1; }
done
[[ -f "$SF_SRC" ]] || { echo "missing SoundFont: $SF_SRC" >&2; exit 1; }
[[ -d "$PKG_SRC" ]] || { echo "missing package: $PKG_SRC" >&2; exit 1; }

echo ">> assembling bundle in $BUILD"
rm -rf "$BUILD"
mkdir -p "$BUILD/models" "$BUILD/sf"
cp "$HERE/app.py" "$HERE/requirements.txt" "$HERE/packages.txt" "$HERE/README.md" "$BUILD/"
cp -R "$HERE/examples" "$BUILD/examples"
cp -R "$PKG_SRC" "$BUILD/drum_dynamics"
for f in "${MODEL_FILES[@]}"; do cp "$PROC/$f" "$BUILD/models/$f"; done
cp "$SF_SRC" "$BUILD/sf/FluidR3_GM.sf2"
find "$BUILD" -name '__pycache__' -type d -prune -exec rm -rf {} +

echo ">> ensuring Space exists: $SPACE"
hf repo create "$SPACE" --repo-type space --space-sdk gradio \
  --flavor zero-a10g --public --exist-ok

echo ">> uploading bundle"
# --delete prunes example files that were renamed/removed since the last deploy
# (hf upload does not delete remote files by default).
hf upload "$SPACE" "$BUILD" . --repo-type space \
  --exclude "**/__pycache__/**" --delete "examples/**"

echo ">> done: https://huggingface.co/spaces/$SPACE"
