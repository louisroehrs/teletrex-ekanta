#!/usr/bin/env bash
# Finish packaging an already-converted Gemma 4 MLC build:
# runs gen_config + compile only (skips convert_weight). Safe to re-run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GEMMA4_BUILD="$PROJECT_DIR/gemma4-build"
MODELS_DIR="$GEMMA4_BUILD/models"
DIST_DIR="$GEMMA4_BUILD/dist"

docker run --rm \
  -v "$MODELS_DIR:/work/models" \
  -v "$DIST_DIR:/work/dist" \
  -v "$SCRIPT_DIR/entrypoint-package.sh:/entrypoint-package.sh:ro" \
  -e MODEL_NAME="gemma-4-E4B-it" \
  -e QUANT="q4f16_1" \
  -e CONV_TEMPLATE="gemma3_instruction" \
  --entrypoint bash \
  mlc-gemma4-builder /entrypoint-package.sh

echo ""
echo "Package ready in: $DIST_DIR/gemma-4-E4B-it-q4f16_1-MLC"
echo "WASM in:          $DIST_DIR/libs"
