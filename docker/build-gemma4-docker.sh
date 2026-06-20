#!/usr/bin/env bash
# Run the Gemma 4 MLC build inside Docker
# Usage: ./docker/build-gemma4-docker.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
GEMMA4_BUILD="$PROJECT_DIR/gemma4-build"
MODELS_DIR="$GEMMA4_BUILD/models"
DIST_DIR="$GEMMA4_BUILD/dist"

# Verify model weights exist
if [[ ! -f "$MODELS_DIR/tokenizer.json" ]]; then
  echo "ERROR: Model weights not found at $MODELS_DIR"
  echo "Run first: mkdir -p $MODELS_DIR && hf download google/gemma-4-E4B-it --local-dir $MODELS_DIR"
  exit 1
fi

mkdir -p "$DIST_DIR"

echo "Building Docker image (first time: ~60 min)..."
docker build \
  -f "$SCRIPT_DIR/Dockerfile.gemma4" \
  -t mlc-gemma4-builder \
  "$SCRIPT_DIR"

echo "Running weight conversion + WASM compile..."
docker run --rm \
  -v "$MODELS_DIR:/work/models" \
  -v "$DIST_DIR:/work/dist" \
  -e MODEL_NAME="gemma-4-E4B-it" \
  -e QUANT="q4f16_1" \
  -e CONV_TEMPLATE="gemma4_instruction" \
  mlc-gemma4-builder

echo ""
echo "Output files are in: $DIST_DIR"
echo ""
echo "Next: upload to HuggingFace and update app/app.js mlcEntry URLs"
