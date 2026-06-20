#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-gemma-4-E4B-it}"
QUANT="${QUANT:-q4f16_1}"
CONV_TEMPLATE="${CONV_TEMPLATE:-gemma4_instruction}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-8192}"
PREFILL_CHUNK="${PREFILL_CHUNK:-1024}"

MODEL_LOCAL="/work/models"
MLC_OUT="/work/dist/${MODEL_NAME}-${QUANT}-MLC"
WASM_OUT="/work/dist/libs/${MODEL_NAME}-${QUANT}-MLC-webgpu.wasm"

source /build/emsdk/emsdk_env.sh 2>/dev/null || true

echo "[1/3] Converting weights → $MLC_OUT"
python -m mlc_llm convert_weight "$MODEL_LOCAL" \
    --quantization "$QUANT" \
    -o "$MLC_OUT"

echo "[2/3] Generating chat config (template: $CONV_TEMPLATE)"
python -m mlc_llm gen_config "$MODEL_LOCAL" \
    --quantization "$QUANT" \
    --conv-template "$CONV_TEMPLATE" \
    --context-window-size "$CONTEXT_WINDOW" \
    -o "$MLC_OUT/" || \
python -m mlc_llm gen_config "$MODEL_LOCAL" \
    --quantization "$QUANT" \
    --conv-template gemma_instruction \
    --context-window-size "$CONTEXT_WINDOW" \
    -o "$MLC_OUT/"

echo "[3/3] Compiling WebGPU WASM → $WASM_OUT"
mkdir -p "/work/dist/libs"
python -m mlc_llm compile "$MLC_OUT/mlc-chat-config.json" \
    --device webgpu \
    --prefill-chunk-size "$PREFILL_CHUNK" \
    -o "$WASM_OUT"

echo ""
echo "Done! Output files:"
echo "  $MLC_OUT/"
echo "  $WASM_OUT"
