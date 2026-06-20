#!/usr/bin/env bash
# Package-only: weights are already converted in /work/dist/<MODEL>-<QUANT>-MLC.
# Run just gen_config (chat config + tokenizer) and compile (WebGPU WASM).
# Skips the expensive/OOM-prone convert_weight step.
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-gemma-4-E4B-it}"
QUANT="${QUANT:-q4f16_1}"
CONV_TEMPLATE="${CONV_TEMPLATE:-gemma3_instruction}"
CONTEXT_WINDOW="${CONTEXT_WINDOW:-8192}"
PREFILL_CHUNK="${PREFILL_CHUNK:-1024}"

MODEL_LOCAL="/work/models"
MLC_OUT="/work/dist/${MODEL_NAME}-${QUANT}-MLC"
WASM_OUT="/work/dist/libs/${MODEL_NAME}-${QUANT}-MLC-webgpu.wasm"

source /build/emsdk/emsdk_env.sh 2>/dev/null || true

# gen_config needs pydantic; compile's config-path parser imports requests.
python -c "import pydantic" 2>/dev/null || pip install --quiet pydantic
python -c "import requests" 2>/dev/null || pip install --quiet requests

if [[ ! -f "$MLC_OUT/tensor-cache.json" && ! -f "$MLC_OUT/ndarray-cache.json" ]]; then
  echo "ERROR: converted weights not found in $MLC_OUT — run convert_weight first."
  exit 1
fi

echo "[1/2] Generating chat config (template: $CONV_TEMPLATE)"
# gen_config imports mlc_llm.tokenizers, whose `mlc.Tokenizer` FFI object lives in
# the mlc-llm C++ runtime we deliberately don't build. Its ONLY use in gen_config
# is Tokenizer.detect_tokenizer_info(). Stub that module with Gemma's known-good
# tokenizer info (SentencePiece byte_fallback) so we avoid the heavy runtime build.
cat > /tmp/gen_config_driver.py <<'PYEOF'
import sys, types, runpy
from dataclasses import dataclass

@dataclass
class TokenizerInfo:
    token_postproc_method: str = "byte_fallback"
    prepend_space_in_encode: bool = True
    strip_space_in_decode: bool = True

class Tokenizer:
    @staticmethod
    def detect_tokenizer_info(_path):
        # Gemma uses a SentencePiece tokenizer → byte_fallback post-processing.
        return TokenizerInfo()

stub = types.ModuleType("mlc_llm.tokenizers")
stub.Tokenizer = Tokenizer
stub.TokenizerInfo = TokenizerInfo
sys.modules["mlc_llm.tokenizers"] = stub

sys.argv = ["mlc_llm"] + sys.argv[1:]
runpy.run_module("mlc_llm", run_name="__main__")
PYEOF

python /tmp/gen_config_driver.py gen_config "$MODEL_LOCAL" \
    --quantization "$QUANT" \
    --conv-template "$CONV_TEMPLATE" \
    --context-window-size "$CONTEXT_WINDOW" \
    -o "$MLC_OUT/" || \
python /tmp/gen_config_driver.py gen_config "$MODEL_LOCAL" \
    --quantization "$QUANT" \
    --conv-template gemma_instruction \
    --context-window-size "$CONTEXT_WINDOW" \
    -o "$MLC_OUT/"

echo "[2/2] Compiling WebGPU WASM → $WASM_OUT"
mkdir -p "/work/dist/libs"

# The WASM link step needs mlc_wasm_runtime.bc. Build just that (the full
# prep_emcc_deps.sh also wants npm + tvm web runtime, which we don't need here).
export TVM_SOURCE_DIR="/build/mlc-llm/3rdparty/tvm"
export MLC_LLM_SOURCE_DIR="/build/mlc-llm"
if [[ ! -f "$MLC_LLM_SOURCE_DIR/web/dist/wasm/mlc_wasm_runtime.bc" ]]; then
    echo "Building mlc_wasm_runtime.bc ..."
    ( cd "$MLC_LLM_SOURCE_DIR/web" && emcc --version >/dev/null && make )
fi
# TVM's own web runtime (wasm_runtime.bc) is also linked in. Build it and place a
# copy where TVM's libinfo searches (3rdparty/tvm/build).
if [[ ! -f "$TVM_SOURCE_DIR/build/wasm_runtime.bc" ]]; then
    echo "Building tvm wasm_runtime.bc ..."
    ( cd "$TVM_SOURCE_DIR/web" && TVM_HOME="$TVM_SOURCE_DIR" make )
    cp "$TVM_SOURCE_DIR/web/dist/wasm/wasm_runtime.bc" "$TVM_SOURCE_DIR/build/wasm_runtime.bc"
fi

python -m mlc_llm compile "$MLC_OUT/mlc-chat-config.json" \
    --device webgpu \
    --overrides "prefill_chunk_size=${PREFILL_CHUNK}" \
    -o "$WASM_OUT"

# WebLLM expects ndarray-cache.json; MLC may emit tensor-cache.json. Provide both.
if [[ -f "$MLC_OUT/tensor-cache.json" && ! -f "$MLC_OUT/ndarray-cache.json" ]]; then
    cp "$MLC_OUT/tensor-cache.json" "$MLC_OUT/ndarray-cache.json"
fi

echo ""
echo "Done! Package contents:"
ls -1 "$MLC_OUT" | grep -v params_shard
echo "WASM: $WASM_OUT"
