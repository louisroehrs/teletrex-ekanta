e#!/usr/bin/env bash
# =============================================================================
# build-gemma4.sh — Compile Gemma 4 E2B for WebGPU / WebLLM on Intel Mac
#
# What this does:
#   1. Installs prerequisites (brew, conda env, Emscripten)
#   2. Clones mlc-llm and merges Gemma 4 draft PR #3485
#   3. Builds mlc-llm + TVM from source
#   4. Downloads Gemma 4 E2B weights from HuggingFace
#   5. Converts weights to MLC q4f16_1 format
#   6. Generates chat config
#   7. Compiles WebGPU WASM library
#
# Prerequisites you must handle manually before running:
#   - Xcode Command Line Tools:  xcode-select --install
#   - Homebrew:                  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
#   - Conda (miniforge):         https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh
#   - HuggingFace account with Gemma 4 license accepted at:
#       https://huggingface.co/google/gemma-4-E2B-it
#   - HuggingFace CLI logged in: pip install huggingface_hub && huggingface-cli login
#
# Usage:
#   chmod +x build-gemma4.sh
#   ./build-gemma4.sh
#
# Outputs (copy these to HuggingFace / GitHub):
#   dist/gemma-4-E2B-it-q4f16_1-MLC/   — weights + config
#   dist/libs/gemma-4-E2B-it-q4f16_1-MLC-webgpu.wasm  — model library
# =============================================================================

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────────
HF_MODEL="google/gemma-4-E4B-it"          # HuggingFace model ID
MODEL_NAME="gemma-4-E4B-it"
QUANT="q4f16_1"
CONV_TEMPLATE="gemma4_instruction"        # added by PR #3485; fall back to gemma_instruction if error
CONTEXT_WINDOW=8192
PREFILL_CHUNK=1024
CONDA_ENV="mlc-gemma4-py311"
WORK_DIR="$(cd "$(dirname "$0")" && pwd)/gemma4-build"
CPUS=$(sysctl -n hw.logicalcpu)

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()   { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# =============================================================================
# STEP 0 — Sanity checks
# =============================================================================
info "Checking prerequisites…"

command -v brew  >/dev/null 2>&1 || die "Homebrew not found. Install it first: https://brew.sh"
command -v conda >/dev/null 2>&1 || die "Conda not found. Install Miniforge: https://github.com/conda-forge/miniforge#download"
command -v git   >/dev/null 2>&1 || die "git not found. Run: xcode-select --install"

# Intel Mac check
ARCH=$(uname -m)
[[ "$ARCH" == "x86_64" ]] || warn "This script targets Intel (x86_64). You are on $ARCH — it may still work."

ok "Prerequisites look good (arch: $ARCH, cpus: $CPUS)"
mkdir -p "$WORK_DIR"
cd "$WORK_DIR"

# =============================================================================
# STEP 1 — Homebrew dependencies
# =============================================================================
info "Installing brew dependencies…"
brew install git-lfs cmake ninja llvm@15 2>/dev/null || true
git lfs install --skip-repo 2>/dev/null || true

# Put brew LLVM on PATH (mlc-llm needs llvm-config)
LLVM_PREFIX="$(brew --prefix llvm@15)"
export PATH="$LLVM_PREFIX/bin:$PATH"
export LDFLAGS="-L$LLVM_PREFIX/lib"
export CPPFLAGS="-I$LLVM_PREFIX/include"
ok "LLVM: $(llvm-config --version)"

# Rust — tokenizers-cpp requires rustc >= 1.79; brew rust may be older
RUST_VERSION=$(rustc --version 2>/dev/null | awk '{print $2}' || echo "0")
RUST_MINOR=$(echo "$RUST_VERSION" | cut -d. -f2)
if [[ "$RUST_MINOR" -lt 79 ]] 2>/dev/null; then
  info "Rust $RUST_VERSION is too old (need >= 1.79) — installing via rustup…"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
fi
ok "Rust: $(rustc --version)"

# =============================================================================
# STEP 2 — Conda environment
# =============================================================================
info "Setting up conda environment '$CONDA_ENV'…"

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if conda env list | grep -q "^$CONDA_ENV "; then
  warn "Conda env '$CONDA_ENV' already exists — reusing it."
else
  conda create -n "$CONDA_ENV" -c conda-forge \
    "cmake>=3.24" rust git python=3.11 ninja -y
fi

conda activate "$CONDA_ENV"
pip install -q huggingface_hub hf_transfer
ok "Conda env active: $(python --version)"

# =============================================================================
# STEP 3 — Clone mlc-llm and apply Gemma 4 PR #3485
# =============================================================================
info "Cloning mlc-llm…"
if [[ -d mlc-llm/.git ]]; then
  warn "mlc-llm already cloned — skipping clone, pulling latest."
  git -C mlc-llm pull --ff-only
else
  git clone --recursive https://github.com/mlc-ai/mlc-llm.git
fi

cd mlc-llm

# Fetch and merge the Gemma 4 draft PR
info "Applying Gemma 4 PR #3485…"
git fetch origin pull/3485/head:gemma4-pr 2>/dev/null || {
  warn "Could not fetch PR branch — it may have been merged already. Continuing."
}
if git branch --list gemma4-pr | grep -q gemma4-pr; then
  # Only merge if not already on it or already merged
  if ! git log --oneline | grep -q "$(git log gemma4-pr -1 --format='%H' 2>/dev/null || echo 'NONE')"; then
    git merge --no-edit gemma4-pr || {
      warn "Auto-merge had conflicts. Attempting to continue with 'ours' strategy on conflicts…"
      git checkout --theirs . 2>/dev/null || true
      git add . 2>/dev/null || true
      git merge --continue --no-edit || die "Could not merge PR #3485. Resolve conflicts manually in $WORK_DIR/mlc-llm"
    }
  else
    ok "PR #3485 already in tree."
  fi
fi

# =============================================================================
# STEP 4 — Build mlc-llm from source (includes TVM as submodule)
# =============================================================================
info "Building mlc-llm from source (this takes 30–60 min)…"

mkdir -p build && cd build

# Generate cmake config interactively via the helper, then patch for Metal + LLVM
if [[ ! -f config.cmake ]]; then
    python ../cmake/gen_cmake_config.py 
fi

# The interactive prompt asks for platform; "1" selects Metal (macOS)

# Write cmake config from scratch — avoids gen_cmake_config.py interactive prompt
# and uses the full llvm-config path which works on Intel Mac brew installs
LLVM_CONFIG_BIN="$(brew --prefix llvm@15)/bin/llvm-config"
[[ -x "$LLVM_CONFIG_BIN" ]] || die "llvm-config not found at $LLVM_CONFIG_BIN — is llvm@15 installed?"

cat > config.cmake <<CMAKECFG
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(USE_LLVM "$LLVM_CONFIG_BIN")
set(USE_METAL ON)
set(USE_VULKAN OFF)
set(USE_CUDA OFF)
set(USE_OPENCL OFF)
set(HIDE_PRIVATE_SYMBOLS OFF)
set(USE_LIBBACKTRACE OFF)
set(BACKTRACE_SUPPORT OFF)
CMAKECFG

cmake .. -G Ninja -DUSE_LIBBACKTRACE=OFF -DBACKTRACE_SUPPORT=OFF
cmake --build . --parallel "$CPUS"

cd ..  # back to mlc-llm root

# Install the Python package in-place
pip install -q scikit-build-core
pip install -e "." --no-build-isolation -q
export PYTHONPATH="$WORK_DIR/mlc-llm/3rdparty/tvm/python:$WORK_DIR/mlc-llm/python:${PYTHONPATH:-}"
export DYLD_LIBRARY_PATH="$WORK_DIR/mlc-llm/build:${DYLD_LIBRARY_PATH:-}"

# Symlink built TVM libs to where the TVM Python package searches for them
mkdir -p "$WORK_DIR/mlc-llm/3rdparty/tvm/build/lib"
for lib in libtvm_runtime.dylib libtvm.dylib; do
  src="$WORK_DIR/mlc-llm/build/$lib"
  dst="$WORK_DIR/mlc-llm/3rdparty/tvm/build/lib/$lib"
  [[ -f "$src" && ! -e "$dst" ]] && ln -sf "$src" "$dst"
done

# NumPy 2.x is incompatible with the compiled TVM extension modules
pip install -q "numpy<2"

ok "mlc-llm built: $(python -m mlc_llm --version 2>/dev/null || echo 'installed')"

cd ..  # back to WORK_DIR

# =============================================================================
# STEP 5 — Install Emscripten (for WebGPU WASM compilation)
# =============================================================================
info "Setting up Emscripten…"
if [[ -d emsdk/.git ]]; then
  warn "emsdk already present — updating."
  git -C emsdk pull --ff-only
else
  git clone https://github.com/emscripten-core/emsdk.git
fi

cd emsdk
./emsdk install latest
./emsdk activate latest
# shellcheck disable=SC1091
source ./emsdk_env.sh
ok "Emscripten: $(emcc --version | head -1)"
cd ..

# =============================================================================
# STEP 6 — Download Gemma 4 weights from HuggingFace
# =============================================================================
info "Downloading $HF_MODEL weights…"
info "(You must have accepted the Gemma 4 license at huggingface.co/$HF_MODEL)"
info "If not logged in yet, run:  huggingface-cli login"
echo ""

MODEL_LOCAL="models"

if [[ -f "$MODEL_LOCAL/tokenizer.json" ]]; then
  warn "Model already downloaded — skipping."
else
  mkdir -p models
  hf download "$HF_MODEL" --local-dir models
fi
ok "Weights at $MODEL_LOCAL"

# =============================================================================
# STEP 7 — Convert weights to MLC q4f16_1 format
# =============================================================================
MLC_MODEL_DIR="dist/${MODEL_NAME}-${QUANT}-MLC"
info "Converting weights to MLC format → $MLC_MODEL_DIR …"

if [[ -f "$MLC_MODEL_DIR/mlc-chat-config.json" ]]; then
  warn "MLC weights already converted — skipping convert_weight."
else
  python -m mlc_llm convert_weight "./$MODEL_LOCAL" \
    --quantization "$QUANT" \
    -o "$MLC_MODEL_DIR"
fi
ok "Weights converted."

# =============================================================================
# STEP 8 — Generate MLC chat config
# =============================================================================
info "Generating chat config (template: $CONV_TEMPLATE)…"

python -m mlc_llm gen_config "./$MODEL_LOCAL" \
  --quantization "$QUANT" \
  --conv-template "$CONV_TEMPLATE" \
  --context-window-size "$CONTEXT_WINDOW" \
  -o "$MLC_MODEL_DIR/" || {
    warn "Template '$CONV_TEMPLATE' not found — retrying with 'gemma_instruction'…"
    mlc_llm gen_config "./$MODEL_LOCAL" \
      --quantization "$QUANT" \
      --conv-template gemma_instruction \
      --context-window-size "$CONTEXT_WINDOW" \
      -o "$MLC_MODEL_DIR/"
}
ok "Chat config written."

# =============================================================================
# STEP 9 — Compile WebGPU WASM model library
# =============================================================================
WASM_OUT="dist/libs/${MODEL_NAME}-${QUANT}-MLC-webgpu.wasm"
info "Compiling WebGPU WASM library → $WASM_OUT …"
info "(This takes another 10–30 min)"
mkdir -p dist/libs

python -m mlc_llm compile "./$MLC_MODEL_DIR/mlc-chat-config.json" \
  --device webgpu \
  --prefill-chunk-size "$PREFILL_CHUNK" \
  -o "$WASM_OUT"

ok "WASM compiled: $WASM_OUT ($(du -sh "$WASM_OUT" | cut -f1))"

# =============================================================================
# Done — print next steps
# =============================================================================
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN} Build complete!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
echo "Output files:"
echo "  Weights:  $WORK_DIR/$MLC_MODEL_DIR/"
echo "  WASM lib: $WORK_DIR/$WASM_OUT"
echo ""
echo "Next steps:"
echo ""
echo "  1. Upload weights to HuggingFace:"
echo "     huggingface-cli upload YOUR_HF_USERNAME/${MODEL_NAME}-${QUANT}-MLC \\"
echo "       $WORK_DIR/$MLC_MODEL_DIR ."
echo ""
echo "  2. Upload WASM to your HuggingFace repo (in a libs/ subfolder) or GitHub releases."
echo ""
echo "  3. Update app/app.js in Ekanta — set mlcEntry.model and mlcEntry.model_lib"
echo "     to point at your HuggingFace repo URLs."
echo ""
echo "  Suggested mlcEntry:"
MODEL_ID="${MODEL_NAME}-${QUANT}-MLC"
echo "    model:     'https://huggingface.co/YOUR_HF_USERNAME/${MODEL_ID}/resolve/main/'"
echo "    model_id:  '${MODEL_ID}'"
echo "    model_lib: 'https://huggingface.co/YOUR_HF_USERNAME/${MODEL_ID}/resolve/main/libs/${MODEL_NAME}-${QUANT}-MLC-webgpu.wasm'"
echo ""
