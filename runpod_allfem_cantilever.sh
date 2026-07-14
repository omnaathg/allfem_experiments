#!/usr/bin/env bash
#
# runpod_allfem_cantilever.sh
#
# One-shot bootstrap + run for the ALL-FEM cantilever-beam test on a
# RunPod Pod. This is the RunPod equivalent of `modal run
# modal_allfem_cantilever.py` — instead of Modal building a container
# image, this idempotently sets up Ollama + FEniCSx directly on your
# pod (a persistent VM), then runs test_allfem_cantilever.py.
#
# PREREQS (do this once, from the RunPod console, before running this
# script):
#   1. Deploy a Pod with a GPU (A10G is enough for the 3B model; see
#      the model->GPU table in RunPod_Modal_setup.md for bigger models).
#   2. Use the latest PyTorch template.
#   3. Under Expose HTTP Ports, add 11434 (only needed if you want to
#      hit Ollama from outside the pod — not required for this script).
#   4. Open the pod's Web Terminal.
#   5. Upload this file AND test_allfem_cantilever.py into the same
#      directory on the pod (drag-and-drop in the web terminal file
#      panel, or scp them up).
#
# USAGE (on the pod):
#   chmod +x runpod_allfem_cantilever.sh
#   ./runpod_allfem_cantilever.sh                                  # default: qwen3-short_think-fenics-local (confirmed PASS)
#   ./runpod_allfem_cantilever.sh rushikesh_67/llama3.2-2new        # a different model
#
# Safe to re-run: each step checks whether it already happened and
# skips if so, so re-running after a pod restart just re-starts the
# Ollama server and jumps straight to the test.
#
# 2026-07-12 field notes (see ALLFEM_RUNPOD_RUNBOOK.md for the full story):
#   - rushikesh_67/llama3.2-2new generates legacy `from dolfin import *`
#     code but with a scalar FunctionSpace instead of VectorFunctionSpace
#     -> real model bug, FAILs regardless of environment.
#   - rushikesh_67/qwen3-short_think-fenics-local generates correct
#     legacy dolfin code incl. `import mshr` -> PASSes (8.7% rel. error).
#   - Anaconda's default channels need their Terms of Service accepted
#     non-interactively before `conda create` will work on a fresh pod;
#     this script now does that up front.
#   - mshr is now installed alongside fenics in the same `conda create`
#     to avoid a second solve.

set -euo pipefail

MODEL="${1:-rushikesh_67/qwen3-short_think-fenics-local}"
FENICS_ENV_NAME="fenics"
FENICS_PACKAGE="fenics"   # legacy dolfin API (2019.1). Both models we've
                          # tried emit this style; if a future model emits
                          # modern `import dolfinx` code instead, change
                          # this to "fenics-dolfinx", drop `mshr` from the
                          # conda create line below, and re-run.
LOG_DIR="${LOG_DIR:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=================================================================="
echo "ALL-FEM cantilever test on RunPod  |  model: $MODEL"
echo "=================================================================="

# --- [1/4] Ollama -----------------------------------------------------
echo
echo "=== [1/4] Ollama ==="
if ! command -v ollama >/dev/null 2>&1; then
    echo "Installing Ollama..."
    apt-get update -qq
    apt-get install -y -qq curl ca-certificates zstd lshw
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "Ollama already installed."
fi

if ! pgrep -x "ollama" >/dev/null 2>&1; then
    echo "Starting ollama serve in the background..."
    mkdir -p "$LOG_DIR"
    OLLAMA_HOST=0.0.0.0 nohup ollama serve > "$LOG_DIR/ollama.log" 2>&1 &
    sleep 5
else
    echo "ollama serve already running."
fi

echo "Pulling model: $MODEL (skips if already present)"
ollama pull "$MODEL"

# --- [2/4] Conda + FEniCS ----------------------------------------------
echo
echo "=== [2/4] Conda + FEniCS (env: $FENICS_ENV_NAME, package: $FENICS_PACKAGE) ==="
if [ ! -d /opt/conda ]; then
    echo "Installing Miniconda..."
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/conda
else
    echo "Miniconda already installed."
fi

# shellcheck disable=SC1091
# conda's own activation scripts aren't `set -u`-safe (e.g. gcc_linux-64
# references unset SYS_SYSROOT), so relax nounset around sourcing them.
set +u
source /opt/conda/bin/activate

echo "Accepting Anaconda default-channel Terms of Service (idempotent, needed on fresh pods)..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main >/dev/null 2>&1 || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true

if ! conda env list | grep -q "^${FENICS_ENV_NAME} "; then
    echo "Creating conda env '$FENICS_ENV_NAME' with $FENICS_PACKAGE + mshr (this can take a few minutes)..."
    conda create -y -n "$FENICS_ENV_NAME" -c conda-forge "$FENICS_PACKAGE" mshr mpich python=3.11
else
    echo "Conda env '$FENICS_ENV_NAME' already exists."
fi

conda activate "$FENICS_ENV_NAME"
set -u

# --- [3/4] Sanity check harness is present -----------------------------
echo
echo "=== [3/4] Locating test harness ==="
if [ ! -f "$SCRIPT_DIR/test_allfem_cantilever.py" ]; then
    echo "ERROR: test_allfem_cantilever.py not found in $SCRIPT_DIR." >&2
    echo "Upload it next to this script and re-run." >&2
    exit 1
fi
echo "Found $SCRIPT_DIR/test_allfem_cantilever.py"

# --- [4/4] Run the test --------------------------------------------------
echo
echo "=== [4/4] Running the cantilever test ==="
python3 "$SCRIPT_DIR/test_allfem_cantilever.py" --model "$MODEL"
