#!/usr/bin/env bash
# run_env_h100.sh -- environment for THIS box (4x H100 80GB, sm_90, CUDA 12.8).
#
# The repo ships repro_env.sh / use_rtx6000_cuda_env.sh for an RTX PRO 6000
# (Blackwell, sm_120) with a preinstalled toolchain. This box is different:
#   - H100 = compute capability 9.0, not 12.0
#   - no system torch/triton -> everything lives in .venv
#   - nvcc is at /usr/local/cuda-12.8 but is NOT on PATH by default
#
# run_matrix_64.sh calls bare `python3`, so .venv/bin MUST come first on PATH
# or the runner picks up the system python that has no torch.
#
#   source run_env_h100.sh

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- venv first on PATH: run_matrix_64.sh + every driver shell out to `python3`
export VIRTUAL_ENV="$PKG/.venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# --- CUDA toolkit (nvcc is required: kernelbench/pareval/backendbench compile)
export CUDA_HOME=/usr/local/cuda
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# --- H100 = sm_90. Unpinned, the kb oracle compiles every arch in
#     torch.cuda.get_arch_list() (many passes, minutes per kernel) and blows
#     EVAL_TIMEOUT. run_matrix_64.sh auto-detects compute_cap, but pin it so a
#     direct runner invocation gets it too.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- LLM endpoint: vLLM on GPU 0, eval on GPUs 1,2,3
export LLM_BASE_URL="${LLM_BASE_URL:-http://127.0.0.1:8000/v1}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-$LLM_BASE_URL}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export EVAL_MODEL="${EVAL_MODEL:-qwen14b}"

export PYTHONPATH="$PKG:${PYTHONPATH:-}"

echo "env: python=$(command -v python3) nvcc=$(command -v nvcc) arch=$TORCH_CUDA_ARCH_LIST"
