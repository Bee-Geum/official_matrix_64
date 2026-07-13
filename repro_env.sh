#!/usr/bin/env bash
# repro_env.sh -- environment for FAITHFUL agent reproduction on this RTX PRO 6000 (Blackwell) box.
#
# Source this before running any real agent framework (CudaForge/autokernel/...) or the
# official kb oracle (telemetry/instrumented_final_eval.py). It encodes the fixes discovered
# while getting CudaForge to run end-to-end against a local Qwen2.5-Coder-14B vLLM endpoint.
#
#   source repro_env.sh
#
# Findings baked in below:
#  1. Blackwell RTX PRO 6000 = compute capability 12.0 (sm_120). torch 2.11+cu130 supports it,
#     but WITHOUT TORCH_CUDA_ARCH_LIST the kb eval compiles every arch in torch.cuda.get_arch_list()
#     (sm_75..sm_120 = 6 passes, ~3 min+ per kernel) and blows the eval timeout. Pinning to 12.0
#     drops a cold load_inline compile to ~33 s.
#  2. A kb eval killed by timeout (SIGKILL) leaves a stale torch FileBaton lock file named `lock`
#     (no extension) inside ~/.cache/torch_extensions/<pyver>/<kernel_name>/. The next process that
#     load_inline()s the same kernel name blocks forever (STAT=Sl, ~0% CPU). Use a FRESH
#     TORCH_EXTENSIONS_DIR per run so a poisoned cache never carries over.
#  3. vLLM and the kernel eval share ONE GPU. Serve the LLM at gpu-memory-utilization 0.4
#     (~38 GB of 96 GB) so the eval has ~57 GB for its (sometimes multi-GB) tensors.
#  4. vLLM 0.20.2 aborts on a flashinfer/flashinfer-cubin version mismatch; bypass with
#     FLASHINFER_DISABLE_VERSION_CHECK=1.

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# LLM endpoint (bring-your-own vLLM server, see start_llm_server below)
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8000/v1}"
export EVAL_MODEL="${EVAL_MODEL:-qwen14b}"

# Give every run its own extensions dir unless the caller already set one.
fresh_ext_dir() {
  local base="${1:-/home/bi_geum/official_matrix_64/results/repro/_extdir}"
  local d="${base}_$$"
  rm -rf "$d"; mkdir -p "$d"
  export TORCH_EXTENSIONS_DIR="$d"
  echo "$d"
}

# Start the shared Qwen14B vLLM server (0.4 util). Usage: start_llm_server &>/dev/null; wait_llm
start_llm_server() {
  local log="${1:-/home/bi_geum/official_matrix_64/logs/vllm_qwen14b.out}"
  python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-Coder-14B-Instruct \
    --served-model-name qwen14b \
    --host 127.0.0.1 --port 8000 \
    --max-model-len 16384 --gpu-memory-utilization 0.4 > "$log" 2>&1
}

wait_llm() {
  local i
  for i in $(seq 1 180); do
    curl -sf -m 3 http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q qwen14b && { echo "LLM ready"; return 0; }
    sleep 5
  done
  echo "LLM not ready"; return 1
}
