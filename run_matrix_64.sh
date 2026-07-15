#!/usr/bin/env bash
# run_matrix_64.sh  --  SELF-CONTAINED portable runner
#
# 11 agents x 8 official benchmarks = 88 cells, each scored by the benchmark's
# OFFICIAL oracle. Everything needed (agent drivers, benchmark repos + oracles,
# evaluators, registry, task lists) is bundled inside THIS directory, so it runs
# on any machine with: a CUDA GPU + nvcc, Python 3.10+, the deps in
# requirements.txt, and an OpenAI-compatible LLM endpoint (see LLM_BASE_URL).
#
# The only thing NOT bundled is the LLM weights (too large). Point LLM_BASE_URL
# at any OpenAI-compatible /v1 endpoint, or run the bundled hf_openai_server.py
# with a model of your choice.
#
# Usage:
#   ./run_matrix_64.sh                          # full 64-cell run, 1 task/cell
#   LLM_BASE_URL=http://HOST:8000/v1 ./run_matrix_64.sh
#   AGENTS=cudaforge,geak BENCHMARKS=pareval LIMIT=3 ./run_matrix_64.sh
#   nohup ./run_matrix_64.sh > run_matrix_64.out 2>&1 &
#
# Multi-GPU (default: every visible GPU, one cell per GPU):
#   GPUS=0,1,2,3 ./run_matrix_64.sh             # pick the GPUs explicitly
#   GPUS=0 ./run_matrix_64.sh                   # original single-GPU serial run
#   WORKERS_PER_GPU=2 ./run_matrix_64.sh        # 2 cells/GPU: faster, but the official
#                                               #   timings contend -- correctness only
#   AUTO_START_SERVER=1 SERVER_GPU=0 ./run_matrix_64.sh   # LLM on GPU 0, eval on the rest
set -uo pipefail

# --- this package IS the ROOT (fully self-contained) -------------------------
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT="$PKG"
RUNNER="$PKG/official_all_matrix_v1.py"
RUN_ID="${RUN_ID:-official_matrix_64}"
RUN_ROOT="$PKG/results/all_official_matrix_v1/$RUN_ID"

# --- matrix selection --------------------------------------------------------
BENCHMARKS="${BENCHMARKS:-all}"        # 'all' = the 8 official benchmarks
AGENTS="${AGENTS:-all}"                # 'all' = the 11 registry agents
LIMIT="${LIMIT:-1}"                    # tasks per cell

# --- LLM endpoint (bring your own OpenAI-compatible server) ------------------
export PORT="${PORT:-8000}"
export OPENAI_BASE_URL="${LLM_BASE_URL:-${OPENAI_BASE_URL:-http://127.0.0.1:${PORT}/v1}}"
export EVAL_MODEL="${EVAL_MODEL:-qwen14b}"          # model name sent to the endpoint
AUTO_START_SERVER="${AUTO_START_SERVER:-0}"        # 1 = try to start bundled server
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-Coder-14B-Instruct}"

# --- run knobs ---------------------------------------------------------------
export FORCE="${FORCE:-1}"
export MAX_CANDIDATES="${MAX_CANDIDATES:-1}"
export CELL_ATTEMPTS="${CELL_ATTEMPTS:-1}"
export GENERATION_TIMEOUT="${GENERATION_TIMEOUT:-600}"
export EVAL_TIMEOUT="${EVAL_TIMEOUT:-900}"
export GPU="${GPU:-0}"
export PYTHONPATH="$PKG:${PYTHONPATH:-}"
mkdir -p "$PKG/logs"
cd "$PKG"

log() { echo; echo "===== $* ====="; }

# --- multi-GPU: one cell per GPU, all GPUs busy ------------------------------
# GPUS=all uses every visible GPU. GPUS=0 restores the original single-GPU run.
# WORKERS_PER_GPU>1 finishes sooner but co-locates cells, so the official timing
# numbers contend with each other -- correctness stays valid, performance does not.
GPUS="${GPUS:-all}"
export WORKERS_PER_GPU="${WORKERS_PER_GPU:-1}"
ALL_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)"
ALL_GPUS="${ALL_GPUS:-0}"
[ "$GPUS" = "all" ] && GPUS="$ALL_GPUS"

# The bundled LLM server needs a GPU of its own: sharing one with an eval cell
# costs VRAM and perturbs that cell's timings. Dedicate SERVER_GPU and evaluate
# on the rest. With an external LLM_BASE_URL no GPU is reserved.
SERVER_GPU="${SERVER_GPU:-0}"
DEDICATE_SERVER_GPU="${DEDICATE_SERVER_GPU:-1}"
if [ "$AUTO_START_SERVER" = "1" ] && [ "$DEDICATE_SERVER_GPU" = "1" ]; then
  REST="$(echo "$GPUS" | tr ',' '\n' | grep -vx "$SERVER_GPU" | paste -sd, -)"
  if [ -n "$REST" ]; then
    echo "LLM server -> GPU $SERVER_GPU (dedicated); eval -> GPUs $REST"
    GPUS="$REST"
  else
    echo "WARNING: only GPU $SERVER_GPU exists; LLM server and eval will share it (timings will contend)"
  fi
fi
export GPUS
# GPU is only the single-GPU fallback now; --gpus wins. Keep it inside the eval set.
export GPU="$(echo "$GPUS" | cut -d, -f1)"
NGPU="$(echo "$GPUS" | tr ',' '\n' | grep -c .)"
export CELL_WORKERS="${CELL_WORKERS:-$((NGPU * WORKERS_PER_GPU))}"

# Blackwell=12.0, Hopper=9.0. Unpinned, the kb eval compiles every arch in
# torch.cuda.get_arch_list() (6 passes, minutes per kernel) and blows EVAL_TIMEOUT.
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "$(echo "$GPUS" | cut -d, -f1)" 2>/dev/null | head -1 | tr -d ' ')"
  [ -n "$CAP" ] && export TORCH_CUDA_ARCH_LIST="$CAP"
fi
# nvcc build jobs, split across concurrently compiling cells so they don't thrash.
export MAX_JOBS="${MAX_JOBS:-$(( $(nproc) / CELL_WORKERS > 0 ? $(nproc) / CELL_WORKERS : 1 ))}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "GPUs=[$GPUS] cell_workers=$CELL_WORKERS arch=${TORCH_CUDA_ARCH_LIST:-auto} MAX_JOBS=$MAX_JOBS"

# --- 1. LLM endpoint ---------------------------------------------------------
check_llm() { curl -sf "${OPENAI_BASE_URL%/v1}/health" >/dev/null 2>&1 || \
              curl -sf "$OPENAI_BASE_URL/models" >/dev/null 2>&1; }
if check_llm; then
  echo "LLM endpoint OK: $OPENAI_BASE_URL"
elif [ "$AUTO_START_SERVER" = "1" ] && [ -f "$PKG/hf_openai_server.py" ]; then
  log "Starting bundled LLM server ($MODEL_ID)"
  CUDA_VISIBLE_DEVICES="$SERVER_GPU" nohup python3 "$PKG/hf_openai_server.py" \
    --model_id "$MODEL_ID" --model_alias "$EVAL_MODEL" --host 127.0.0.1 \
    --port "$PORT" --max_new_tokens 1024 --temperature 0.2 --top_p 0.95 \
    > "$PKG/logs/hf_server.out" 2>&1 &
  for _ in $(seq 1 180); do check_llm && break; sleep 2; done
  check_llm || { echo "ERROR: bundled server did not become healthy"; exit 1; }
else
  echo "ERROR: no LLM endpoint at $OPENAI_BASE_URL"
  echo "  Set LLM_BASE_URL=http://HOST:PORT/v1 to an OpenAI-compatible server,"
  echo "  or AUTO_START_SERVER=1 (with a local GPU + model) to launch the bundled one."
  exit 1
fi

# --- 2. prepare tasks from bundled benchmark repos (local paths) -------------
if [ ! -d "$RUN_ROOT/prepared/tasks" ] || [ "${REPREPARE:-0}" = "1" ]; then
  log "Preparing official tasks from bundled third_party (writes local-path manifests)"
  python3 "$RUNNER" prepare --root "$PKG" --run-id "$RUN_ID" --benchmarks "$BENCHMARKS" \
    || { echo "ERROR: prepare failed"; exit 1; }
fi

# --- 3. run the matrix -------------------------------------------------------
log "RUN agents=[$AGENTS] x benchmarks=[$BENCHMARKS] limit=$LIMIT (ROOT=$PKG)"
python3 "$RUNNER" run --root "$PKG" --run-id "$RUN_ID" \
  --benchmarks "$BENCHMARKS" --agents "$AGENTS" --limit "$LIMIT" \
  --gpus "$GPUS" --cell-workers "$CELL_WORKERS" --workers-per-gpu "$WORKERS_PER_GPU"
echo "runner rc=$? (3 = ran but not every cell scored 'correct' -- a normal benchmark outcome)"

# --- 4. summarize ------------------------------------------------------------
log "Summarize -> $PKG/results"
python3 "$PKG/summarize_matrix_64.py" --run-root "$RUN_ROOT" --out "$PKG/results"
echo
echo "Raw per-cell data : $RUN_ROOT/cells/<benchmark>/<agent>/cell_result.json"
echo "Summary tables    : $PKG/results/"
