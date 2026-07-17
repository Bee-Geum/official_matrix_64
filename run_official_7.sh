#!/usr/bin/env bash
# run_official_7.sh -- the 7 wired agents x 8 official benchmarks.
#
# Every agent now runs as itself (upstream repo or released weights), so each
# needs its own backbone. They do not fit on this box at once (Qwen 14B +
# AutoTriton 8B + KernelLLM 8B + IndustrialCoder 32B + Dr.Kernel 14B ~= 158 GB
# of weights), and co-locating a server with an eval cell would perturb that
# cell's timings anyway. So: serve one phase's backbone on GPU 0, evaluate on
# GPUs 1-3, tear down, move to the next phase. Cells accumulate under one RUN_ID.
#
#   ROUNDS=1 ./run_official_7.sh     # mechanical "does every cell run" check
#   ROUNDS=3 ./run_official_7.sh     # exercises NCU / multi-turn (much slower)
set -uo pipefail

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PKG"
source "$PKG/run_env_h100.sh" >/dev/null 2>&1

RUN_ID="${RUN_ID:-official_7}"
ROUNDS="${ROUNDS:-1}"
LIMIT="${LIMIT:-1}"
BENCHMARKS="${BENCHMARKS:-all}"
EVAL_GPUS="${EVAL_GPUS:-1,2,3}"
SERVER_GPU="${SERVER_GPU:-0}"
LOGS="$PKG/logs"
mkdir -p "$LOGS"

export ROUNDS LIMIT
export PYTHONPATH="$PKG:${PYTHONPATH:-}"

serve() {  # serve <hf_model> <served_name> <port> <util> [extra...]
  local model="$1" name="$2" port="$3" util="$4"; shift 4
  echo "  [serve] $name on GPU $SERVER_GPU port $port"
  CUDA_VISIBLE_DEVICES="$SERVER_GPU" setsid nohup "$PKG/.venv/bin/python" \
    -m vllm.entrypoints.openai.api_server --model "$model" --served-model-name "$name" \
    --host 127.0.0.1 --port "$port" --gpu-memory-utilization "$util" \
    --max-model-len 32768 "$@" > "$LOGS/serve_${name}.out" 2>&1 < /dev/null &
  disown
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && { echo "  [serve] $name ready"; return 0; }
    sleep 5
  done
  echo "  [serve] ERROR: $name never became healthy"; tail -5 "$LOGS/serve_${name}.out"; return 1
}

unserve() {  # unserve <port>
  pkill -f "port $1" 2>/dev/null
  for _ in $(seq 1 24); do
    curl -sf "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1 || break
    sleep 5
  done
  sleep 5
}

phase() {  # phase <label> <agents-csv>
  echo; echo "===== PHASE $1: agents=$2 rounds=$ROUNDS ====="
  python3 "$PKG/official_all_matrix_v1.py" run --root "$PKG" --run-id "$RUN_ID" \
    --benchmarks "$BENCHMARKS" --agents "$2" --limit "$LIMIT" --rounds "$ROUNDS" \
    --gpus "$EVAL_GPUS" --cell-workers 3 --workers-per-gpu 1
  echo "  phase $1 rc=$?  (3 = ran, not all correct -- a normal benchmark outcome)"
}

# --- prepare task manifests once -------------------------------------------
if [ ! -d "$PKG/results/all_official_matrix_v1/$RUN_ID/prepared/tasks" ]; then
  echo "===== prepare ====="
  python3 "$PKG/official_all_matrix_v1.py" prepare --root "$PKG" --run-id "$RUN_ID" \
    --benchmarks "$BENCHMARKS" || { echo "prepare failed"; exit 1; }
fi

# --- A: the three live agents share the Qwen backbone -----------------------
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1" EVAL_MODEL="qwen14b"
serve "Qwen/Qwen2.5-Coder-14B-Instruct" qwen14b 8000 0.85 && \
  phase A "cudaforge,autokernel,kernelskill"
unserve 8000

# --- B/C/D: model-only agents, each is its own artifact ----------------------
serve "ai9stars/AutoTriton" autotriton8b 8001 0.85 && phase B "autotriton"
unserve 8001

serve "facebook/KernelLLM" kernelllm8b 8002 0.85 && phase C "kernelllm"
unserve 8002

serve "Multilingual-Multimodal-NLP/IndustrialCoder" incoder32b 8003 0.90 \
  --trust-remote-code && phase D "incoder32b"
unserve 8003

# --- E: RL model + KernelGYM ------------------------------------------------
# The driver refuses to run if the gym is not healthy, so check it here too.
if curl -sf http://127.0.0.1:10907/health >/dev/null 2>&1; then
  serve "hkust-nlp/drkernel-14b" drkernel14b 8004 0.85 && phase E "drkernel"
  unserve 8004
else
  echo; echo "===== PHASE E SKIPPED: KernelGYM not healthy at :10907 ====="
  echo "  drkernel needs its gym; start it before running this phase."
fi

# --- summarize ---------------------------------------------------------------
echo; echo "===== summarize ====="
python3 "$PKG/summarize_matrix_64.py" \
  --run-root "$PKG/results/all_official_matrix_v1/$RUN_ID" --out "$PKG/results_official_7"
echo "Raw cells: $PKG/results/all_official_matrix_v1/$RUN_ID/cells/<benchmark>/<agent>/cell_result.json"
