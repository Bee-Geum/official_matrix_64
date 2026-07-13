#!/usr/bin/env bash
# run_matrix_88.sh  --  SELF-CONTAINED portable runner
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
#   ./run_matrix_88.sh                          # full 88-cell run, 1 task/cell
#   LLM_BASE_URL=http://HOST:8000/v1 ./run_matrix_88.sh
#   AGENTS=cudaforge,geak BENCHMARKS=pareval LIMIT=3 ./run_matrix_88.sh
#   nohup ./run_matrix_88.sh > run_matrix_88.out 2>&1 &
set -uo pipefail

# --- this package IS the ROOT (fully self-contained) -------------------------
PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROOT="$PKG"
RUNNER="$PKG/official_all_matrix_v1.py"
RUN_ID="${RUN_ID:-official_matrix_88}"
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

# --- 1. LLM endpoint ---------------------------------------------------------
check_llm() { curl -sf "${OPENAI_BASE_URL%/v1}/health" >/dev/null 2>&1 || \
              curl -sf "$OPENAI_BASE_URL/models" >/dev/null 2>&1; }
if check_llm; then
  echo "LLM endpoint OK: $OPENAI_BASE_URL"
elif [ "$AUTO_START_SERVER" = "1" ] && [ -f "$PKG/hf_openai_server.py" ]; then
  log "Starting bundled LLM server ($MODEL_ID)"
  CUDA_VISIBLE_DEVICES="$GPU" nohup python3 "$PKG/hf_openai_server.py" \
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
  --benchmarks "$BENCHMARKS" --agents "$AGENTS" --limit "$LIMIT"
echo "runner rc=$? (3 = ran but not every cell scored 'correct' -- a normal benchmark outcome)"

# --- 4. summarize ------------------------------------------------------------
log "Summarize -> $PKG/results"
python3 "$PKG/summarize_matrix_88.py" --run-root "$RUN_ROOT" --out "$PKG/results"
echo
echo "Raw per-cell data : $RUN_ROOT/cells/<benchmark>/<agent>/cell_result.json"
echo "Summary tables    : $PKG/results/"
