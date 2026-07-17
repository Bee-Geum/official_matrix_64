#!/usr/bin/env bash
# rerun_round2.sh -- the two things the first re-run could not fix.
#
#  1. multikernelbench x cudaforge, kernelskill  (and pareval x the same)
#     Both crashed on the prompt .txt and still scored PASS/verdicts, because the
#     runner's candidate_files() rglobs the candidate dir for *.py and the drivers
#     kept their scratch trees INSIDE it -- so the crash debris was picked up as a
#     candidate even after harvest() started refusing to emit one. The drivers now
#     put scratch beside the candidate dir instead. Expect these cells to come back
#     with "no candidate was evaluated", which is the honest outcome.
#
#  2. backendbench x ALL at LIMIT=8
#     The task list starts add, mul, sub, div -- all four are elementwise ops with
#     no local test case in the available suites, so LIMIT=1 can never produce a
#     verdict for any agent. The first covered ops (relu, sigmoid, tanh, gelu) are
#     tasks 5-8, so LIMIT=8 is the smallest limit that actually scores this
#     benchmark.
set -uo pipefail

PKG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PKG"
source "$PKG/run_env_h100.sh" >/dev/null 2>&1

RUN_ID="${RUN_ID:-official_7}"
ROUNDS="${ROUNDS:-1}"
EVAL_GPUS="${EVAL_GPUS:-1,2,3}"
SERVER_GPU="${SERVER_GPU:-0}"
export FORCE=1 ROUNDS
export PYTHONPATH="$PKG:${PYTHONPATH:-}"

serve() {
  local model="$1" name="$2" port="$3" util="$4"; shift 4
  echo "  [serve] $name port $port"
  CUDA_VISIBLE_DEVICES="$SERVER_GPU" setsid nohup "$PKG/.venv/bin/python" \
    -m vllm.entrypoints.openai.api_server --model "$model" --served-model-name "$name" \
    --host 127.0.0.1 --port "$port" --gpu-memory-utilization "$util" \
    --max-model-len 32768 "$@" > "$PKG/logs/r2_${name}.out" 2>&1 < /dev/null &
  disown
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && return 0
    sleep 5
  done
  echo "  [serve] ERROR: $name never healthy"; return 1
}
unserve() { pkill -f "port $1" 2>/dev/null; sleep 10; }

cells() {  # cells <agents> <benchmarks> <limit>
  python3 "$PKG/official_all_matrix_v1.py" run --root "$PKG" --run-id "$RUN_ID" \
    --benchmarks "$2" --agents "$1" --limit "$3" --rounds "$ROUNDS" \
    --gpus "$EVAL_GPUS" --cell-workers 3 --workers-per-gpu 1
  echo "  rc=$?"
}

export OPENAI_BASE_URL="http://127.0.0.1:8000/v1" EVAL_MODEL="qwen14b"
serve "Qwen/Qwen2.5-Coder-14B-Instruct" qwen14b 8000 0.85 || exit 1
echo "===== 1. debris cells, now that scratch lives outside the candidate dir ====="
cells "cudaforge,kernelskill" "multikernelbench,pareval" 1
echo "===== 2. backendbench LIMIT=8 (qwen-backed) ====="
cells "cudaforge,autokernel,kernelskill" "backendbench" 8
unserve 8000

echo "===== 2. backendbench LIMIT=8 (model-only) ====="
serve "ai9stars/AutoTriton" autotriton8b 8001 0.85 && cells "autotriton" "backendbench" 8
unserve 8001
serve "facebook/KernelLLM" kernelllm8b 8002 0.85 && cells "kernelllm" "backendbench" 8
unserve 8002
serve "Multilingual-Multimodal-NLP/IndustrialCoder" incoder32b 8003 0.90 --trust-remote-code \
  && cells "incoder32b" "backendbench" 8
unserve 8003
if curl -sf http://127.0.0.1:10907/health >/dev/null 2>&1; then
  serve "hkust-nlp/drkernel-14b" drkernel14b 8004 0.85 && cells "drkernel" "backendbench" 8
  unserve 8004
fi

echo; echo "===== classify ====="
python3 "$PKG/classify_official_7.py" --run-root "$PKG/results/all_official_matrix_v1/$RUN_ID"
