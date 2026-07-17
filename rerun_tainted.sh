#!/usr/bin/env bash
# rerun_tainted.sh -- re-run only the cells whose results are known-bad.
#
# The first official_7 pass produced cells that are not benchmark results:
#
#  1. backendbench x ALL       -- BackendBench's official CLI died at import
#                                 (tenacity -> expecttest -> pyarrow, all declared
#                                 in its pyproject but missing from requirements.txt).
#                                 official_eval=0 for a reason unrelated to any agent.
#  2. autokernel x kernelbench, robust_kbench
#                              -- AutoKernel's playbook tells the model to
#                                 `from kernels.cuda._compile import compile_cuda`,
#                                 which only resolves inside AutoKernel's tree. Its own
#                                 bench_kb said correct=True; the official oracle got
#                                 ModuleNotFoundError. Recorded as an agent failure; it
#                                 was a wiring bug. Fixed with a path shim.
#  3. cudaforge/kernelskill x multikernelbench, pareval, sol_execbench
#                              -- both crashed on the prompt .txt (they need a
#                                 KernelBench model .py) and the drivers' rglob
#                                 fallback harvested the crash debris, which the
#                                 oracle then scored -- producing two false PASSes.
#                                 Both drivers now refuse. Expect these to come back
#                                 with no candidate, which is the honest outcome.
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
    --max-model-len 32768 "$@" > "$PKG/logs/rerun_${name}.out" 2>&1 < /dev/null &
  disown
  for _ in $(seq 1 240); do
    curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && return 0
    sleep 5
  done
  echo "  [serve] ERROR: $name never healthy"; return 1
}
unserve() { pkill -f "port $1" 2>/dev/null; sleep 10; }

cells() {  # cells <agents> <benchmarks>
  python3 "$PKG/official_all_matrix_v1.py" run --root "$PKG" --run-id "$RUN_ID" \
    --benchmarks "$2" --agents "$1" --limit 1 --rounds "$ROUNDS" \
    --gpus "$EVAL_GPUS" --cell-workers 3 --workers-per-gpu 1
  echo "  rc=$?"
}

export OPENAI_BASE_URL="http://127.0.0.1:8000/v1" EVAL_MODEL="qwen14b"
serve "Qwen/Qwen2.5-Coder-14B-Instruct" qwen14b 8000 0.85 || exit 1
echo "===== autokernel kb cells (path shim) ====="
cells "autokernel" "kernelbench,robust_kbench"
echo "===== cudaforge/kernelskill non-KB cells (debris refusal) ====="
cells "cudaforge,kernelskill" "multikernelbench,pareval,sol_execbench"
echo "===== backendbench: qwen-backed agents (deps) ====="
cells "cudaforge,autokernel,kernelskill" "backendbench"
unserve 8000

echo "===== backendbench: model-only agents (deps) ====="
serve "ai9stars/AutoTriton" autotriton8b 8001 0.85 && cells "autotriton" "backendbench"
unserve 8001
serve "facebook/KernelLLM" kernelllm8b 8002 0.85 && cells "kernelllm" "backendbench"
unserve 8002
serve "Multilingual-Multimodal-NLP/IndustrialCoder" incoder32b 8003 0.90 --trust-remote-code \
  && cells "incoder32b" "backendbench"
unserve 8003

if curl -sf http://127.0.0.1:10907/health >/dev/null 2>&1; then
  serve "hkust-nlp/drkernel-14b" drkernel14b 8004 0.85 && cells "drkernel" "backendbench"
  unserve 8004
else
  echo "===== drkernel backendbench SKIPPED: KernelGYM down ====="
fi

echo; echo "===== classify ====="
python3 "$PKG/classify_official_7.py" --run-root "$PKG/results/all_official_matrix_v1/$RUN_ID"
