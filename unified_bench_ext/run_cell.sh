#!/usr/bin/env bash
# run_cell.sh -- run ONE (agent, benchmark) cell over a task list.
#
# Reproduces the verified unified_bench pipeline (driver -> candidates ->
# instrumented_final_eval -> summary.json) but with the benchmark name correctly
# embedded in the run-dir name so the matrix collector never mixes benchmarks, and
# with per-cell timing/telemetry that the original archive was missing.
#
# Driver + evaluator CLIs are taken from registry/agents.csv and were observed in the
# A100 official5 telemetry for cudaforge / autokernel / autotriton.
#
# Usage:
#   ./unified_bench_ext/run_cell.sh <agent> <benchmark> [task_list] \
#        [rounds] [repeat] [temp] [timeout] [limit]
#
#   task_list  defaults to the benchmark's prepared list under task_lists/
#   limit      cap number of tasks (e.g. 1 or 5 for smoke runs); 0 = all
#
# Example:
#   ./unified_bench_ext/run_cell.sh cudaforge kernelbench '' 3 1 0.2 1800 5
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
ensure_layout

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GPU_NAME="${GPU_NAME:-RTX PRO 6000}"
export UB_TARGET_GPU_LABEL="${UB_TARGET_GPU_LABEL:-RTX PRO 6000}"

AGENT="${1:?agent}"; BENCH="${2:?benchmark}"
TASK_LIST="${3:-}"
ROUNDS="${4:-1}"; REPEAT="${5:-1}"; TEMP="${6:-0.2}"
TIMEOUT="${7:-1800}"; LIMIT="${8:-0}"
SEED="${SEED:-20260611}"
GPU_JSON="${GPU_JSON:-h100.json}"   # RTX PRO 6000 has no CUDA-L1 JSON; h100 is cross-hardware replay

# --- compatibility gate ---
read -r STATUS REASON < <(cell_status "$AGENT" "$BENCH" | tr '\t' ' ' | sed 's/ /\t/' | awk -F'\t' '{print $1" "$2}')
if [ "$STATUS" != "RUN" ]; then
  echo "[cell] SKIP $AGENT x $BENCH -- $REASON"
  exit 0
fi

MODEL="$(agent_field "$AGENT" model)"
DRIVER="$(agent_field "$AGENT" driver)"
GEN_ARGS="$(agent_field "$AGENT" gen_args)"
GLOB="$(agent_field "$AGENT" glob)"
EVAL_MODE="$(bench_field "$BENCH" eval_mode)"
TASK_FORMAT="$(bench_field "$BENCH" task_format)"

# default task list = the prepared list for this benchmark
if [ -z "$TASK_LIST" ]; then
  for cand in "$EXT/task_lists/${BENCH}.txt" "$EXT/task_lists/${BENCH}_cuda.txt" \
              "$EXT/task_lists/${BENCH}_l12.txt" "$EXT/task_lists/${BENCH}_smoke.txt" \
              "$EXT/task_lists/${BENCH}_all250.txt"; do
    [ -f "$cand" ] && { TASK_LIST="$cand"; break; }
  done
fi
[ -f "$TASK_LIST" ] || die "no task list for $BENCH (run benchmarks/$BENCH/prepare.sh). looked under $EXT/task_lists/"

SUBSET="$(basename "$TASK_LIST" .txt | sed "s/^${BENCH}_*//")"; SUBSET="${SUBSET:-full}"
RUN="${AGENT}_${MODEL}_${BENCH}_${SUBSET}_round${ROUNDS}_repeat${REPEAT}_temp${TEMP}"
echo "[cell] RUN $AGENT x $BENCH  (eval=$EVAL_MODE fmt=$TASK_FORMAT)  -> runs/$RUN"

source_env

# determine driver-side shim from task format
SHIM_KIND=""
case "$TASK_FORMAT" in
  op_fill) SHIM_KIND="op" ;;
  triton_standalone) SHIM_KIND="triton" ;;
  parallel_prompt) SHIM_KIND="parallel" ;;
esac

mapfile -t TASKS < <(grep -v '^[[:space:]]*$' "$TASK_LIST")
[ "$LIMIT" -gt 0 ] && TASKS=("${TASKS[@]:0:$LIMIT}")
echo "[cell] ${#TASKS[@]} tasks x $REPEAT repeats"

for task_rel in "${TASKS[@]}"; do
  stem="$(basename "$task_rel")"
  stem="${stem%.py}"
  for rep in $(seq 0 $((REPEAT - 1))); do
    work="$UB_ROOT/runs/$RUN/${stem}__rep${rep}"
    cand="$work/candidates"; feval="$work/final_eval"; tele="$work/telemetry"
    mkdir -p "$cand" "$feval" "$tele"

    # resolve the task file the DRIVER will consume (real task, or a shim)
    if [ -n "$SHIM_KIND" ]; then
      task_abs="$work/shim_task.py"
      python3 "$EXT/adapters/format_shim.py" --kind "$SHIM_KIND" --task "$stem" --out "$task_abs" \
        >> "$tele/shim.log" 2>&1 || { echo "[cell] shim failed for $stem"; continue; }
    else
      # task lists may hold either absolute paths (regenerated official lists)
      # or paths relative to UB_ROOT (older lists). Do not double-prefix ROOT.
      case "$task_rel" in
        /*) task_abs="$task_rel" ;;
        *)  task_abs="$UB_ROOT/$task_rel" ;;
      esac
    fi

    # substitute the driver arg template
    args="${GEN_ARGS//\{TASK\}/$task_abs}"
    args="${args//\{CAND\}/$cand}"
    args="${args//\{ROUNDS\}/$ROUNDS}"
    args="${args//\{SEED\}/$SEED}"
    args="${args//\{TEMP\}/$TEMP}"
    args="${args//\{GPU_JSON\}/$GPU_JSON}"

    # --- generation (timed, time-limited) ---
    gen_start=$(date +%s)
    # shellcheck disable=SC2086
    /usr/bin/time -v -o "$tele/generation_resource.txt" \
      timeout --kill-after=60s "$TIMEOUT" \
      env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" GPU_NAME="$GPU_NAME" python3 "$UB_ROOT/$DRIVER" $args > "$tele/generation.out" 2>&1 || true
    gen_end=$(date +%s)

    # --- evaluation -> canonical summary.json ---
    case "$EVAL_MODE" in
      kb_instrumented)
        /usr/bin/time -v -o "$tele/eval_resource.txt" \
          timeout --kill-after=60s "$TIMEOUT" \
          env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" GPU_NAME="$GPU_NAME" python3 "$UB_ROOT/telemetry/instrumented_final_eval.py" \
            --task "$task_abs" --cand_dir "$cand" --glob "$GLOB" \
            --task_work_dir "$feval" > "$tele/eval.out" 2>&1 || true
        ;;
      backendbench)
        # pick the first candidate file as the op implementation
        cfile="$(ls "$cand"/$GLOB 2>/dev/null | head -1 || true)"
        python3 "$EXT/adapters/backendbench_adapter.py" \
          --bb-root "$UB_ROOT/third_party/BackendBench" --op "$stem" \
          --suite "${BACKENDBENCH_SUITE:-smoke}" --backend "${BACKENDBENCH_BACKEND:-aten}" \
          --cand-file "${cfile:-/dev/null}" --task "$stem" \
          --out "$feval/summary.json" > "$tele/eval.out" 2>&1 || true
        ;;
      native)
        nat="$EXT/adapters/${BENCH}_native.py"
        if [ -f "$nat" ]; then
          python3 "$nat" --bench-root "$UB_ROOT/third_party/$(bench_field "$BENCH" repo | sed 's#.*/##')" \
            --task "$stem" --cand-dir "$cand" --glob "$GLOB" \
            --out "$feval/summary.json" > "$tele/eval.out" 2>&1 || true
        else
          msg="[cell] native adapter missing: copy adapters/native_eval_template.py to ${nat} and implement run_native"
          echo "$msg" | tee "$tele/eval.out"
        fi
        ;;
    esac

    # --- per-cell telemetry the original archive lacked ---
    UB_OUT="$tele/cell_meta.json" UB_RUN="$RUN" UB_AGENT="$AGENT" UB_BENCH="$BENCH" \
    UB_TASK="$task_rel" UB_REP="$rep" UB_GEN="$((gen_end - gen_start))" \
    UB_CAND="$cand" UB_GLOB="$GLOB" python3 - <<'PY'
import json, os, glob
cand = os.environ["UB_CAND"]; g = os.environ["UB_GLOB"]
# count any LLM-io the driver may have left (cudaforge/autotriton write candidates/llm_io/*)
llm_io = glob.glob(os.path.join(cand, "llm_io", "*")) + glob.glob(os.path.join(cand, "..", "**", "*_reply*.txt"), recursive=True)
meta = {
    "run": os.environ["UB_RUN"], "agent": os.environ["UB_AGENT"],
    "benchmark": os.environ["UB_BENCH"], "task": os.environ["UB_TASK"],
    "repeat": int(os.environ["UB_REP"]),
    "generation_seconds": int(os.environ["UB_GEN"]),
    "n_candidates": len(glob.glob(os.path.join(cand, g))),
    "n_llm_io_files": len(llm_io),
}
json.dump(meta, open(os.environ["UB_OUT"], "w"), indent=2)
PY
    echo "[cell]   $stem rep$rep done (gen $((gen_end - gen_start))s)"
  done
done

echo "[cell] finished $RUN"
