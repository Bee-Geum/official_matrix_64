#!/usr/bin/env bash
# run_matrix.sh -- run many (agent, benchmark) cells in one pass.
#
# Walks the compatibility matrix (registry/compat.py --runnable) and calls run_cell.sh
# for every RUN cell, optionally filtered by agent and/or benchmark. SKIP cells are
# listed with their reason and not run. This is the matrix analogue of the harness's
# run_all_with_server.sh, but benchmark-aware.
#
# Usage:
#   ./unified_bench_ext/run_matrix.sh [options]
#
# Options (all optional):
#   --agent A[,B...]       only these agents
#   --benchmark X[,Y...]   only these benchmarks
#   --clean-only           only clean cells (ModelNew + A100, no shim, no partial)
#   --rounds N             generation rounds            (default 1)
#   --repeat N             repeats per task             (default 1)
#   --temp T               sampling temperature         (default 0.2)
#   --timeout S            per-task wall limit, seconds (default 1800)
#   --limit N              cap tasks per cell, 0 = all  (default 0; use small for smoke)
#   --dry-run              print the cells that would run, do nothing
#
# Examples:
#   ./unified_bench_ext/run_matrix.sh --dry-run
#   ./unified_bench_ext/run_matrix.sh --benchmark kernelbench --limit 5
#   ./unified_bench_ext/run_matrix.sh --agent cudaforge,autotriton --clean-only
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"
ensure_layout

AGENT_FILTER=""; BENCH_FILTER=""; CLEAN_ONLY=0; DRY=0
ROUNDS=1; REPEAT=1; TEMP=0.2; TIMEOUT=1800; LIMIT=0

while [ $# -gt 0 ]; do
  case "$1" in
    --agent)      AGENT_FILTER="$2"; shift 2 ;;
    --benchmark)  BENCH_FILTER="$2"; shift 2 ;;
    --clean-only) CLEAN_ONLY=1; shift ;;
    --rounds)     ROUNDS="$2"; shift 2 ;;
    --repeat)     REPEAT="$2"; shift 2 ;;
    --temp)       TEMP="$2"; shift 2 ;;
    --timeout)    TIMEOUT="$2"; shift 2 ;;
    --limit)      LIMIT="$2"; shift 2 ;;
    --dry-run)    DRY=1; shift ;;
    *) die "unknown option: $1" ;;
  esac
done

in_filter() {  # in_filter <value> <csv-filter>   (empty filter = match all)
  local v="$1" f="$2"
  [ -z "$f" ] && return 0
  local IFS=','; for x in $f; do [ "$x" = "$v" ] && return 0; done
  return 1
}

# pull the runnable cell list from the single source of truth (compat.py)
COMPAT_ARGS="--runnable"
[ "$CLEAN_ONLY" -eq 1 ] && COMPAT_ARGS="$COMPAT_ARGS --clean-only"
mapfile -t CELLS < <(UB_EXT_DIR="$EXT" python3 "$EXT/registry/compat.py" $COMPAT_ARGS)

log "matrix run: ${#CELLS[@]} runnable cells before filtering"
declare -i n_run=0 n_skip_filter=0

for line in "${CELLS[@]}"; do
  agent="${line%% *}"; bench="${line##* }"
  if ! in_filter "$agent" "$AGENT_FILTER" || ! in_filter "$bench" "$BENCH_FILTER"; then
    n_skip_filter+=1
    continue
  fi
  n_run+=1
  if [ "$DRY" -eq 1 ]; then
    conf="$(agent_field "$agent" confidence)"
    printf '  RUN  %-12s x %-16s  (agent_cli=%s)\n' "$agent" "$bench" "$conf"
    continue
  fi
  log "cell $n_run: $agent x $bench"
  # per-cell env the collector and adapters key on
  BENCHMARK="$bench" SYSTEMS="$agent" \
  "$EXT/run_cell.sh" "$agent" "$bench" "" "$ROUNDS" "$REPEAT" "$TEMP" "$TIMEOUT" "$LIMIT" \
    || echo "[matrix] cell $agent x $bench returned nonzero (continuing)"
done

log "done: $n_run cell(s) $([ "$DRY" -eq 1 ] && echo would run || echo ran), \
$n_skip_filter filtered out"
if [ "$DRY" -eq 0 ] && [ "$n_run" -gt 0 ]; then
  echo "next: python3 $EXT/collect_matrix.py   # aggregate runs/ -> results/matrix/"
fi
