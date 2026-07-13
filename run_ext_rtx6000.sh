#!/usr/bin/env bash
# Self-contained wrapper: UB_ROOT defaults to THIS script's directory.
set -euo pipefail
ROOT="${UB_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
export UB_ROOT="$ROOT"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
[ -f "$ROOT/use_rtx6000_cuda_env.sh" ] && source "$ROOT/use_rtx6000_cuda_env.sh"
exec "$ROOT/unified_bench_ext/bench.sh" "$@"
