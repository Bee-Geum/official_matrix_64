#!/usr/bin/env bash
# lib/common.sh -- shared helpers for the unified_bench benchmark-matrix extension.
# Source this from every runner:  source "$(dirname "$0")/lib/common.sh"
#
# Resolves:
#   UB_ROOT   the existing unified_bench checkout (default /home/bi_geum/unified_bench when installed there)
#   EXT       this extension dir (unified_bench_ext, lives inside UB_ROOT)

set -euo pipefail

EXT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UB_ROOT="${UB_ROOT:-$(cd "$EXT/.." && pwd)}"

AGENTS_CSV="$EXT/registry/agents.csv"
BENCH_CSV="$EXT/registry/benchmarks.csv"

log() {
  echo
  echo "=========================================="
  echo "$*"
  echo "=========================================="
}

die() { echo "ERROR: $*" >&2; exit 1; }

# source whatever CUDA/env setup the harness uses (mirrors a100_official5_resume_expand.sh)
source_env() {
  export UB_TARGET_GPU_LABEL="${UB_TARGET_GPU_LABEL:-RTX PRO 6000}"
  export GPU_NAME="${GPU_NAME:-RTX PRO 6000}"
  if [ -f "$UB_ROOT/use_rtx6000_cuda_env.sh" ]; then
    # shellcheck disable=SC1091
    source "$UB_ROOT/use_rtx6000_cuda_env.sh"
  elif [ -f "$UB_ROOT/scripts/use_rtx6000_cuda_env.sh" ]; then
    # shellcheck disable=SC1091
    source "$UB_ROOT/scripts/use_rtx6000_cuda_env.sh"
  elif [ -f "$UB_ROOT/use_a100_cuda126_env.sh" ]; then
    # shellcheck disable=SC1091
    source "$UB_ROOT/use_a100_cuda126_env.sh"
  elif [ -f "$UB_ROOT/scripts/use_a100_cuda_env.sh" ]; then
    # shellcheck disable=SC1091
    source "$UB_ROOT/scripts/use_a100_cuda_env.sh"
  fi
}

# csv_get <file> <key_col> <key_val> <want_col>
# Pulls one field from a '#'-commented CSV. Inner lists use '|', so commas are safe.
csv_get() {
  python3 - "$1" "$2" "$3" "$4" <<'PY'
import csv, sys
path, key_col, key_val, want = sys.argv[1:5]
with open(path) as fh:
    lines = [ln for ln in fh if ln.strip() and not ln.lstrip().startswith("#")]
for row in csv.DictReader(lines):
    if row.get(key_col, "").strip() == key_val:
        print(row.get(want, "").strip())
        break
PY
}

agent_field() { csv_get "$AGENTS_CSV" agent "$1" "$2"; }
bench_field() { csv_get "$BENCH_CSV" benchmark "$1" "$2"; }

# is this (agent,benchmark) cell a RUN? prints RUN or SKIP and the reason.
cell_status() {
  UB_EXT_DIR="$EXT" python3 - "$1" "$2" <<'PY'
import os, sys
sys.path.insert(0, os.path.join(os.environ["UB_EXT_DIR"], "registry"))
import compat
agents = {a["agent"]: a for a in compat.load_csv(compat.HERE / "agents.csv")}
benches = {b["benchmark"]: b for b in compat.load_csv(compat.HERE / "benchmarks.csv")}
a, b = sys.argv[1], sys.argv[2]
if a not in agents:  print("SKIP\tunknown agent"); raise SystemExit
if b not in benches: print("SKIP\tunknown benchmark"); raise SystemExit
c = compat.decide(agents[a], benches[b])
print(c["status"] + "\t" + c["reason"])
PY
}

ensure_layout() {
  mkdir -p "$UB_ROOT/logs" "$UB_ROOT/runs" \
           "$UB_ROOT/results/matrix" "$UB_ROOT/third_party" \
           "$EXT/task_lists"
}

# clone_repo <url> <dest_dir_under_third_party> [extra git-clone flags...]
# Idempotent: skips if present. Records resolved SHA into third_party/repo_lock_ext.json.
# On clone failure prints a precise message instead of proceeding with a wrong URL.
clone_repo() {
  local url="$1" dest="$2"; shift 2
  local path="$UB_ROOT/third_party/$dest"
  if [ -d "$path/.git" ] || [ -d "$path" ]; then
    echo "[prepare] reuse $path"
  else
    echo "[prepare] cloning $url -> $path"
    if ! git clone "$@" "$url" "$path"; then
      echo "[prepare] FAILED to clone $url"
      echo "[prepare]   This URL may differ from your environment. Either:"
      echo "[prepare]     - set the repo override env var and re-run, or"
      echo "[prepare]     - place the checkout manually at $path"
      return 1
    fi
  fi
  local sha; sha="$(git -C "$path" rev-parse HEAD 2>/dev/null || echo unknown)"
  UB_LOCK="$UB_ROOT/third_party/repo_lock_ext.json" \
  UB_DEST="$dest" UB_URL="$url" UB_SHA="$sha" python3 - <<'PY'
import json, os
lock = os.environ["UB_LOCK"]
try:
    data = json.load(open(lock))
except Exception:
    data = {}
data[os.environ["UB_DEST"]] = {"url": os.environ["UB_URL"], "sha": os.environ["UB_SHA"]}
json.dump(data, open(lock, "w"), indent=2)
print("[prepare] locked", os.environ["UB_DEST"], "@", os.environ["UB_SHA"][:12])
PY
}

# build_kb_task_list <kb_root_under_third_party> <levels_csv> <out_basename> [name_filter_substr]
# Emits $EXT/task_lists/<out_basename>.txt (+ .csv) listing KernelBench-form task files.
# This is the generalized form of the harness's build_all250.
build_kb_task_list() {
  local kb_rel="$1" levels="$2" out="$3" filt="${4:-}"
  UB_ROOT="$UB_ROOT" KB_REL="$kb_rel" LEVELS="$levels" OUT="$EXT/task_lists/$out" FILT="$filt" \
  python3 - <<'PY'
import csv, os, re
from pathlib import Path
root = Path(os.environ["UB_ROOT"])
kb = root / "third_party" / os.environ["KB_REL"]
levels = [x for x in os.environ["LEVELS"].split(",") if x]
filt = os.environ["FILT"]
if not kb.exists():
    raise SystemExit(f"KernelBench-form root not found: {kb}")

def tid(p):
    m = re.match(r"(\d+)_", p.name)
    return int(m.group(1)) if m else 10**9

rows = []
for lv in levels:
    d = kb / lv
    if not d.exists():
        continue
    for p in sorted(d.glob("*.py"), key=tid):
        if filt and filt not in p.name:
            continue
        rows.append({"level": lv, "task_id": tid(p),
                     "task_path": str(p.relative_to(root)), "task_name": p.name})

out = Path(os.environ["OUT"])
out.with_suffix(".txt").write_text("\n".join(r["task_path"] for r in rows) + ("\n" if rows else ""))
with out.with_suffix(".csv").open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["level", "task_id", "task_path", "task_name"])
    w.writeheader(); w.writerows(rows)
print(f"[prepare] {out.name}: {len(rows)} tasks  ({', '.join(lv + '=' + str(sum(r['level']==lv for r in rows)) for lv in levels)})")
PY
}
