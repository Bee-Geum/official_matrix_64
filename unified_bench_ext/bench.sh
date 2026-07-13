#!/usr/bin/env bash
# bench.sh -- unified_bench 벤치마크 매트릭스 확장 진입점
#
# 논문(Table 2)의 opensource 에이전트 8종 x opensource 벤치마크 10종을
# 호환성 매트릭스로 묶어, RTX PRO 6000 단일 GPU에서 실행 가능한 모든 셀을 돌립니다.
# 기존 unified_bench harness는 건드리지 않고, 검증된 driver/evaluator CLI를
# 그대로 재사용하는 drop-in 확장입니다.
#
# 기본 경로:
#   UB_ROOT = /home/bi_geum/unified_bench   (환경변수로 override 가능)
#   확장 디렉터리는 이 스크립트가 있는 곳(unified_bench_ext)
#
# 사용법:
#   cd /home/bi_geum/unified_bench
#   chmod +x unified_bench_ext/bench.sh
#
#   ./unified_bench_ext/bench.sh compat            # 매트릭스 출력 (어떤 셀이 RUN/SKIP인지)
#   ./unified_bench_ext/bench.sh prepare-all       # 모든 벤치마크 checkout + task list 생성
#   ./unified_bench_ext/bench.sh prepare <bench>   # 특정 벤치마크만 준비
#   ./unified_bench_ext/bench.sh matrix [opts]     # RUN 셀 전체 실행 (run_matrix.sh로 위임)
#   ./unified_bench_ext/bench.sh cell <agent> <bench> [opts]  # 단일 셀 실행
#   ./unified_bench_ext/bench.sh collect [bench]   # runs/ -> results/matrix/ 집계
#   ./unified_bench_ext/bench.sh pack              # results/matrix/ 아카이브 생성
#
# 주의:
# - 기존 runs/를 rm 하지 않습니다.
# - SKIP 셀(AMD/NPU/TPU 전용, 언어 불일치)은 이유와 함께 건너뜁니다.
# - agents.csv의 'inferred' 4종(drkernel/geak/ksearch/cuda_agent)은 driver CLI가
#   아카이브에 없어 추정값입니다. 한 줄씩 확인 후 매트릭스 전체 실행을 권장합니다.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/common.sh"

MODE="${1:-help}"; shift || true

case "$MODE" in
  compat)
    # 단일 진실 공급원: registry/compat.py. 매트릭스 + compat_matrix.csv 출력.
    UB_EXT_DIR="$EXT" python3 "$EXT/registry/compat.py" "$@"
    ;;

  prepare-all)
    ensure_layout
    log "모든 벤치마크 준비 (checkout + task list)"
    rc=0
    for b in $(awk -F, 'NR>1 && $1!~/^#/ {print $1}' "$BENCH_CSV"); do
      prep="$(bench_field "$b" prepare)"
      script="$EXT/${prep#unified_bench_ext/}"
      if [ -f "$script" ]; then
        echo; echo "--- prepare $b ---"
        bash "$script" || { echo "[bench] $b 준비 실패 (계속)"; rc=1; }
      else
        echo "[bench] $b: prepare 스크립트 없음 ($script) -- 건너뜀"
      fi
    done
    echo; echo "[bench] task_lists/:"; ls -1 "$EXT/task_lists" 2>/dev/null || true
    exit $rc
    ;;

  prepare)
    ensure_layout
    b="${1:?사용법: bench.sh prepare <benchmark>}"
    prep="$(bench_field "$b" prepare)"
    [ -n "$prep" ] || die "알 수 없는 벤치마크: $b"
    bash "$EXT/${prep#unified_bench_ext/}"
    ;;

  matrix)
    "$EXT/run_matrix.sh" "$@"
    ;;

  cell)
    a="${1:?사용법: bench.sh cell <agent> <benchmark> [task_list rounds repeat temp timeout limit]}"
    b="${2:?benchmark 필요}"; shift 2
    "$EXT/run_cell.sh" "$a" "$b" "$@"
    ;;

  collect)
    # [bench] 인자가 있으면 그 벤치마크만 (campaign 혼입 방지)
    if [ $# -gt 0 ]; then
      python3 "$EXT/collect_matrix.py" --benchmark "$1"
    else
      python3 "$EXT/collect_matrix.py"
    fi
    ;;

  pack)
    ensure_layout
    ts="$(date +%Y%m%d_%H%M%S)"
    out="$UB_ROOT/results/matrix_archive_${ts}.tar.gz"
    log "결과 아카이브 생성 -> $out"
    tar -czf "$out" -C "$UB_ROOT" \
      results/matrix \
      $( [ -f "$UB_ROOT/third_party/repo_lock_ext.json" ] && echo third_party/repo_lock_ext.json ) \
      2>/dev/null || die "아카이브 생성 실패 (먼저 collect 실행)"
    echo "[bench] wrote $out"
    ;;

  help|--help|-h|*)
    sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
    ;;
esac
