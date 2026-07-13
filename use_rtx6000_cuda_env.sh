#!/usr/bin/env bash
# RTX PRO 6000 / bi_geum default environment for unified_bench_ext.

export UB_ROOT="${UB_ROOT:-/home/bi_geum/unified_bench}"
export UB_TARGET_GPU_LABEL="${UB_TARGET_GPU_LABEL:-RTX PRO 6000}"
export GPU_NAME="${GPU_NAME:-RTX PRO 6000}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# CUDA-L1 has no official RTX PRO 6000 JSON in the public artifact.
# h100.json is used as cross-hardware replay unless GPU_JSON is overridden.
export GPU_JSON="${GPU_JSON:-h100.json}"
export CUDA_L1_CROSS_HARDWARE="${CUDA_L1_CROSS_HARDWARE:-1}"

NVCC_ROOT=$(python3 - <<'PY'
from pathlib import Path
try:
    import nvidia.cuda_nvcc
    p = Path(nvidia.cuda_nvcc.__file__).resolve()
    for c in [p.parent] + list(p.parents):
        if (c / 'bin' / 'nvcc').exists():
            print(c)
            raise SystemExit(0)
except Exception:
    pass
print('')
PY
)

if [ -n "$NVCC_ROOT" ] && [ -x "$NVCC_ROOT/bin/nvcc" ]; then
  export CUDA_HOME="$NVCC_ROOT"
elif [ -d /usr/local/cuda-12.6 ]; then
  export CUDA_HOME=/usr/local/cuda-12.6
elif [ -d /usr/local/cuda-12.5 ]; then
  export CUDA_HOME=/usr/local/cuda-12.5
elif [ -d /usr/local/cuda-12 ]; then
  export CUDA_HOME=/usr/local/cuda-12
elif [ -d /usr/local/cuda ]; then
  export CUDA_HOME=/usr/local/cuda
fi

if [ -n "${CUDA_HOME:-}" ]; then
  export CUDA_PATH="$CUDA_HOME"
  export CUDACXX="$CUDA_HOME/bin/nvcc"
  export PATH="$CUDA_HOME/bin:$PATH"
  [ -d "$CUDA_HOME/lib64" ] && export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi
