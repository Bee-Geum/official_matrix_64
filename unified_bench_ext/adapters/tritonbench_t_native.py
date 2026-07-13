from __future__ import annotations

from pathlib import Path
import sys
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
from unified_bench_ext.adapters._adapter_common import normalize_call_args, smoke_eval, cli_main

ADAPTER_NAME = "tritonbench_t_native"


def run_native(*args, **kwargs):
    task, cand_dir, out_dir = normalize_call_args(*args, **kwargs)
    extra = {
        "native_note": "non_rocm_adapters_v2 compatibility adapter. This performs candidate presence + syntax/import/compile smoke unless a benchmark-specific official oracle is later connected.",
    }
    return smoke_eval(task, cand_dir, out_dir, ADAPTER_NAME, official_eval=0, extra=extra)


if __name__ == "__main__":
    cli_main(ADAPTER_NAME, official_eval=0)
