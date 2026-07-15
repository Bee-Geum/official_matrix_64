# official_matrix_64 (self-contained)

8 kernel-generation agents × 8 official benchmarks = **64 cells**, each scored by the benchmark's
**official oracle** (faithful reproduction; the upstream harness also ships 3 more agents that are
model-unreleased or AMD-only — see `THIRD_PARTY.md`). This directory is **self-contained and portable** — every
agent driver, benchmark repo + oracle, evaluator, registry, and task list is
bundled inside. Copy the folder to another machine and run it.

The only thing **not** bundled is the LLM weights (too large). Point the runner
at any OpenAI-compatible endpoint (`LLM_BASE_URL`), or run the bundled
`hf_openai_server.py` with a model of your choice.

## What runs

| # | benchmark | official oracle |
|---|-----------|-----------------|
| 1 | kernelbench      | instrumented eval (compile + correctness + perf) |
| 2 | robust_kbench    | instrumented eval (KernelBench L1/L2 task set) |
| 3 | tritonbench_t    | TritonBench `EVAL/eval_T` scripts |
| 4 | tritonbench_g    | TritonBench `EVAL/eval_G` scripts |
| 5 | multikernelbench | `eval_single_runner.py` |
| 6 | backendbench     | BackendBench official CLI (`scripts/main.py`) |
| 7 | pareval          | ParEval `drivers/run-all.py` |
| 8 | sol_execbench    | `python -m sol_execbench.cli.main` |

11 agents (all route through one LLM endpoint via `drivers/generic_llm_kernel_driver.py`):
cudaforge, autokernel, cuda_l1, autotriton, drkernel, geak, ksearch, cuda_agent,
kernelllm, incoder32b, kernelskill.

> flashinfer_bench (no upstream oracle wired) and rocm_tritonbench (AMD-only) are
> excluded — these 8 are the benchmarks with a working official oracle.

## What's inside (this is a self-sufficient ROOT)

```
official_matrix_64/
├── official_all_matrix_v1.py     # the runner (patched oracles + fixed parsers)
├── run_matrix_64.sh              # self-contained launcher (ROOT = this dir)
├── summarize_matrix_64.py        # cells -> 8x8 tables
├── requirements.txt
├── drivers/                      # all 11 agent drivers + generic LLM driver
├── telemetry/                    # instrumented_final_eval.py (kb oracle)
├── unified_bench_ext/registry/   # agents.csv, benchmarks.csv
├── third_party/                  # 6 benchmark repos WITH their official oracles
│   ├── KernelBench/  TritonBench/  MultiKernelBench/
│   └── BackendBench/  ParEval/  SOL-ExecBench/
├── kernelbench_all250.txt        # task lists (paths relative to this dir)
├── robust_kbench_l12.txt
├── hf_openai_server.py           # optional bundled LLM server
└── results/                      # summaries land here after a run
```

Prepared task manifests are **generated on first run** (`prepare`) from the
bundled repos, so all paths are local to wherever you unpack this — nothing is
hard-coded to the original machine.

## Prerequisites (target machine)

- NVIDIA GPU + driver, and **`nvcc` on PATH** (KernelBench/ParEval/BackendBench compile kernels).
- Python 3.10+ and `pip install -r requirements.txt` (install the torch/triton build matching your CUDA).
- An OpenAI-compatible LLM endpoint. Either:
  - point `LLM_BASE_URL` at an existing server, **or**
  - `AUTO_START_SERVER=1` to launch the bundled `hf_openai_server.py` (needs `transformers`/`fastapi`/`uvicorn` + a local model).

## Run

```bash
cd official_matrix_64
pip install -r requirements.txt

# A) use an existing LLM endpoint
LLM_BASE_URL=http://127.0.0.1:8000/v1 ./run_matrix_64.sh

# B) launch the bundled server (local GPU + model)
AUTO_START_SERVER=1 MODEL_ID=Qwen/Qwen2.5-Coder-14B-Instruct ./run_matrix_64.sh

# background
nohup ./run_matrix_64.sh > run_matrix_64.out 2>&1 &
tail -f run_matrix_64.out
```

### Knobs (environment variables)

| var | default | meaning |
|-----|---------|---------|
| `LLM_BASE_URL` | `http://127.0.0.1:8000/v1` | OpenAI-compatible endpoint used by every agent |
| `EVAL_MODEL` | `qwen14b` | model name sent to the endpoint |
| `AUTO_START_SERVER` | `0` | `1` = start bundled `hf_openai_server.py` |
| `MODEL_ID` | Qwen2.5-Coder-14B | model for the bundled server |
| `LIMIT` | `1` | tasks per cell |
| `AGENTS` / `BENCHMARKS` | `all` | comma list to subset |
| `MAX_CANDIDATES` / `CELL_ATTEMPTS` | `1` / `1` | candidates per task / retries per cell |
| `GPUS` | `all` | eval GPUs: `all`, or `0,1,2,3`. `GPUS=0` = single-GPU serial run |
| `WORKERS_PER_GPU` | `1` | cells co-located per GPU (see the warning below) |
| `CELL_WORKERS` | `#GPUs × WORKERS_PER_GPU` | concurrent cells |
| `SERVER_GPU` / `DEDICATE_SERVER_GPU` | `0` / `1` | with `AUTO_START_SERVER=1`, the LLM gets this GPU and eval uses the rest |
| `GPU` | first of `GPUS` | single-GPU fallback; `--gpus` takes precedence |
| `TORCH_CUDA_ARCH_LIST` | auto (`compute_cap`) | `9.0` on H100, `12.0` on Blackwell |
| `MAX_JOBS` | `nproc / CELL_WORKERS` | nvcc build jobs per cell |
| `REPREPARE` | `0` | `1` = regenerate task manifests |

Examples:
```bash
LIMIT=3 ./run_matrix_64.sh
AGENTS=cudaforge,geak BENCHMARKS=pareval,sol_execbench ./run_matrix_64.sh
```

### Multi-GPU

By default every visible GPU runs one cell at a time, and cells are dispatched to
whichever GPU frees up first. Each cell is pinned with `CUDA_VISIBLE_DEVICES` and gets a
private `TORCH_EXTENSIONS_DIR` / `TRITON_CACHE_DIR` / `TMPDIR` under its own cell
directory, so concurrent cells never race on a build dir and a cell killed by timeout
cannot strand a stale torch `FileBaton` lock in a shared cache.

```bash
# 4 GPUs, external LLM endpoint -> 4 cells at a time
LLM_BASE_URL=http://HOST:8000/v1 GPUS=0,1,2,3 ./run_matrix_64.sh

# 4 GPUs, bundled server on GPU 0 -> eval runs 3-wide on GPUs 1,2,3
AUTO_START_SERVER=1 SERVER_GPU=0 ./run_matrix_64.sh
```

> **`WORKERS_PER_GPU>1` invalidates the performance numbers.** Co-located cells contend
> for SMs and memory bandwidth, so `best_score` / speedup stop being meaningful. Use it
> only when you care about `correct` alone. One cell per GPU keeps timings comparable to
> the single-GPU run.

## Outputs

Written to `results/`:
- `SUMMARY.md`, `matrix_88_official_eval.csv`, `matrix_88_correct.csv`, `matrix_88_summary.csv`

Raw per-cell data: `results/all_official_matrix_v1/official_matrix_64/cells/<benchmark>/<agent>/cell_result.json`

## Reading the numbers

- `official_eval > 0` ⇒ the benchmark's official oracle actually ran and returned a verdict.
- `correct` ⇒ that verdict was a pass. `correct < official_eval` is a normal
  benchmark outcome (the agent's kernel failed the real check), not a pipeline error.
- backendbench: elementwise overhead-dominated ops (add/mul/sub/div) have no local
  test case in the available suites and are reported as "not covered"; compute-bound
  ops (relu/mm/bmm/gelu/…) get a real verdict.

## Notes on portability

- All 11 agents call one LLM endpoint; no per-agent model weights are needed.
- third_party repos are bundled **without `.git`** and without unused training
  data (e.g. TritonBench `LLM_generated/` and `train_*.json` were dropped) — the
  official EVAL scripts and task files are kept.
- Oracle fixes are applied idempotently at eval time (ParEval `contextlib.chdir`
  3.10 backport; SOL-ExecBench `cupti`→CUDA-events; TritonBench f-string patch),
  each leaving a `.bak_official_matrix` backup.

## Re-summarize without re-running

```bash
python3 summarize_matrix_64.py \
  --run-root results/all_official_matrix_v1/official_matrix_64 --out results
```

## License & Attribution

This harness **vendors upstream open-source benchmarks and kernel-generation agents** under
`third_party/`. All bundled code remains under its original authors' licenses. See
[`THIRD_PARTY.md`](THIRD_PARTY.md) for the full list of components, upstream URLs, licenses, and
paper citation keys. Model weights are not bundled (shared HF cache); each model follows its own
model-card license. Please cite the original benchmark/agent papers and comply with each
component's license when using this repository.
