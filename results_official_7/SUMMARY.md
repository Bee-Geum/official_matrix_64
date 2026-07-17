# Official Matrix 88 (11 agents x 8 official benchmarks)

- cells executed: **56/88**
- cells with official-oracle verdict: **36/88**

## Per-benchmark

| benchmark | official oracle | agents ran | official_eval | correct |
|---|---|---|---|---|
| kernelbench | kb_instrumented | 7/11 | 6/11 | 3/11 |
| robust_kbench | kb_instrumented | 7/11 | 6/11 | 4/11 |
| tritonbench_t | TritonBench EVAL | 7/11 | 5/11 | 5/11 |
| tritonbench_g | TritonBench EVAL | 7/11 | 5/11 | 4/11 |
| multikernelbench | eval_single_runner | 7/11 | 5/11 | 3/11 |
| backendbench | BackendBench CLI | 7/11 | 0/11 | 0/11 |
| pareval | ParEval run-all | 7/11 | 5/11 | 0/11 |
| sol_execbench | sol_execbench.cli | 7/11 | 4/11 | 0/11 |

## official_eval task-count matrix (rows=benchmark, cols=agent)

| benchmark | cudaforg | autokern | autotrit | drkernel | kernelll | incoder3 | kernelsk |
|---|---|---|---|---|---|---|---|
| kernelbench | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| robust_kbench | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| tritonbench_t | 0 | 1 | 1 | 1 | 1 | 1 | 0 |
| tritonbench_g | 0 | 1 | 1 | 1 | 1 | 1 | 0 |
| multikernelbench | 0 | 1 | 1 | 1 | 1 | 1 | 0 |
| backendbench | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| pareval | 0 | 1 | 1 | 1 | 1 | 1 | 0 |
| sol_execbench | 0 | 0 | 1 | 1 | 1 | 1 | 0 |

_official_eval = the benchmark's official oracle ran and returned a verdict._
_correct = that official verdict was a pass. correct<official is a normal benchmark outcome (the agent's kernel failed the real check), not a pipeline error._
