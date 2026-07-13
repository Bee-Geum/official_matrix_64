# Official Matrix 88 (11 agents x 8 official benchmarks)

- cells executed: **88/88**
- cells with official-oracle verdict: **87/88**

## Per-benchmark

| benchmark | official oracle | agents ran | official_eval | correct |
|---|---|---|---|---|
| kernelbench | kb_instrumented | 11/11 | 11/11 | 11/11 |
| robust_kbench | kb_instrumented | 11/11 | 11/11 | 11/11 |
| tritonbench_t | TritonBench EVAL | 11/11 | 11/11 | 11/11 |
| tritonbench_g | TritonBench EVAL | 11/11 | 11/11 | 11/11 |
| multikernelbench | eval_single_runner | 11/11 | 10/11 | 8/11 |
| backendbench | BackendBench CLI | 11/11 | 11/11 | 1/11 |
| pareval | ParEval run-all | 11/11 | 11/11 | 0/11 |
| sol_execbench | sol_execbench.cli | 11/11 | 11/11 | 11/11 |

## official_eval task-count matrix (rows=benchmark, cols=agent)

| benchmark | cudaforg | autokern | cuda_l1 | autotrit | drkernel | geak | ksearch | cuda_age | kernelll | incoder3 | kernelsk |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kernelbench | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| robust_kbench | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| tritonbench_t | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| tritonbench_g | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| multikernelbench | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 1 | 1 | 1 | 1 |
| backendbench | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| pareval | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| sol_execbench | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

_official_eval = the benchmark's official oracle ran and returned a verdict._
_correct = that official verdict was a pass. correct<official is a normal benchmark outcome (the agent's kernel failed the real check), not a pipeline error._
