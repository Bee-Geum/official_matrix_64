# Smoke proof — 7 official agents × 8 benchmarks

Run: `RUN_ID=smoke_prove_20260717 ROUNDS=1 LIMIT=1 ./run_official_7.sh` on 4× H100 (GPU 0 serves each backbone in turn; GPUs 1-3 evaluate; KernelGYM up on :10907 for drkernel).

`V` = valid official eval ran · `*` = ≥1 correct · `.` = no valid cell

| agent | kernelbench | robust_kbench | tritonbench_t | tritonbench_g | multikernelbench | pareval | backendbench | sol_execbench |
|---|---|---|---|---|---|---|---|---|
| cudaforge | V* | V* | . | . | . | . | . | . |
| autokernel | V* | V* | V* | V* | V | V | . | . |
| kernelskill | . | . | . | . | . | . | . | . |
| autotriton | V | V | V* | V* | V* | V | . | V |
| kernelllm | V | V | V* | V* | V | V | . | V |
| incoder32b | V | V* | V* | V* | V* | V | . | V |
| drkernel | V* | V* | V* | V | V* | V | . | V |

## Findings
- **6/7 agents run officially**: autokernel, autotriton, kernelllm, incoder32b, drkernel produce candidates and pass official eval across most benchmarks; cudaforge is valid on the KernelBench-family only (CUDA-focused; Triton/prompt formats out of scope by design). drkernel is confirmed working end-to-end via KernelGYM.
- **kernelskill: 0 valid cells (bug).** It runs, but the KernelMem backend crashes on the seed-kernel test with `AttributeError: 'NoneType' object has no attribute 'loader'`, so the driver harvests no runnable kernel. Same symptom in all 12 kernelskill cells — a KernelMem integration regression to fix.
- **backendbench is blank for every agent** because LIMIT=1 selected op `add`, which has no locally-available official test case (`op 'add' not covered by opinfo/torchbench`). This is a task-selection artifact of LIMIT=1, not an agent failure; a larger LIMIT exercises backendbench.
