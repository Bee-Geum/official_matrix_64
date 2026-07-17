# BackendBench Run Summary

## Command
```bash
python -m BackendBench.scripts.main --suite torchbench --backend directory --ops-directory /home/bi_geum/official_matrix_64/results/all_official_matrix_v1/official_7/cells/backendbench/kernelllm/001_mul/official_eval/candidate_001/ops --log-dir /home/bi_geum/official_matrix_64/results/all_official_matrix_v1/official_7/cells/backendbench/kernelllm/001_mul/official_eval/candidate_001/results_1_torchbench --check-overhead-dominated-ops
```

## Results

| Metric | Value |
|--------|-------|
| Correctness Score | 1.00 |
| Performance Score (geomean speedup) | 1.00 |
| Perf@1.0 Score | 1.00 |

### Metric Descriptions

- **Correctness Score**: Mean pass rate over all operators
- **Performance Score**: Geometric mean speedup over all operators
- **Perf@1.0 Score**: Rate of correct samples with a speedup greater than 1.0

## Output Files

The following files are saved in this directory:

- `full_results.json`: Complete test results for all operators
- `operator_summary.csv`: Operator-level summary statistics
- `failed_tests.json`: Log of failed tests (if any)
- `OVERALL_SUMMARY.md`: This file
### Operator Speedups vs Eager in Descending Order

| Operator | Correctness Ratio | Speedup vs Eager |
|----------|-----------|----------------|
| mm.default | 100.0000% | 1.0015x|
| relu.default | 100.0000% | 1.0006x|
| sigmoid.default | 100.0000% | 1.0005x|
