# BackendBench Run Summary

## Command
```bash
python -m BackendBench.scripts.main --suite smoke --backend directory --ops-directory /home/bi_geum/official_matrix_64/results/all_official_matrix_v1/official_7/cells/backendbench/autotriton/002_sub/official_eval/candidate_000/ops --log-dir /home/bi_geum/official_matrix_64/results/all_official_matrix_v1/official_7/cells/backendbench/autotriton/002_sub/official_eval/candidate_000/results_2_smoke
```

## Results

| Metric | Value |
|--------|-------|
| Correctness Score | 1.00 |
| Performance Score (geomean speedup) | 0.90 |
| Perf@1.0 Score | 0.00 |

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
| relu.default | 100.0000% | 0.8982x|
