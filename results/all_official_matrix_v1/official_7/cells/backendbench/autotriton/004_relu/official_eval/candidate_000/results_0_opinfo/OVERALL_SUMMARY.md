# BackendBench Run Summary

## Command
```bash
python -m BackendBench.scripts.main --suite opinfo --backend directory --ops-directory /home/bi_geum/official_matrix_64/results/all_official_matrix_v1/official_7/cells/backendbench/autotriton/004_relu/official_eval/candidate_000/ops --log-dir /home/bi_geum/official_matrix_64/results/all_official_matrix_v1/official_7/cells/backendbench/autotriton/004_relu/official_eval/candidate_000/results_0_opinfo
```

## Results

| Metric | Value |
|--------|-------|
| Correctness Score | 0.89 |
| Performance Score (geomean speedup) | nan |
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
| sum.default | 100.0000% | N/A|
| mean.default | 100.0000% | N/A|
| tanh.default | 100.0000% | N/A|
| bmm.default | 100.0000% | N/A|
| mm.default | 100.0000% | N/A|
| sigmoid.default | 100.0000% | N/A|
| gelu.default | 100.0000% | N/A|
| relu.default | 0.0000% | N/A|
