"""MLX Mamba selective scan forward task adapter.

This adds an Apple-Silicon-only MLX task to K-Search.
The model must generate a `submission.py` defining `run(...)` for the forward pass.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict

from k_search.tasks.task_base import (
    BuildSpec,
    EvalResult,
    Solution,
    SourceFile,
    SupportedLanguages,
    load_ksearch_solution_json,
    solution_from_json_dict,
)

from k_search.tasks.mlx_mamba.evaluator import evaluate_mlx_mamba_selective_scan_fwd_submission
from k_search.tasks.mlx_mamba.spec import (
    MAMBA_SELECTIVE_SCAN_FWD_WORKLOADS,
    MambaSelectiveScanFwdWorkload,
    get_definition_text_mlx,
)
from k_search.utils.metal_gpu_info import get_gpu_info, get_metal_device_name


@dataclass(frozen=True)
class MlxMambaTaskConfig:
    warmup_runs: int = 3
    iterations: int = 10
    rtol: float = 1e-3
    atol: float = 1e-3
    timeout_seconds: int = 300
    per_workload_timeout_seconds: int | None = None
    max_failure_excerpt_chars: int = int(os.getenv("KSEARCH_MLX_MAMBA_FAILURE_EXCERPT_CHARS", "4000"))


class MlxMambaSelectiveScanFwdTask:
    """K-Search Task: MLX Mamba selective scan forward on Apple Silicon."""

    def __init__(
        self,
        *,
        warmup_runs: int = 3,
        iterations: int = 10,
        rtol: float = 1e-3,
        atol: float = 1e-3,
        timeout_seconds: int = 300,
        artifacts_dir: str | None = None,
        name: str = "mlx_mamba_selective_scan_fwd",
        workloads: list[MambaSelectiveScanFwdWorkload] | None = None,
    ) -> None:
        self._name = str(name or "mlx_mamba_selective_scan_fwd")

        per_wl_timeout: int | None = None
        per_wl_env = str(os.getenv("KSEARCH_MLX_MAMBA_PER_WORKLOAD_TIMEOUT_SECONDS", "")).strip()
        if per_wl_env:
            try:
                v = int(per_wl_env)
                if v > 0:
                    per_wl_timeout = v
            except Exception:
                per_wl_timeout = None

        self._cfg = MlxMambaTaskConfig(
            warmup_runs=int(warmup_runs),
            iterations=int(iterations),
            rtol=float(rtol),
            atol=float(atol),
            timeout_seconds=int(timeout_seconds),
            per_workload_timeout_seconds=per_wl_timeout,
        )
        self._workloads = list(workloads or MAMBA_SELECTIVE_SCAN_FWD_WORKLOADS)
        self._ksearch_artifacts_dir: str | None = (str(artifacts_dir) if artifacts_dir is not None else None)
        self._solutions: dict[str, Solution] = {}

        # Last-round feedback cache (read by generators via getattr).
        self._last_round_trace_logs_for_prompt: str = ""
        self._last_round_passed_count: int = 0
        self._last_round_total_workloads: int = len(self._workloads)
        self._last_round_summary_line: str = ""

    @property
    def name(self) -> str:
        return self._name

    def get_definition_text(self, language: str | None = None) -> str:
        lang = str(language or "").strip().lower() or "mlx"
        if lang != "mlx":
            raise ValueError(f"MlxMambaSelectiveScanFwdTask only supports language='mlx'; got {lang!r}")
        return get_definition_text_mlx() + "\n"

    def get_solution(self, solution_name: str) -> Solution | None:
        if solution_name in self._solutions:
            return self._solutions[solution_name]
        if self._ksearch_artifacts_dir:
            try:
                d = load_ksearch_solution_json(
                    solution_ref=solution_name,
                    definition_name=self._name,
                    artifacts_dir=self._ksearch_artifacts_dir,
                )
                sol = solution_from_json_dict(d)
                self._solutions[solution_name] = sol
                return sol
            except Exception:
                return None
        return None

    def get_generation_prompt(self, *, language: str, target_gpu: str) -> str:
        # NOTE: `target_gpu` is ignored for MLX tasks; we auto-detect hardware when possible.
        hw = get_gpu_info().strip()
        if not hw:
            hw = "Target hardware: Apple Silicon (auto-detect unavailable)"
        return f"{self.get_definition_text(language=language).strip()}\n\n{hw}\n"

    def get_optimization_prompt(
        self,
        *,
        language: str,
        target_gpu: str,
        trace_logs: str,
        current_code: str,
        current_best: str | None = None,
        previous_round_summary: str | None = None,
    ) -> str:
        base = self.get_definition_text(language=language).rstrip()
        hw = get_gpu_info().strip()
        if not hw:
            hw = "Target hardware: Apple Silicon (auto-detect unavailable)"
        parts: list[str] = [
            f"{base}\n\n{hw}",
            f"Current implementation:\n{str(current_code or '').strip()}",
        ]
        if previous_round_summary:
            parts.append(f"Previous round summary:\n{previous_round_summary.strip()}")
        if trace_logs:
            parts.append(f"Execution log / metrics:\n{trace_logs.strip()}")
        if current_best:
            parts.append(f"Current best:\n{current_best.strip()}")
        parts.append(
            "Before changing the code: briefly analyze bottlenecks using latency and memory pressure. "
            "Then implement the optimized version.\n\n"
            "Rules: return only Python code (no markdown), keep entrypoint run(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=True)."
        )
        return "\n\n".join(p for p in parts if p.strip())

    def get_code_format_text(self, *, language: str, target_gpu: str) -> str:
        return (
            "MLX code format rules:\n"
            "- Return valid Python only. No markdown fences, no explanations.\n"
            "- Keep entrypoint exactly: run(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=True)\n"
            "- Use mlx.core as mx (import mlx.core as mx).\n"
        )

    def make_solution_from_generated_code(
        self,
        *,
        cleaned_code: Any,
        raw_code: Any,
        round_num: int,
        model_name: str,
        target_gpu: str,
        language: str,
    ) -> Solution:
        code_text = str(cleaned_code or "") or str(raw_code or "")
        sol_name = f"{model_name}_{self._name}_mlx_r{int(round_num)}"
        hw = get_metal_device_name().strip() or "AppleSilicon"
        sol = Solution(
            name=sol_name,
            definition=self._name,
            author=str(model_name),
            spec=BuildSpec(
                language=SupportedLanguages.MLX,
                target_hardware=[hw],
                entry_point="submission.py::run",
            ),
            sources=[SourceFile(path="submission.py", content=code_text)],
            description=f"MLX Mamba selective scan fwd on Apple Silicon (round {round_num})",
        )
        self._solutions[sol_name] = sol
        return sol

    def code_for_world_model_from_raw(self, *, raw: Any, language: str) -> str:
        return str(raw or "")

    def seed_eval_for_base_solution(self, *, base_solution: Solution, config: Any = None) -> EvalResult:
        return self.run_benchmark(solution=base_solution, round_num=0)

    def run_benchmark(
        self,
        *,
        solution: Solution,
        config: Any = None,
        dump_traces: bool = False,
        round_num: int | None = None,
    ) -> EvalResult:
        code = self._extract_submission_code(solution)
        summary = evaluate_mlx_mamba_selective_scan_fwd_submission(
            submission_code=code,
            workloads=self._workloads,
            warmup_runs=self._cfg.warmup_runs,
            iterations=self._cfg.iterations,
            rtol=self._cfg.rtol,
            atol=self._cfg.atol,
            timeout_seconds=self._cfg.timeout_seconds,
            per_workload_timeout_seconds=self._cfg.per_workload_timeout_seconds,
            max_failure_excerpt_chars=self._cfg.max_failure_excerpt_chars,
        )

        self._last_round_passed_count = int(summary.passed_count)
        self._last_round_total_workloads = int(summary.total_count)
        self._last_round_trace_logs_for_prompt = self._format_trace_logs(summary)

        if summary.status == "passed":
            spd = summary.mean_vs_baseline_factor or 0.0
            self._last_round_summary_line = (
                f"PASSED {summary.passed_count}/{summary.total_count} workloads, median speedup {spd:.3f}×"
            )
        else:
            spd = summary.mean_vs_baseline_factor
            spd_str = f", median speedup {spd:.3f}× (on passed)" if spd is not None else ""
            self._last_round_summary_line = f"FAILED {summary.passed_count}/{summary.total_count} workloads{spd_str}"

        rn = f" (round {round_num})" if round_num is not None else ""
        print(f"[{self._name}]{rn} {self._last_round_summary_line}")
        if self._last_round_trace_logs_for_prompt.strip():
            print(self._last_round_trace_logs_for_prompt)

        sp = summary.mean_vs_baseline_factor or 0.0
        score = float(summary.passed_count) * 1000.0 + float(sp)

        metrics: dict[str, Any] = {
            "passed_count": int(summary.passed_count),
            "total_count": int(summary.total_count),
            "per_workload": summary.per_workload,
            "score_name": "speedup_vs_naive_selective_scan",
            "score": float(score),
        }

        return EvalResult(
            status=str(summary.status),
            latency_ms=summary.latency_ms,
            reference_latency_ms=summary.reference_latency_ms,
            mean_vs_baseline_factor=summary.mean_vs_baseline_factor,
            speedup_factor=summary.mean_vs_baseline_factor,
            log_excerpt=str(summary.log_excerpt or ""),
            metrics=metrics,
        )

    def get_config_for_logging(self) -> Dict[str, Any]:
        return {
            "name": self._name,
            "warmup_runs": self._cfg.warmup_runs,
            "iterations": self._cfg.iterations,
            "rtol": self._cfg.rtol,
            "atol": self._cfg.atol,
            "timeout_seconds": self._cfg.timeout_seconds,
            "per_workload_timeout_seconds": self._cfg.per_workload_timeout_seconds,
            "num_workloads": len(self._workloads),
        }

    def run_final_evaluation(
        self,
        *,
        solutions: list[Solution],
        config: Any = None,
        dump_traces: bool = False,
        workload_limit: int | None = None,
    ) -> dict[str, Any]:
        wls = self._workloads
        if workload_limit is not None:
            wls = wls[: int(workload_limit)]

        report: dict[str, Any] = {"solutions": [], "summary": {}}
        for sol in solutions:
            code = self._extract_submission_code(sol)
            summary = evaluate_mlx_mamba_selective_scan_fwd_submission(
                submission_code=code,
                workloads=wls,
                warmup_runs=self._cfg.warmup_runs,
                iterations=self._cfg.iterations,
                rtol=self._cfg.rtol,
                atol=self._cfg.atol,
                timeout_seconds=self._cfg.timeout_seconds,
                per_workload_timeout_seconds=self._cfg.per_workload_timeout_seconds,
                max_failure_excerpt_chars=self._cfg.max_failure_excerpt_chars,
            )
            sol_report = {
                "name": sol.name,
                "status": summary.status,
                "passed_count": summary.passed_count,
                "total_count": summary.total_count,
                "sub_median_ms": summary.latency_ms,
                "ref_median_ms": summary.reference_latency_ms,
                "median_speedup": summary.mean_vs_baseline_factor,
                "per_workload": summary.per_workload,
                "log_excerpt": summary.log_excerpt,
            }
            report["solutions"].append(sol_report)

        if report["solutions"]:
            best = max(
                (s for s in report["solutions"] if s.get("status") == "passed"),
                key=lambda s: float(s.get("median_speedup") or 0.0),
                default=None,
            )
            report["summary"] = {
                "best_solution": (best.get("name") if best else None),
                "best_median_speedup": (best.get("median_speedup") if best else None),
            }
        return report

    # --- prompt feedback hooks ---
    def get_last_round_trace_logs_for_prompt(self) -> str:
        return self._last_round_trace_logs_for_prompt

    def get_last_round_passed_count(self) -> int:
        return int(self._last_round_passed_count)

    def get_last_round_total_workloads(self) -> int:
        return int(self._last_round_total_workloads)

    def get_last_round_summary_line(self) -> str:
        return str(self._last_round_summary_line)

    # --- helpers ---
    def _extract_submission_code(self, solution: Solution) -> str:
        src = solution.get_entry_source()
        if src is not None:
            return str(src.content or "")
        return "\n\n".join(str(s.content or "") for s in (solution.sources or []))

    def _format_trace_logs(self, summary: Any) -> str:
        lines: list[str] = []
        lines.append(str(summary.log_excerpt or "").strip())
        try:
            per_wl = list(getattr(summary, "per_workload", []) or [])
            if per_wl:
                lines.append("\nPer-workload:")
                for w in per_wl:
                    if not isinstance(w, dict):
                        continue
                    label = str(w.get("label") or "")
                    st = str(w.get("status") or "")
                    sm = w.get("sub_median_ms")
                    sp = w.get("speedup")
                    pr = w.get("unified_memory_pressure")
                    row = f"- {label}: {st}"
                    if isinstance(sm, (int, float)):
                        row += f" | sub_ms={float(sm):.4f}"
                    if isinstance(sp, (int, float)):
                        row += f" | speedup={float(sp):.3f}x"
                    if isinstance(pr, (int, float)):
                        row += f" | pressure={float(pr):.6f}"

                    if st != "passed":
                        err = w.get("error")
                        if err is not None:
                            s = str(err).strip()
                            excerpt_len = int(os.getenv("KSEARCH_MLX_MAMBA_WORKLOAD_ERROR_EXCERPT", "500"))
                            if len(s) > excerpt_len:
                                s = s[-excerpt_len:]
                            s = " ".join(s.split())
                            if s:
                                row += f" | error={s}"
                        elif st == "wrong_answer":
                            row += (
                                f" | max_abs_diff={w.get('max_abs_diff')} | max_rel_diff={w.get('max_rel_diff')}"
                            )

                    lines.append(row)
        except Exception:
            pass
        return "\n".join([ln for ln in lines if ln.strip()])
