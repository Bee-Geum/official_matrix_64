"""MLX Mamba selective scan forward evaluator.

Runs a generated submission in an isolated subprocess and measures:
- correctness vs a naive MLX reference implementation
- latency (trimmed median ms)
- speedup vs naive baseline
- unified memory pressure (best-effort, Apple Silicon)

The submission must define `run(...)` and return an output with shape (B, D, L).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .spec import MAMBA_SELECTIVE_SCAN_FWD_WORKLOADS, MambaSelectiveScanFwdWorkload


@dataclass
class MlxMambaEvalSummary:
    status: str  # passed|failed|error
    latency_ms: Optional[float]
    reference_latency_ms: Optional[float]
    mean_vs_baseline_factor: Optional[float]
    log_excerpt: str
    per_workload: list[dict[str, Any]]
    passed_count: int
    total_count: int


_EVAL_HARNESS = textwrap.dedent(
    '''\
"""Isolated MLX Mamba selective scan fwd eval harness.

Usage: python harness.py <submission_py> <naive_py> <workloads_json> <config_json> <out_json>
"""

import importlib.util
import json
import subprocess
import sys
import time
import traceback


def _load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    return xs[n // 2] if (n % 2) else (xs[n // 2 - 1] + xs[n // 2]) / 2.0


def _trimmed_median(xs):
    if len(xs) < 4:
        return _median(xs)
    s = sorted(xs)
    n = len(s)
    q1 = s[n // 4]
    q3 = s[3 * n // 4]
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    f = [x for x in s if lo <= x <= hi]
    if not f:
        f = s
    return _median(f)


def _get_total_mem_bytes():
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], stderr=subprocess.DEVNULL)
        return int(out.decode().strip())
    except Exception:
        return None


def _unwrap_out(out):
    # Allow submissions to return (out, state) or similar.
    if isinstance(out, (tuple, list)) and out:
        return out[0]
    return out


def _run(submission_py: str, naive_py: str, workloads_json: str, config_json: str, out_json: str):
    with open(workloads_json) as f:
        workloads = json.load(f)
    with open(config_json) as f:
        cfg = json.load(f)

    warmup_runs = int(cfg.get("warmup_runs", 3))
    iterations = int(cfg.get("iterations", 10))
    rtol = float(cfg.get("rtol", 1e-3))
    atol = float(cfg.get("atol", 1e-3))

    try:
        import mlx.core as mx
    except Exception as exc:
        with open(out_json, "w") as f:
            json.dump({"error": f"mlx import failed: {exc}", "workloads": []}, f)
        sys.exit(1)

    try:
        naive_mod = _load_module(naive_py, "_naive")
        ref_fn = getattr(naive_mod, "run", None)
        if not callable(ref_fn):
            raise AttributeError("naive module must define run(...)")
    except Exception as exc:
        with open(out_json, "w") as f:
            json.dump({"error": f"naive import failed: {exc}\\n{traceback.format_exc()}", "workloads": []}, f)
        sys.exit(1)

    try:
        sub_mod = _load_module(submission_py, "_submission")
        run_fn = getattr(sub_mod, "run", None)
        if not callable(run_fn):
            raise AttributeError("submission must define a callable run(u,delta,A,B,C,...)")
    except Exception as exc:
        with open(out_json, "w") as f:
            json.dump({"error": f"submission import failed: {exc}\\n{traceback.format_exc()}", "workloads": []}, f)
        sys.exit(1)

    total_mem = _get_total_mem_bytes()
    wl_results = []

    for wl in workloads:
        B = int(wl["batch"])
        D = int(wl["dim"])
        N = int(wl["dstate"])
        L = int(wl["seqlen"])
        ngroups = int(wl.get("ngroups", 1))
        has_z = bool(wl.get("has_z", True))
        delta_softplus = bool(wl.get("delta_softplus", True))
        dtype_name = str(wl.get("dtype", "float32"))
        dtype = getattr(mx, dtype_name)
        label = wl.get("label") or f"B{B}_D{D}_N{N}_L{L}_G{ngroups}_{'z' if has_z else 'noz'}_{dtype_name}"

        # ---- correctness ----
        ok = True
        max_abs = 0.0
        mean_abs = 0.0
        max_rel = 0.0

        for seed in (42, 123, 7):
            mx.random.seed(seed)
            u = mx.random.normal([B, D, L]).astype(dtype)
            delta = mx.random.normal([B, D, L]).astype(dtype)
            A = -mx.abs(mx.random.normal([D, N]).astype(mx.float32))
            if ngroups > 1:
                B_var = mx.random.normal([B, ngroups, N, L]).astype(dtype)
                C_var = mx.random.normal([B, ngroups, N, L]).astype(dtype)
            else:
                B_var = mx.random.normal([B, N, L]).astype(dtype)
                C_var = mx.random.normal([B, N, L]).astype(dtype)
            D_skip = mx.random.normal([D]).astype(mx.float32)
            z = mx.random.normal([B, D, L]).astype(dtype) if has_z else None
            delta_bias = mx.random.normal([D]).astype(mx.float32)
            mx.eval(u, delta, A, B_var, C_var, D_skip, delta_bias)
            if z is not None:
                mx.eval(z)

            ref = ref_fn(u, delta, A, B_var, C_var, D=D_skip, z=z, delta_bias=delta_bias, delta_softplus=delta_softplus)
            ref = _unwrap_out(ref)
            mx.eval(ref)

            try:
                out = run_fn(u, delta, A, B_var, C_var, D=D_skip, z=z, delta_bias=delta_bias, delta_softplus=delta_softplus)
                out = _unwrap_out(out)
                mx.eval(out)
            except Exception:
                wl_results.append({"label": label, "workload": wl, "status": "error", "error": traceback.format_exc()})
                ok = False
                break

            if seed == 42 and list(out.shape) != list(ref.shape):
                wl_results.append({"label": label, "workload": wl, "status": "wrong_shape", "error": f"expected {ref.shape}, got {out.shape}"})
                ok = False
                break

            ref32 = ref.astype(mx.float32)
            out32 = out.astype(mx.float32)
            diff = mx.abs(ref32 - out32)
            _max_abs = float(mx.max(diff).item())
            _mean_abs = float(mx.mean(diff).item())
            _max_rel = float(mx.max(diff / (mx.abs(ref32) + 1e-8)).item())
            if _max_abs != _max_abs:
                _max_abs = 1e10
            if _max_rel != _max_rel:
                _max_rel = 1e10
            max_abs = max(max_abs, _max_abs)
            mean_abs = max(mean_abs, _mean_abs)
            max_rel = max(max_rel, _max_rel)

            passes = bool(mx.all(diff <= (atol + rtol * mx.abs(ref32))).item())
            if not passes:
                wl_results.append({
                    "label": label,
                    "workload": wl,
                    "status": "wrong_answer",
                    "max_abs_diff": max_abs,
                    "mean_abs_diff": mean_abs,
                    "max_rel_diff": max_rel,
                    "atol": atol,
                    "rtol": rtol,
                    "failed_seed": seed,
                })
                ok = False
                break

        if not ok:
            continue

        # ---- timing + memory ----
        try:
            mx.metal.reset_peak_memory()
        except Exception:
            pass
        try:
            mx.clear_cache()
        except Exception:
            pass
        try:
            mx.synchronize()
        except Exception:
            pass

        mx.random.seed(42)
        u = mx.random.normal([B, D, L]).astype(dtype)
        delta = mx.random.normal([B, D, L]).astype(dtype)
        A = -mx.abs(mx.random.normal([D, N]).astype(mx.float32))
        if ngroups > 1:
            B_var = mx.random.normal([B, ngroups, N, L]).astype(dtype)
            C_var = mx.random.normal([B, ngroups, N, L]).astype(dtype)
        else:
            B_var = mx.random.normal([B, N, L]).astype(dtype)
            C_var = mx.random.normal([B, N, L]).astype(dtype)
        D_skip = mx.random.normal([D]).astype(mx.float32)
        z = mx.random.normal([B, D, L]).astype(dtype) if has_z else None
        delta_bias = mx.random.normal([D]).astype(mx.float32)
        mx.eval(u, delta, A, B_var, C_var, D_skip, delta_bias)
        if z is not None:
            mx.eval(z)
        try:
            mx.synchronize()
        except Exception:
            pass

        # Warmup (interleaved) to stabilize caches.
        for _ in range(max(1, warmup_runs)):
            o = _unwrap_out(run_fn(u, delta, A, B_var, C_var, D=D_skip, z=z, delta_bias=delta_bias, delta_softplus=delta_softplus))
            mx.eval(o)
            r = _unwrap_out(ref_fn(u, delta, A, B_var, C_var, D=D_skip, z=z, delta_bias=delta_bias, delta_softplus=delta_softplus))
            mx.eval(r)
        try:
            mx.synchronize()
        except Exception:
            pass

        sub_times = []
        ref_times = []
        for _ in range(max(1, iterations)):
            try:
                mx.synchronize()
            except Exception:
                pass
            t0 = time.perf_counter()
            o = _unwrap_out(run_fn(u, delta, A, B_var, C_var, D=D_skip, z=z, delta_bias=delta_bias, delta_softplus=delta_softplus))
            mx.eval(o)
            try:
                mx.synchronize()
            except Exception:
                pass
            sub_times.append((time.perf_counter() - t0) * 1000.0)

            try:
                mx.synchronize()
            except Exception:
                pass
            t0 = time.perf_counter()
            r = _unwrap_out(ref_fn(u, delta, A, B_var, C_var, D=D_skip, z=z, delta_bias=delta_bias, delta_softplus=delta_softplus))
            mx.eval(r)
            try:
                mx.synchronize()
            except Exception:
                pass
            ref_times.append((time.perf_counter() - t0) * 1000.0)

        sub_med = _trimmed_median(sub_times)
        ref_med = _trimmed_median(ref_times)
        speedup = (ref_med / sub_med) if (sub_med and ref_med and sub_med > 0) else None

        active_b = None
        peak_b = None
        cache_b = None
        try:
            active_b = int(mx.metal.get_active_memory())
        except Exception:
            pass
        try:
            peak_b = int(mx.metal.get_peak_memory())
        except Exception:
            pass
        try:
            cache_b = int(mx.metal.get_cache_memory())
        except Exception:
            pass

        pressure = None
        if peak_b is not None and total_mem:
            pressure = float(peak_b) / float(total_mem)

        wl_results.append({
            "label": label,
            "workload": wl,
            "status": "passed",
            "sub_median_ms": sub_med,
            "ref_median_ms": ref_med,
            "speedup": speedup,
            "active_mem_bytes": active_b,
            "peak_mem_bytes": peak_b,
            "cache_mem_bytes": cache_b,
            "unified_memory_pressure": pressure,
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "max_rel_diff": max_rel,
        })

    with open(out_json, "w") as f:
        json.dump({"workloads": wl_results, "total_mem_bytes": total_mem}, f)


if __name__ == "__main__":
    submission_py, naive_py, workloads_json, config_json, out_json = sys.argv[1:6]
    _run(submission_py, naive_py, workloads_json, config_json, out_json)
'''
)


def evaluate_mlx_mamba_selective_scan_fwd_submission(
    *,
    submission_code: str,
    workloads: list[MambaSelectiveScanFwdWorkload] | None = None,
    warmup_runs: int = 3,
    iterations: int = 10,
    rtol: float = 1e-3,
    atol: float = 1e-3,
    timeout_seconds: int = 300,
    per_workload_timeout_seconds: int | None = None,
    max_failure_excerpt_chars: int = 4000,
) -> MlxMambaEvalSummary:
    """Evaluate a submission across workloads.

    `timeout_seconds` is a suite-level timeout budget. To avoid losing partial
    results when a long workload stalls, we run each workload in its own
    subprocess and record `timeout` for the ones that exceed the budget.
    """

    if workloads is None:
        workloads = list(MAMBA_SELECTIVE_SCAN_FWD_WORKLOADS)

    # Load naive implementation source from this package.
    naive_src_path = Path(__file__).resolve().parent / "naive_selective_scan_fwd.py"
    naive_src = naive_src_path.read_text(encoding="utf-8")

    per_wl: list[dict[str, Any]] = []
    last_stdout: str = ""
    last_stderr: str = ""

    def _wl_to_dict(wl: MambaSelectiveScanFwdWorkload) -> dict[str, Any]:
        return {
            "label": wl.label,
            "batch": int(wl.batch),
            "dim": int(wl.dim),
            "dstate": int(wl.dstate),
            "seqlen": int(wl.seqlen),
            "ngroups": int(wl.ngroups),
            "has_z": bool(wl.has_z),
            "delta_softplus": bool(wl.delta_softplus),
            "dtype": str(wl.dtype),
        }

    suite_deadline: float | None
    if timeout_seconds and int(timeout_seconds) > 0:
        suite_deadline = time.monotonic() + float(timeout_seconds)
    else:
        suite_deadline = None

    per_wl_cap = (
        float(per_workload_timeout_seconds)
        if per_workload_timeout_seconds is not None and int(per_workload_timeout_seconds) > 0
        else None
    )

    with tempfile.TemporaryDirectory(prefix="ksearch_mlx_mamba_") as td:
        td_path = Path(td)
        submission_py = td_path / "submission.py"
        naive_py = td_path / "naive.py"
        harness_py = td_path / "harness.py"
        workloads_json = td_path / "workloads.json"
        config_json = td_path / "config.json"
        out_json = td_path / "out.json"

        submission_py.write_text(str(submission_code or ""), encoding="utf-8")
        naive_py.write_text(str(naive_src or ""), encoding="utf-8")
        harness_py.write_text(_EVAL_HARNESS, encoding="utf-8")

        config_json.write_text(
            json.dumps(
                {
                    "warmup_runs": int(warmup_runs),
                    "iterations": int(iterations),
                    "rtol": float(rtol),
                    "atol": float(atol),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        cmd = [
            sys.executable,
            str(harness_py),
            str(submission_py),
            str(naive_py),
            str(workloads_json),
            str(config_json),
            str(out_json),
        ]

        for wl in workloads:
            wl_d = _wl_to_dict(wl)

            # Enforce suite-level budget while still reporting earlier workloads.
            if suite_deadline is not None:
                remaining = suite_deadline - time.monotonic()
                if remaining <= 0:
                    per_wl.append(
                        {
                            "label": wl_d["label"],
                            "workload": wl_d,
                            "status": "timeout",
                            "error": f"Suite timeout exceeded ({int(timeout_seconds)}s)",
                        }
                    )
                    continue
                wl_timeout = remaining
            else:
                wl_timeout = None

            if per_wl_cap is not None:
                wl_timeout = min(wl_timeout, per_wl_cap) if wl_timeout is not None else per_wl_cap

            # Write a single-workload file so a timeout doesn't discard partial results.
            workloads_json.write_text(json.dumps([wl_d], indent=2), encoding="utf-8")
            try:
                out_json.unlink()
            except FileNotFoundError:
                pass

            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=float(wl_timeout) if wl_timeout is not None else None,
                    env={**os.environ, "PYTHONPATH": ""},
                )
                last_stdout = proc.stdout or ""
                last_stderr = proc.stderr or ""
            except subprocess.TimeoutExpired:
                per_wl.append(
                    {
                        "label": wl_d["label"],
                        "workload": wl_d,
                        "status": "timeout",
                        "error": f"Timed out after {int(wl_timeout) if wl_timeout is not None else int(timeout_seconds)}s",
                    }
                )
                continue

            # Parse harness JSON for this workload.
            try:
                payload = json.loads(out_json.read_text(encoding="utf-8"))
            except Exception:
                combined = (last_stdout + "\n" + last_stderr)[-max_failure_excerpt_chars:]
                per_wl.append(
                    {
                        "label": wl_d["label"],
                        "workload": wl_d,
                        "status": "error",
                        "error": f"Harness failed to produce valid JSON.\n{combined}",
                    }
                )
                continue

            if "error" in payload:
                msg = str(payload.get("error") or "")
                combined = (msg + "\n" + last_stdout + "\n" + last_stderr)[-max_failure_excerpt_chars:]
                per_wl.append(
                    {
                        "label": wl_d["label"],
                        "workload": wl_d,
                        "status": "error",
                        "error": combined,
                    }
                )
                # Global error (e.g. import failed); no point continuing.
                break

            wl_results = list(payload.get("workloads") or [])
            if not wl_results:
                per_wl.append(
                    {
                        "label": wl_d["label"],
                        "workload": wl_d,
                        "status": "error",
                        "error": "Harness returned no workload results.",
                    }
                )
            else:
                per_wl.extend(wl_results)

    total_count = len(workloads)
    passed = [w for w in per_wl if str(w.get("status")) == "passed"]
    passed_count = len(passed)

    def _med(vals: list[float]) -> Optional[float]:
        if not vals:
            return None
        s = sorted(vals)
        n = len(s)
        return s[n // 2] if (n % 2) else (s[n // 2 - 1] + s[n // 2]) / 2.0

    sub_meds = [float(w.get("sub_median_ms")) for w in passed if isinstance(w.get("sub_median_ms"), (int, float))]
    ref_meds = [float(w.get("ref_median_ms")) for w in passed if isinstance(w.get("ref_median_ms"), (int, float))]
    speedups = [float(w.get("speedup")) for w in passed if isinstance(w.get("speedup"), (int, float))]

    latency_ms = _med(sub_meds)
    ref_latency_ms = _med(ref_meds)
    mean_vs = _med(speedups)

    has_global_error = any(str(w.get("status")) == "error" for w in per_wl)
    status = "passed" if (passed_count == total_count and total_count > 0) else ("error" if has_global_error and passed_count == 0 else "failed")

    timeout_count = sum(1 for w in per_wl if str(w.get("status")) == "timeout")

    excerpt_lines: list[str] = []
    excerpt_lines.append(
        f"MLX Mamba selective scan fwd eval: {status} ({passed_count}/{total_count} workloads passed)"
    )
    if timeout_count:
        excerpt_lines.append(f"timed_out_workloads={timeout_count}")
    if latency_ms is not None:
        excerpt_lines.append(f"submission_median_ms={latency_ms:.4f}")
    if ref_latency_ms is not None:
        excerpt_lines.append(f"baseline_median_ms={ref_latency_ms:.4f}")
    if mean_vs is not None:
        excerpt_lines.append(f"median_speedup_vs_baseline={mean_vs:.3f}x")

    # memory stats (max pressure over passed)
    pressure = None
    if passed:
        pressures = [w.get("unified_memory_pressure") for w in passed if isinstance(w.get("unified_memory_pressure"), (int, float))]
        pressure = max(pressures) if pressures else None
    if pressure is not None:
        excerpt_lines.append(f"unified_memory_pressure={float(pressure):.6f}")

    if status != "passed":
        tail = (last_stdout + "\n" + last_stderr).strip()
        if tail:
            excerpt_lines.append("--- last workload stdout/stderr (tail) ---")
            excerpt_lines.append(tail[-max_failure_excerpt_chars:])

    log_excerpt = "\n".join(excerpt_lines)[-max_failure_excerpt_chars:]

    return MlxMambaEvalSummary(
        status=status,
        latency_ms=latency_ms,
        reference_latency_ms=ref_latency_ms,
        mean_vs_baseline_factor=mean_vs,
        log_excerpt=log_excerpt,
        per_workload=per_wl,
        passed_count=passed_count,
        total_count=total_count,
    )
