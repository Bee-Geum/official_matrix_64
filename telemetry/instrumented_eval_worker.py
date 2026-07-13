#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_tensor(value: Any):
    import torch
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return first_tensor(item)
            except TypeError:
                pass
    if isinstance(value, dict):
        for item in value.values():
            try:
                return first_tensor(item)
            except TypeError:
                pass
    raise TypeError("Model forward did not return Tensor")


def move_inputs(values, device):
    return [value.to(device) if hasattr(value, "to") else value for value in values]


def main(job_path: str):
    job = json.loads(Path(job_path).read_text())
    out_path = Path(job["out_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    verdict = {
        "compiled": False,
        "correct": False,
        "latency_ms": None,
        "ref_latency_ms": None,
        "speedup": None,
        "error": None,
        "stage_s": {
            "reference_import": 0.0,
            "candidate_compile_import": 0.0,
            "model_init": 0.0,
            "correctness": 0.0,
            "reference_benchmark": 0.0,
            "candidate_benchmark": 0.0,
            "total": 0.0,
        },
        "correctness_trials_requested": int(job.get("trials", 5)),
        "correctness_trials_executed": 0,
        "warmup": int(job.get("warmup", 10)),
        "timing_iters": int(job.get("timing_iters", 100)),
        "compile_attempts": 1,
        "profile_runs": 0,
    }
    total_start = time.perf_counter()

    try:
        import torch
        torch.manual_seed(0)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        start = time.perf_counter()
        reference_module = load_module(job["ref_path"], "ref_mod")
        verdict["stage_s"]["reference_import"] = time.perf_counter() - start

        start = time.perf_counter()
        try:
            candidate_module = load_module(job["cand_path"], "cand_mod")
            verdict["compiled"] = True
        except Exception:
            verdict["stage_s"]["candidate_compile_import"] = time.perf_counter() - start
            verdict["error"] = "compile: " + traceback.format_exc(limit=8)
            verdict["stage_s"]["total"] = time.perf_counter() - total_start
            out_path.write_text(json.dumps(verdict, indent=2, ensure_ascii=False))
            return

        verdict["stage_s"]["candidate_compile_import"] = time.perf_counter() - start

        if not hasattr(candidate_module, "ModelNew"):
            verdict["error"] = "interface: ModelNew missing"
            verdict["stage_s"]["total"] = time.perf_counter() - total_start
            out_path.write_text(json.dumps(verdict, indent=2, ensure_ascii=False))
            return

        start = time.perf_counter()
        init_inputs = reference_module.get_init_inputs()
        reference_model = reference_module.Model(*init_inputs).to(device).eval()
        candidate_model = candidate_module.ModelNew(*init_inputs).to(device).eval()
        verdict["stage_s"]["model_init"] = time.perf_counter() - start

        start = time.perf_counter()
        with torch.no_grad():
            for trial in range(int(job.get("trials", 5))):
                torch.manual_seed(int(job.get("input_seed_base", 100)) + trial)
                inputs = move_inputs(reference_module.get_inputs(), device)
                reference_output = first_tensor(reference_model(*inputs)).contiguous()
                candidate_output = first_tensor(candidate_model(*inputs)).contiguous()
                verdict["correctness_trials_executed"] += 1

                if candidate_output.shape != reference_output.shape:
                    verdict["error"] = (
                        f"correctness: shape mismatch "
                        f"{tuple(candidate_output.shape)} != {tuple(reference_output.shape)}"
                    )
                    verdict["stage_s"]["correctness"] = time.perf_counter() - start
                    verdict["stage_s"]["total"] = time.perf_counter() - total_start
                    out_path.write_text(json.dumps(verdict, indent=2, ensure_ascii=False))
                    return

                if not torch.allclose(
                    candidate_output,
                    reference_output,
                    atol=float(job.get("atol", 1e-2)),
                    rtol=float(job.get("rtol", 1e-2)),
                ):
                    diff = (candidate_output - reference_output).abs()
                    verdict["error"] = (
                        f"correctness: trial {trial} mismatch "
                        f"max_abs={diff.max().item():.6e} "
                        f"mean_abs={diff.mean().item():.6e}"
                    )
                    verdict["stage_s"]["correctness"] = time.perf_counter() - start
                    verdict["stage_s"]["total"] = time.perf_counter() - total_start
                    out_path.write_text(json.dumps(verdict, indent=2, ensure_ascii=False))
                    return

        verdict["correct"] = True
        verdict["stage_s"]["correctness"] = time.perf_counter() - start

        def time_model(model, stage_name: str):
            torch.manual_seed(int(job.get("input_seed_base", 100)))
            inputs = move_inputs(reference_module.get_inputs(), device)
            stage_start = time.perf_counter()
            with torch.no_grad():
                for _ in range(int(job.get("warmup", 10))):
                    model(*inputs)
                if device == "cuda":
                    torch.cuda.synchronize()

                samples = []
                for _ in range(int(job.get("timing_iters", 100))):
                    if device == "cuda":
                        begin = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        begin.record()
                        model(*inputs)
                        end.record()
                        torch.cuda.synchronize()
                        samples.append(float(begin.elapsed_time(end)))
                    else:
                        s = time.perf_counter()
                        model(*inputs)
                        samples.append((time.perf_counter() - s) * 1000.0)
            verdict["stage_s"][stage_name] = time.perf_counter() - stage_start
            verdict["profile_runs"] += int(job.get("warmup", 10)) + int(job.get("timing_iters", 100))
            return statistics.median(samples), samples

        verdict["ref_latency_ms"], ref_samples = time_model(reference_model, "reference_benchmark")
        verdict["latency_ms"], cand_samples = time_model(candidate_model, "candidate_benchmark")
        verdict["speedup"] = (
            verdict["ref_latency_ms"] / verdict["latency_ms"]
            if verdict["latency_ms"] and verdict["latency_ms"] > 0
            else None
        )
        verdict["ref_latency_samples_ms"] = ref_samples
        verdict["latency_samples_ms"] = cand_samples

    except Exception:
        verdict["error"] = "runtime: " + traceback.format_exc(limit=8)

    verdict["stage_s"]["total"] = time.perf_counter() - total_start
    out_path.write_text(json.dumps(verdict, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
