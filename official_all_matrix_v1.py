#!/usr/bin/env python3
"""
Strict all-agents x official-benchmarks runner for unified_bench.

It bypasses the old native benchmark task discovery/evaluation path that
accepted README.md, .pre-commit-config.yaml, fixture files, or placeholder
summary directories as successful benchmark evidence.

A task is counted as official only when:
  1. it was discovered from the benchmark's official dataset/repository,
  2. the selected agent produced at least one candidate,
  3. the upstream evaluator was actually invoked, and
  4. a structured evaluator result artifact was produced and parsed.

Candidate correctness is separate from evaluator success. A wrong kernel may
have official_eval=1 and correct=0, which is the intended behavior.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Optional


BENCHMARKS = [
    "kernelbench",
    "robust_kbench",
    "tritonbench_t",
    "tritonbench_g",
    "multikernelbench",
    "backendbench",
    "pareval",
    "sol_execbench",
]

KB_BENCHMARKS = {"kernelbench", "robust_kbench"}

DEFAULT_AGENTS = [
    "cudaforge",
    "autokernel",
    "cuda_l1",
    "autotriton",
    "drkernel",
    "geak",
    "ksearch",
    "cuda_agent",
    "kernelllm",
    "incoder32b",
    "kernelskill",
]

KNOWN_BACKENDBENCH_OPS = [
    "add", "mul", "sub", "div", "relu", "sigmoid", "tanh", "gelu",
    "matmul", "mm", "bmm", "sum", "mean", "softmax", "layer_norm",
]


@dataclasses.dataclass
class Agent:
    name: str
    model: str
    langs: list[str]
    driver: Path
    out_flag: str
    gen_args: str
    glob: str


@dataclasses.dataclass
class Task:
    benchmark: str
    task_id: str
    source_path: Optional[Path]
    manifest_path: Path
    prompt_text: str
    metadata: dict[str, Any]


@dataclasses.dataclass
class CommandResult:
    rc: int
    stdout: str
    stderr: str
    wall_s: float
    log: str
    cmd: list[str]


def now_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def slug(value: str, max_len: int = 120) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
    return (value or "task")[:max_len]


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(errors="ignore"))
    except Exception:
        return default


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), errors="ignore")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    lines = [
        line for line in path.read_text(errors="ignore").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return list(csv.DictReader(lines)) if lines else []


def run_cmd(
    cmd: list[str],
    cwd: Path,
    log_path: Path,
    timeout: int,
    env: Optional[dict[str, str]] = None,
) -> CommandResult:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_env = os.environ.copy()
    if env:
        full_env.update({key: str(value) for key, value in env.items()})
    start = time.time()
    try:
        proc = subprocess.run(
            cmd, cwd=str(cwd), env=full_env, capture_output=True, text=True,
            timeout=timeout,
        )
        wall = time.time() - start
        log_path.write_text(
            "$ " + " ".join(map(str, cmd))
            + f"\ncwd={cwd}\nreturncode={proc.returncode}\nwall_s={wall:.6f}\n"
            + "----- STDOUT -----\n" + proc.stdout
            + "\n----- STDERR -----\n" + proc.stderr + "\n",
            errors="ignore",
        )
        return CommandResult(proc.returncode, proc.stdout, proc.stderr, wall, str(log_path), [str(x) for x in cmd])
    except subprocess.TimeoutExpired as exc:
        wall = time.time() - start
        stdout = exc.stdout.decode(errors="ignore") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode(errors="ignore") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        log_path.write_text(
            "$ " + " ".join(map(str, cmd))
            + f"\ncwd={cwd}\nreturncode=124\nwall_s={wall:.6f}\nTIMEOUT\n"
            + "----- STDOUT -----\n" + stdout
            + "\n----- STDERR -----\n" + stderr + "\n",
            errors="ignore",
        )
        return CommandResult(124, stdout, stderr or "TIMEOUT", wall, str(log_path), [str(x) for x in cmd])
    except Exception as exc:
        wall = time.time() - start
        log_path.write_text(
            "$ " + " ".join(map(str, cmd))
            + f"\ncwd={cwd}\nreturncode=125\nERROR\n{exc!r}\n",
            errors="ignore",
        )
        return CommandResult(125, "", repr(exc), wall, str(log_path), [str(x) for x in cmd])


def list_agents(root: Path, selected: str) -> list[Agent]:
    registry = root / "unified_bench_ext/registry/agents.csv"
    rows = read_csv_rows(registry)
    by_name: dict[str, Agent] = {}
    for row in rows:
        name = row.get("agent", "").strip()
        driver = row.get("driver", "").strip()
        if not name or not driver:
            continue
        by_name[name] = Agent(
            name=name,
            model=row.get("model", "unknown").strip() or "unknown",
            langs=[x for x in row.get("langs", "").split("|") if x],
            driver=(root / driver).resolve(),
            out_flag=row.get("out_flag", "--out"),
            gen_args=row.get("gen_args", ""),
            glob=row.get("glob", "candidate_*.py") or "candidate_*.py",
        )
    names = (
        [x.strip() for x in selected.split(",") if x.strip()]
        if selected and selected != "all"
        else [name for name in DEFAULT_AGENTS if name in by_name]
    )
    missing = [name for name in names if name not in by_name]
    if missing:
        raise RuntimeError(f"Agents missing from registry: {missing}")
    return [by_name[name] for name in names]


def selected_benchmarks(value: str) -> list[str]:
    benches = (
        [x.strip() for x in value.split(",") if x.strip()]
        if value and value != "all" else BENCHMARKS[:]
    )
    unknown = [b for b in benches if b not in BENCHMARKS]
    if unknown:
        raise RuntimeError(f"Unsupported benchmark(s): {unknown}")
    return benches


def repo_candidates(root: Path, benchmark: str) -> list[Path]:
    mapping = {
        "kernelbench": [root / "third_party/KernelBench", root / "third_party/kernelbench"],
        "robust_kbench": [root / "third_party/robust-kbench", root / "third_party/robust_kbench"],
        "tritonbench_t": [root / "third_party/TritonBench"],
        "tritonbench_g": [root / "third_party/TritonBench"],
        "multikernelbench": [root / "third_party/MultiKernelBench"],
        "backendbench": [root / "third_party/BackendBench"],
        "pareval": [root / "third_party/ParEval"],
        "sol_execbench": [root / "third_party/SOL-ExecBench", root / "third_party/sol-execbench"],
    }
    return mapping.get(benchmark, [])


def repo_for(root: Path, benchmark: str) -> Optional[Path]:
    for path in repo_candidates(root, benchmark):
        if path.exists():
            return path.resolve()
    return None


def resolve_task_path(root: Path, raw: str) -> Optional[Path]:
    raw = raw.strip()
    if not raw or raw.startswith("#"):
        return None
    for path in [Path(raw), root / raw, root / "unified_bench_ext/task_lists" / raw]:
        if path.exists():
            return path.resolve()
    return None


def is_valid_python_task(path: Path) -> bool:
    if not path.is_file() or path.suffix != ".py":
        return False
    text = path.read_text(errors="ignore")[:12000]
    return "class Model" in text or "torch.nn.Module" in text or "def get_inputs" in text


def read_existing_tasklist(root: Path, names: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for name in names:
        for tasklist in [root / name, root / "unified_bench_ext/task_lists" / name]:
            if not tasklist.exists():
                continue
            for line in tasklist.read_text(errors="ignore").splitlines():
                path = resolve_task_path(root, line)
                if path and path not in paths:
                    paths.append(path)
    return paths


def discover_kernelbench_tasks(root: Path) -> list[Path]:
    existing = [
        path for path in read_existing_tasklist(root, ["kernelbench_all250.txt", "kernelbench_tiny5.txt"])
        if is_valid_python_task(path)
    ]
    if existing:
        return existing
    repo = repo_for(root, "kernelbench")
    if not repo:
        return []
    candidates: list[Path] = []
    for pattern in ["**/level1/*.py", "**/level2/*.py", "**/level3/*.py", "**/Level1/*.py", "**/Level2/*.py", "**/Level3/*.py"]:
        candidates.extend(repo.glob(pattern))
    return sorted({p.resolve() for p in candidates if is_valid_python_task(p)})


def discover_robust_tasks(root: Path) -> list[Path]:
    existing = [
        path for path in read_existing_tasklist(root, ["robust_kbench_l12.txt"])
        if is_valid_python_task(path)
    ]
    if existing:
        return existing
    # The upstream robust-kbench repo may be absent (e.g. trimmed portable bundle);
    # its l12 task set is drawn from KernelBench level1/level2, so fall back to the
    # bundled KernelBench repo when robust-kbench itself is not present.
    repo = repo_for(root, "robust_kbench") or repo_for(root, "kernelbench")
    if not repo:
        return []
    candidates: list[Path] = []
    for pattern in ["**/level1/*.py", "**/level2/*.py", "**/Level1/*.py", "**/Level2/*.py"]:
        candidates.extend(repo.glob(pattern))
    return sorted({p.resolve() for p in candidates if is_valid_python_task(p)})


def find_triton_track_dirs(repo: Path, benchmark: str) -> list[Path]:
    wanted = "t" if benchmark == "tritonbench_t" else "g"
    dirs: list[Path] = []
    for path in repo.rglob("*"):
        if not path.is_dir():
            continue
        low = path.name.lower().replace("-", "_")
        if wanted == "t" and ("tritonbench_t_v1" in low or low in {"tritonbench_t", "t"} or low.endswith("_t_v1")):
            dirs.append(path)
        if wanted == "g" and ("tritonbench_g_v1" in low or low in {"tritonbench_g", "g"} or low.endswith("_g_v1")):
            dirs.append(path)
    return sorted(set(dirs), key=lambda p: (len(p.parts), str(p)))


def discover_triton_tasks(root: Path, benchmark: str) -> list[Path]:
    repo = repo_for(root, benchmark)
    if not repo:
        return []
    candidates: list[Path] = []
    for directory in find_triton_track_dirs(repo, benchmark):
        candidates.extend(directory.glob("*.py"))
    if not candidates:
        for path in repo.rglob("*.py"):
            low = str(path).lower()
            if any(x in low for x in ["/eval/", "performance_metrics", "write_file.py", "multiprocess"]):
                continue
            if benchmark == "tritonbench_t" and re.search(r"(tritonbench[_-]?t|/t/)", low):
                candidates.append(path)
            if benchmark == "tritonbench_g" and re.search(r"(tritonbench[_-]?g|/g/)", low):
                candidates.append(path)
    def valid(path: Path) -> bool:
        text = path.read_text(errors="ignore")[:16000]
        return len(text.strip()) > 40 and ("triton" in text.lower() or "torch" in text.lower() or "def " in text)
    return sorted({p.resolve() for p in candidates if valid(p)})


def import_dataset_keys(repo: Path) -> tuple[list[str], dict[str, str]]:
    script = """
import json, sys
sys.path.insert(0, '.')
from dataset import dataset
payload = {}
for key, value in dataset.items():
    try:
        payload[str(key)] = repr(value)[:20000]
    except Exception:
        payload[str(key)] = '<unrepresentable>'
print(json.dumps(payload))
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=str(repo), capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        return [], {}
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
        return list(payload.keys()), payload
    except Exception:
        return [], {}


def exact_prompt_for_op(repo: Path, op: str) -> Optional[Path]:
    clean_op = op.lower()
    matches: list[Path] = []
    for path in repo.rglob("*"):
        if not path.is_file() or path.stat().st_size > 3_000_000:
            continue
        if path.suffix.lower() not in {".json", ".txt", ".md", ".py"}:
            continue
        low = str(path).lower()
        if any(x in low for x in ["ascend", "npu", "musa", "pallas", "sycl", "tilelang"]):
            continue
        stem_tokens = re.split(r"[^a-z0-9]+", path.stem.lower())
        if clean_op in stem_tokens or path.stem.lower() == clean_op:
            matches.append(path)
    if not matches:
        return None
    matches.sort(key=lambda p: (0 if "prompt" in str(p).lower() else 1, 0 if "cuda" in str(p).lower() else 1, len(str(p))))
    return matches[0]


def discover_multikernel_tasks(root: Path) -> list[dict[str, Any]]:
    repo = repo_for(root, "multikernelbench")
    if not repo:
        return []
    keys, reprs = import_dataset_keys(repo)
    tasks: list[dict[str, Any]] = []
    for op in keys:
        prompt_path = exact_prompt_for_op(repo, op)
        prompt_text = prompt_path.read_text(errors="ignore")[:40000] if prompt_path else ""
        if not prompt_text:
            prompt_text = f"MultiKernelBench operator: {op}\nOfficial dataset entry: {reprs.get(op, '')}\n"
        tasks.append({"op": op, "prompt_path": str(prompt_path) if prompt_path else "", "prompt_text": prompt_text})
    return tasks


def eager_backend_impl(op: str) -> str:
    expressions = {
        "add": "torch.add(args[0], args[1])", "mul": "torch.mul(args[0], args[1])",
        "sub": "torch.sub(args[0], args[1])", "div": "torch.div(args[0], args[1])",
        "relu": "torch.relu(args[0])", "sigmoid": "torch.sigmoid(args[0])",
        "tanh": "torch.tanh(args[0])", "gelu": "torch.nn.functional.gelu(args[0])",
        "matmul": "torch.matmul(args[0], args[1])", "mm": "torch.mm(args[0], args[1])",
        "bmm": "torch.bmm(args[0], args[1])", "sum": "torch.sum(args[0])",
        "mean": "torch.mean(args[0])", "softmax": "torch.softmax(args[0], dim=-1)",
        "layer_norm": "torch.nn.functional.layer_norm(args[0], args[0].shape[-1:])",
    }
    expression = expressions.get(op, "args[0]")
    return "import torch\n\n" + f"def {op}_kernel_impl(*args, **kwargs):\n    return {expression}\n"


def first_string(data: Any, keys: list[str]) -> str:
    if isinstance(data, dict):
        for key in keys:
            value = data.get(key)
            if isinstance(value, str) and value:
                return value
        for value in data.values():
            result = first_string(value, keys)
            if result:
                return result
    elif isinstance(data, list):
        for value in data:
            result = first_string(value, keys)
            if result:
                return result
    return ""


def normalize_backend_op(name: str) -> str:
    low = name.lower().replace("aten.", "").replace("torch.", "")
    low = low.replace("__tensor", "").replace("__scalar", "").split(".")[0]
    low = re.sub(r"[^a-z0-9_]+", "_", low).strip("_")
    for op in KNOWN_BACKENDBENCH_OPS:
        if low == op or low.startswith(op + "_") or op in low.split("_"):
            return op
    return low


def backendbench_probe_ops(root: Path, work_dir: Path) -> list[str]:
    repo = repo_for(root, "backendbench")
    if not repo:
        return []
    main = repo / "BackendBench/scripts/main.py"
    if not main.exists():
        return []
    ops_dir = work_dir / "backend_probe_ops"
    log_dir = work_dir / "backend_probe_results"
    shutil.rmtree(ops_dir, ignore_errors=True)
    shutil.rmtree(log_dir, ignore_errors=True)
    ops_dir.mkdir(parents=True, exist_ok=True)
    for op in KNOWN_BACKENDBENCH_OPS:
        directory = ops_dir / op
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{op}_implementation_1.py").write_text(eager_backend_impl(op), errors="ignore")
        (directory / "README.md").write_text(f"# {op}\n", errors="ignore")
    run_cmd(
        [sys.executable, str(main), "--suite", "smoke", "--backend", "directory", "--ops-directory", str(ops_dir), "--log-dir", str(log_dir)],
        repo, work_dir / "backend_probe.log.txt", timeout=1800,
    )
    found: list[str] = []
    full = read_json(log_dir / "full_results.json", [])
    if isinstance(full, list):
        for item in full:
            if not isinstance(item, dict):
                continue
            name = first_string(item, ["operator", "op", "name", "operator_name", "test_name"])
            if name:
                normalized = normalize_backend_op(name)
                if normalized and normalized not in found:
                    found.append(normalized)
    return found or KNOWN_BACKENDBENCH_OPS[:]


def load_pareval_prompts(repo: Path) -> list[dict[str, Any]]:
    candidates = [repo / "prompts/generation-prompts.json", repo / "prompts/generation_prompts.json"]
    candidates.extend(repo.rglob("*generation*prompts*.json"))
    for path in candidates:
        if not path.exists():
            continue
        data = read_json(path, None)
        if isinstance(data, list):
            prompts = data
        elif isinstance(data, dict):
            prompts = data.get("prompts") or data.get("items") or data.get("data") or []
        else:
            prompts = []
        selected: list[dict[str, Any]] = []
        for prompt in prompts:
            if not isinstance(prompt, dict):
                continue
            model = str(prompt.get("parallelism_model", "")).lower()
            if model and model != "cuda":
                continue
            item = dict(prompt)
            item["parallelism_model"] = "cuda"
            item["language"] = "cpp"
            item.pop("outputs", None)
            if "name" not in item:
                item["name"] = item.get("problem") or f"pareval_{len(selected):04d}"
            selected.append(item)
        if selected:
            return selected
    selected = []
    for path in sorted(repo.glob("prompts/raw/*/*/cuda")):
        parts = list(path.parts)
        idx = parts.index("raw") if "raw" in parts else -1
        problem_type = parts[idx + 1] if idx >= 0 and len(parts) > idx + 1 else "unknown"
        name = parts[idx + 2] if idx >= 0 and len(parts) > idx + 2 else path.parent.name
        selected.append({"name": name, "problem_type": problem_type, "parallelism_model": "cuda", "language": "cpp", "prompt": path.read_text(errors="ignore"), "_raw_path": str(path)})
    return selected


def discover_sol_problems(root: Path) -> list[Path]:
    repo = repo_for(root, "sol_execbench")
    if not repo:
        return []
    preferred_roots = [repo / "data/SOL-ExecBench/benchmark", repo / "data/benchmark", repo / "tests/sol_execbench/samples", repo / "examples"]
    found: list[Path] = []
    for base in preferred_roots:
        if not base.exists():
            continue
        for directory in base.rglob("*"):
            if directory.is_dir() and (directory / "definition.json").exists() and (directory / "workload.jsonl").exists():
                found.append(directory.resolve())
    if not found:
        for directory in repo.rglob("*"):
            if directory.is_dir() and (directory / "definition.json").exists() and (directory / "workload.jsonl").exists():
                found.append(directory.resolve())
    return sorted(set(found))


def task_prompt_header(benchmark: str, task_id: str) -> str:
    return f"Benchmark: {benchmark}\nOfficial task: {task_id}\nReturn only source code, without Markdown fences or explanations.\n\n"


def create_task_manifests(root: Path, benchmark: str, prepare_dir: Path) -> list[Task]:
    task_dir = prepare_dir / "tasks" / benchmark
    shutil.rmtree(task_dir, ignore_errors=True)
    task_dir.mkdir(parents=True, exist_ok=True)
    tasks: list[Task] = []
    if benchmark == "kernelbench":
        sources = discover_kernelbench_tasks(root)
        for index, source in enumerate(sources):
            task_id = source.stem
            manifest = task_dir / f"{index:04d}_{slug(task_id)}.json"
            payload = {"benchmark": benchmark, "task_id": task_id, "source_path": str(source), "official_task": True}
            write_json(manifest, payload)
            tasks.append(Task(benchmark, task_id, source, manifest, source.read_text(errors="ignore"), payload))
    elif benchmark == "robust_kbench":
        sources = discover_robust_tasks(root)
        for index, source in enumerate(sources):
            task_id = source.stem
            manifest = task_dir / f"{index:04d}_{slug(task_id)}.json"
            payload = {"benchmark": benchmark, "task_id": task_id, "source_path": str(source), "official_task": True}
            write_json(manifest, payload)
            tasks.append(Task(benchmark, task_id, source, manifest, source.read_text(errors="ignore"), payload))
    elif benchmark in {"tritonbench_t", "tritonbench_g"}:
        for index, source in enumerate(discover_triton_tasks(root, benchmark)):
            task_id = source.stem.replace("_perf", "")
            manifest = task_dir / f"{index:04d}_{slug(task_id)}.json"
            prompt = task_prompt_header(benchmark, task_id) + "Implement the official TritonBench task as a Python module and preserve its public interface.\n\n" + source.read_text(errors="ignore")[:60000]
            payload = {"benchmark": benchmark, "task_id": task_id, "source_path": str(source), "official_task": True, "prompt": prompt}
            write_json(manifest, payload)
            tasks.append(Task(benchmark, task_id, source, manifest, prompt, payload))
    elif benchmark == "multikernelbench":
        for index, item in enumerate(discover_multikernel_tasks(root)):
            op = item["op"]
            manifest = task_dir / f"{index:04d}_{slug(op)}.json"
            prompt = task_prompt_header(benchmark, op) + f"Generate a kernel response for the exact MultiKernelBench operator '{op}'. Use CUDA C++ or Triton and do not replace the computation with eager PyTorch.\n\n" + item["prompt_text"]
            payload = {"benchmark": benchmark, "task_id": op, "op": op, "official_task": True, "prompt_path": item.get("prompt_path", ""), "prompt": prompt}
            write_json(manifest, payload)
            tasks.append(Task(benchmark, op, None, manifest, prompt, payload))
    elif benchmark == "backendbench":
        ops = backendbench_probe_ops(root, prepare_dir / "backend_probe")
        for index, op in enumerate(ops):
            manifest = task_dir / f"{index:04d}_{slug(op)}.json"
            prompt = task_prompt_header(benchmark, op) + f"Write a Python module for BackendBench DirectoryBackend exposing `{op}_kernel_impl`. Use a custom Triton/CUDA/compiled implementation. Do not use the eager PyTorch operator as a fallback.\n"
            payload = {"benchmark": benchmark, "task_id": op, "op": op, "official_task": True, "smoke_ops": ops, "prompt": prompt}
            write_json(manifest, payload)
            tasks.append(Task(benchmark, op, None, manifest, prompt, payload))
    elif benchmark == "pareval":
        repo = repo_for(root, benchmark)
        prompts = load_pareval_prompts(repo) if repo else []
        for index, prompt_obj in enumerate(prompts):
            name = str(prompt_obj.get("name", f"pareval_{index:04d}"))
            manifest = task_dir / f"{index:04d}_{slug(name)}.json"
            pareval_body = str(prompt_obj.get("prompt") or prompt_obj.get("instruction") or prompt_obj.get("problem") or json.dumps(prompt_obj, ensure_ascii=False))
            # ParEval is a COMPLETION task: its driver compiles (prompt + "\n" + output),
            # where `prompt` already ends with an open function signature followed by '{'.
            # The candidate must therefore be ONLY the function body continuation. Asking
            # for a full source file makes agents re-declare headers/structs/signature,
            # which then fails to compile (duplicate definitions). Instruct accordingly.
            prompt_text = (
                task_prompt_header(benchmark, name)
                + "This is a ParEval CUDA COMPLETION task. Below is the BEGINNING of a "
                  "CUDA C++ file; its last line is a function signature ending with '{' "
                  "and the body is missing. Your output will be appended VERBATIM right "
                  "after it and compiled by ParEval's official C++ driver.\n\n"
                  "Output ONLY the code that completes that open function body, ending "
                  "with its closing '}'. Do NOT repeat anything shown below: no #include, "
                  "no struct/typedef, no 'using', no function signature, and no main(). "
                  "Emit raw C++ only (no Markdown fences, no prose).\n\n"
                  "--- code so far (context only; do NOT repeat it) ---\n"
                + pareval_body
            )
            payload = {"benchmark": benchmark, "task_id": name, "official_task": True, "prompt_obj": prompt_obj, "prompt": prompt_text}
            write_json(manifest, payload)
            tasks.append(Task(benchmark, name, None, manifest, prompt_text, payload))
    elif benchmark == "sol_execbench":
        for index, problem_dir in enumerate(discover_sol_problems(root)):
            definition = read_json(problem_dir / "definition.json", {})
            first_workload = ""
            workload_path = problem_dir / "workload.jsonl"
            if workload_path.exists():
                first_workload = next((line for line in workload_path.read_text(errors="ignore").splitlines() if line.strip()), "")
            definition_name = definition.get("name") if isinstance(definition, dict) else None
            task_id = str(definition_name or problem_dir.name)
            manifest = task_dir / f"{index:04d}_{slug(task_id)}.json"
            prompt = task_prompt_header(benchmark, task_id) + "Implement the SOL-ExecBench Definition below. Prefer a Python or Triton module with a callable `run`; CUDA C++ is also accepted. The adapter will package the source into the official Solution schema.\n\nDefinition:\n" + json.dumps(definition, indent=2, ensure_ascii=False) + "\n\nFirst workload:\n" + first_workload
            payload = {"benchmark": benchmark, "task_id": task_id, "official_task": True, "problem_dir": str(problem_dir), "definition": definition, "prompt": prompt}
            write_json(manifest, payload)
            tasks.append(Task(benchmark, task_id, problem_dir, manifest, prompt, payload))
    return tasks


def write_compat_tasklists(root: Path, tasks_by_benchmark: dict[str, list[Task]]) -> None:
    tasklist_dir = root / "unified_bench_ext/task_lists"
    tasklist_dir.mkdir(parents=True, exist_ok=True)
    timestamp = now_id()
    names = {
        "kernelbench": "kernelbench_all250.txt", "robust_kbench": "robust_kbench_l12.txt",
        "tritonbench_t": "tritonbench_t.txt", "tritonbench_g": "tritonbench_g.txt",
        "multikernelbench": "multikernelbench.txt", "backendbench": "backendbench_smoke.txt",
        "pareval": "pareval_cuda.txt", "sol_execbench": "sol_execbench.txt",
    }
    for benchmark, filename in names.items():
        tasks = tasks_by_benchmark.get(benchmark, [])
        lines = [str(task.source_path) for task in tasks if task.source_path] if benchmark in KB_BENCHMARKS else [str(task.manifest_path) for task in tasks]
        for path in [root / filename, tasklist_dir / filename]:
            if path.exists():
                backup = path.with_suffix(path.suffix + f".bak_official_{timestamp}")
                if not backup.exists():
                    shutil.copy2(path, backup)
            path.write_text("\n".join(lines) + ("\n" if lines else ""), errors="ignore")


def load_prepared_tasks(run_root: Path, benchmark: str) -> list[Task]:
    task_dir = run_root / "prepared/tasks" / benchmark
    tasks: list[Task] = []
    for manifest in sorted(task_dir.glob("*.json")):
        data = read_json(manifest, {})
        if not isinstance(data, dict) or not data.get("official_task"):
            continue
        source_path = data.get("source_path") or data.get("problem_dir")
        source = Path(source_path).resolve() if source_path and Path(source_path).exists() else None
        tasks.append(Task(benchmark, str(data.get("task_id", manifest.stem)), source, manifest, str(data.get("prompt", "")), data))
    return tasks


def prepare_all(root: Path, run_root: Path, benchmarks: list[str]) -> dict[str, Any]:
    prepared = run_root / "prepared"
    prepared.mkdir(parents=True, exist_ok=True)
    tasks_by_benchmark: dict[str, list[Task]] = {}
    report: dict[str, Any] = {"generated_at": iso_now(), "benchmarks": {}}
    for benchmark in benchmarks:
        tasks = create_task_manifests(root, benchmark, prepared)
        tasks_by_benchmark[benchmark] = tasks
        report["benchmarks"][benchmark] = {"task_count": len(tasks), "valid": int(bool(tasks)), "first_tasks": [task.task_id for task in tasks[:10]], "repo": str(repo_for(root, benchmark) or "")}
    write_compat_tasklists(root, tasks_by_benchmark)
    report["all_have_tasks"] = all(report["benchmarks"].get(b, {}).get("task_count", 0) > 0 for b in benchmarks)
    write_json(prepared / "prepare_report.json", report)
    return report


def extract_code(text: str) -> str:
    if not text:
        return ""
    blocks = re.findall(r"```(?:python|py|cuda|cpp|c\+\+|triton)?\s*\n(.*?)```", text, flags=re.I | re.S)
    if blocks:
        text = max(blocks, key=len)
    text = text.strip()
    # Some drivers' own extraction leaves a bare markdown language-tag line (e.g. a
    # lone "cpp") as the first line, which then breaks compilation. Drop it.
    first, sep, rest = text.partition("\n")
    if sep and first.strip().lower().lstrip("`") in {"cpp", "c++", "cuda", "c", "python", "py", "triton"}:
        text = rest.strip()
    return text.strip() + "\n"


def candidate_files(candidate_dir: Path, agent: Agent) -> list[Path]:
    patterns = [agent.glob, "candidate_*.py", "round*_kernel.py", "*.py", "*.cu", "*.cpp"]
    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in candidate_dir.rglob(pattern):
            if not path.is_file() or path in seen:
                continue
            low = path.name.lower()
            if any(x in low for x in ["raw_reply", "meta", "generation.out", "eval.out"]):
                continue
            if path.stat().st_size > 20_000_000:
                continue
            seen.add(path)
            found.append(path)
    found.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return found


def generate_candidates(root: Path, agent: Agent, task: Task, cell_dir: Path, args: argparse.Namespace) -> tuple[list[Path], dict[str, Any]]:
    candidate_dir = cell_dir / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = cell_dir / "official_prompt.txt"
    prompt_path.write_text(task.prompt_text, errors="ignore")
    task_argument = task.source_path if task.benchmark in KB_BENCHMARKS and task.source_path else prompt_path
    gpu_json = json.dumps({"name": os.environ.get("GPU_NAME", "NVIDIA RTX PRO 6000"), "index": args.gpu}, separators=(",", ":"))
    template = agent.gen_args or f"--task {{TASK}} {agent.out_flag} {{CAND}} --rounds {{ROUNDS}} --seed {{SEED}} --temperature {{TEMP}}"
    replacements = {"{TASK}": str(task_argument), "{CAND}": str(candidate_dir), "{ROUNDS}": str(args.rounds), "{SEED}": str(args.seed), "{TEMP}": str(args.temp), "{GPU_JSON}": gpu_json}
    rendered = template
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    cmd = [sys.executable, str(agent.driver)] + shlex.split(rendered)
    env = {"CUDA_VISIBLE_DEVICES": str(args.gpu), "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL", f"http://127.0.0.1:{args.port}/v1"), "PYTHONPATH": f"{root}:{os.environ.get('PYTHONPATH', '')}", "PYTHONUNBUFFERED": "1"}
    result = run_cmd(cmd, root, cell_dir / "generation.log.txt", timeout=args.generation_timeout, env=env)
    files = candidate_files(candidate_dir, agent)
    normalized: list[Path] = []
    for index, path in enumerate(files[:args.max_candidates]):
        raw = path.read_text(errors="ignore")
        code = extract_code(raw)
        suffix = path.suffix if path.suffix in {".py", ".cu", ".cpp"} else ".py"
        normalized_path = candidate_dir / f"official_candidate_{index:03d}{suffix}"
        normalized_path.write_text(code or raw, errors="ignore")
        normalized.append(normalized_path)
    report = {"agent": agent.name, "driver": str(agent.driver), "cmd": result.cmd, "returncode": result.rc, "wall_s": result.wall_s, "log": result.log, "candidate_count": len(normalized), "raw_candidates": [str(path) for path in files], "normalized_candidates": [str(path) for path in normalized]}
    write_json(cell_dir / "generation_report.json", report)
    return normalized, report


def recursive_values(data: Any, wanted_keys: set[str]) -> list[Any]:
    found: list[Any] = []
    if isinstance(data, dict):
        for key, value in data.items():
            if key.lower() in wanted_keys:
                found.append(value)
            found.extend(recursive_values(value, wanted_keys))
    elif isinstance(data, list):
        for item in data:
            found.extend(recursive_values(item, wanted_keys))
    return found


def parse_bool(data: Any, keys: Iterable[str]) -> Optional[bool]:
    for value in recursive_values(data, {key.lower() for key in keys}):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            low = value.strip().lower()
            if low in {"true", "yes", "pass", "passed", "success", "correct", "ok"}:
                return True
            if low in {"false", "no", "fail", "failed", "incorrect", "error"}:
                return False
    return None


def parse_number(data: Any, keys: Iterable[str]) -> Optional[float]:
    for value in recursive_values(data, {key.lower() for key in keys}):
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
        if isinstance(value, str):
            match = re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?", value)
            if match:
                try:
                    return float(match.group(0))
                except Exception:
                    pass
    return None


def result_base(benchmark: str, task: Task, candidate: Path) -> dict[str, Any]:
    return {"benchmark": benchmark, "task_id": task.task_id, "task_manifest": str(task.manifest_path), "candidate": str(candidate), "official_task": 1, "official_attempted": 0, "official_eval": 0, "correct": 0, "performance_available": 0, "score": 0.0, "evaluator_rc": "", "infrastructure_error": "", "candidate_error": "", "logs": [], "artifacts": [], "wall_s": 0.0}


def find_summary_file(task_dir: Path) -> Optional[Path]:
    preferred = task_dir / "final_eval/summary.json"
    if preferred.is_file():
        return preferred
    candidates = [p for p in task_dir.rglob("summary.json") if p.is_file() and "candidates" not in str(p)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def backup_existing_run(root: Path, run_pattern: str, backup_root: Path) -> None:
    for path in root.glob(run_pattern):
        if not path.is_dir():
            continue
        target = backup_root / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        shutil.move(str(path), str(target))


def first_candidate_in_task_dir(task_dir: Path) -> Optional[Path]:
    for pattern in ["candidates/official_candidate_*", "candidates/candidate_*.py", "candidates/round*_kernel.py"]:
        matches = list(task_dir.glob(pattern))
        if matches:
            return sorted(matches)[0]
    return None


def evaluate_kb_cell(root: Path, run_root: Path, agent: Agent, benchmark: str, tasks: list[Task], args: argparse.Namespace, cell_dir: Path) -> list[dict[str, Any]]:
    runner = root / "run_ext_rtx6000.sh"
    if not runner.exists():
        return [{**result_base(benchmark, task, Path("")), "infrastructure_error": "run_ext_rtx6000.sh missing"} for task in tasks]
    backup_root = run_root / "backups" / now_id() / benchmark / agent.name
    label = "kernelbench_all250" if benchmark == "kernelbench" else "robust_kbench_l12"
    pattern = f"runs/{agent.name}_{agent.model}_{label}_round{args.rounds}_repeat{args.repeat}_temp{args.temp}*"
    backup_existing_run(root, pattern, backup_root)
    env = {"CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONPATH": f"{root}:{os.environ.get('PYTHONPATH', '')}", "OPENAI_BASE_URL": os.environ.get("OPENAI_BASE_URL", f"http://127.0.0.1:{args.port}/v1"), "FORCE": "1", "PYTHONUNBUFFERED": "1"}
    cmd = ["bash", str(runner), "matrix", "--benchmark", benchmark, "--agent", agent.name, "--rounds", str(args.rounds), "--repeat", str(args.repeat), "--temp", str(args.temp), "--timeout", str(args.eval_timeout), "--limit", str(len(tasks))]
    result = run_cmd(cmd, root, cell_dir / "kb_matrix.log.txt", timeout=args.cell_timeout, env=env)
    run_dirs = [path for path in (root / "runs").glob(f"{agent.name}_{agent.model}_{label}_round{args.rounds}_repeat{args.repeat}_temp{args.temp}*") if path.is_dir()]
    rows: list[dict[str, Any]] = []
    for task in tasks:
        row = result_base(benchmark, task, Path(""))
        row["official_attempted"] = 1
        row["evaluator_rc"] = result.rc
        row["logs"] = [result.log]
        matched_dirs: list[Path] = []
        for run_dir in run_dirs:
            for task_dir in run_dir.iterdir():
                if task_dir.is_dir() and slug(task.task_id).lower() in slug(task_dir.name).lower():
                    matched_dirs.append(task_dir)
        if not matched_dirs:
            row["infrastructure_error"] = "official task run directory not found"
            rows.append(row)
            continue
        task_dir = max(matched_dirs, key=lambda p: p.stat().st_mtime)
        summary_path = find_summary_file(task_dir)
        if not summary_path:
            row["infrastructure_error"] = "instrumented evaluator summary.json not found"
            rows.append(row)
            continue
        summary = read_json(summary_path, None)
        item = summary[0] if isinstance(summary, list) and summary else summary
        if not isinstance(item, dict):
            row["infrastructure_error"] = "unexpected KernelBench summary schema"
            rows.append(row)
            continue
        n_candidates = int(float(item.get("n_candidates", 0) or 0))
        n_correct = int(float(item.get("n_correct", 0) or 0))
        score = float(item.get("best_score", item.get("geomean_speedup", 0)) or 0)
        row.update({"candidate": str(first_candidate_in_task_dir(task_dir) or ""), "official_eval": int(n_candidates > 0), "correct": int(n_correct > 0), "performance_available": int(score > 0), "score": score, "artifacts": [str(summary_path)]})
        if not row["official_eval"]:
            row["candidate_error"] = str(item.get("error") or item.get("blocked_reason") or "")
        rows.append(row)
    return rows


def patch_triton_repo(repo: Path) -> list[str]:
    patched: list[str] = []
    old = 'print(f"Above is call test for {path.split("/")[-1].replace(".jsonl", "")}")'
    new = 'print(f"Above is call test for {path.split(chr(47))[-1].replace(chr(46)+\'jsonl\', \'\')}")'
    eval_root = repo / "EVAL"
    if eval_root.exists():
        for path in eval_root.rglob("0_call_acc.py"):
            text = path.read_text(errors="ignore")
            if old in text:
                backup = path.with_suffix(path.suffix + ".bak_official_matrix")
                if not backup.exists():
                    backup.write_text(text, errors="ignore")
                path.write_text(text.replace(old, new), errors="ignore")
                patched.append(f"{path}:fstring")
        hardcoded = "/home/lijianling/miniconda3/envs/LLM/bin/python"
        for path in eval_root.rglob("*.py"):
            text = path.read_text(errors="ignore")
            if hardcoded not in text:
                continue
            backup = path.with_suffix(path.suffix + ".bak_official_matrix")
            if not backup.exists():
                backup.write_text(text, errors="ignore")
            text = text.replace(f'"{hardcoded}"', "sys.executable").replace(f"'{hardcoded}'", "sys.executable")
            if "sys.executable" in text and "import sys" not in text:
                text = "import sys\n" + text
            path.write_text(text, errors="ignore")
            patched.append(f"{path}:python_path")
    return patched


def triton_eval_dirs(repo: Path, benchmark: str) -> tuple[Path, Path]:
    suffix = "T" if benchmark == "tritonbench_t" else "G"
    return repo / "EVAL" / f"eval_{suffix}", repo / "performance_metrics" / f"perf_{suffix}"


def clean_directory(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def parse_speed(text: str) -> Optional[float]:
    for pattern in [r"speed\s*up\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", r"speedup\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", r"efficiency\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", r"score\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"]:
        match = re.search(pattern, text, re.I)
        if match:
            return float(match.group(1))
    return None


def evaluate_triton(root: Path, benchmark: str, task: Task, candidate: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    row = result_base(benchmark, task, candidate)
    repo = repo_for(root, benchmark)
    if not repo:
        row["infrastructure_error"] = "TritonBench repo missing"
        return row
    patch_triton_repo(repo)
    eval_dir, perf_dir = triton_eval_dirs(repo, benchmark)
    call_script, exe_script, efficiency_script = eval_dir / "0_call_acc.py", eval_dir / "1_exe_acc.py", eval_dir / "2_efficiency.py"
    if not call_script.exists() or not exe_script.exists():
        row["infrastructure_error"] = "TritonBench official correctness scripts missing"
        return row
    if not task.source_path or not task.source_path.exists():
        row["infrastructure_error"] = "TritonBench official source task missing"
        return row
    start = time.time()
    stem = task.task_id
    stage = out_dir / "stage"
    source_dir, target_dir = stage / "source", stage / "target"
    clean_directory(source_dir); clean_directory(target_dir)
    source_file, target_file = source_dir / f"{stem}.py", target_dir / f"{stem}.py"
    source_file.write_text(task.source_path.read_text(errors="ignore"), errors="ignore")
    target_file.write_text(candidate.read_text(errors="ignore"), errors="ignore")
    common_env = {"CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONPATH": f"{repo}:{root}:{os.environ.get('PYTHONPATH', '')}"}
    c0 = run_cmd([sys.executable, str(call_script), "--source", str(source_dir), "--target", str(target_dir), "--GPUs", str(args.gpu)], eval_dir, out_dir / "0_call_acc.log.txt", args.eval_timeout, common_env)
    c1 = run_cmd([sys.executable, str(exe_script), "--folder", str(target_dir), "--GPUs", str(args.gpu)], eval_dir, out_dir / "1_exe_acc.log.txt", args.eval_timeout, common_env)
    row["official_attempted"] = 1; row["evaluator_rc"] = max(c0.rc, c1.rc); row["logs"] = [c0.log, c1.log]
    correctness_text = c0.stdout + c0.stderr + "\n" + c1.stdout + c1.stderr
    rate_match = re.search(r"Correct execution rate:\s*([0-9.]+)%", correctness_text, re.I)
    infrastructure_failure = any(marker in correctness_text for marker in ["SyntaxError: f-string", "/home/lijianling/"])
    row["official_eval"] = int(rate_match is not None and not infrastructure_failure)
    if rate_match:
        row["correct"] = int(float(rate_match.group(1)) >= 99.999)
    if not row["official_eval"]:
        row["infrastructure_error"] = correctness_text[-3000:]
        row["wall_s"] = time.time() - start
        return row
    if row["correct"] and perf_dir.exists() and efficiency_script.exists():
        for directory in [perf_dir / "tmp", perf_dir / "run_bench/tmp"]:
            if directory.exists():
                for path in directory.iterdir():
                    shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink(missing_ok=True)
            directory.mkdir(parents=True, exist_ok=True)
        write_file, multi = perf_dir / "run_bench/write_file.py", perf_dir / "run_bench/multiprocess_gpu_run.py"
        perf_results = out_dir / "perf_results"; clean_directory(perf_results)
        if write_file.exists():
            source = write_file.read_text(errors="ignore")
            flag = "--result_folder_path" if "--result_folder_path" in source else "--results_path"
            c2 = run_cmd([sys.executable, str(write_file), "--input_folder_path", str(target_dir), flag, str(perf_results)], perf_dir, out_dir / "perf_write_file.log.txt", args.eval_timeout, common_env)
            row["logs"].append(c2.log)
        if multi.exists():
            c3 = run_cmd([sys.executable, str(multi)], perf_dir, out_dir / "perf_multiprocess.log.txt", args.eval_timeout, common_env)
            row["logs"].append(c3.log)
        c4 = run_cmd([sys.executable, str(efficiency_script), "--gen_folder", str(perf_results)], eval_dir, out_dir / "2_efficiency.log.txt", args.eval_timeout, common_env)
        row["logs"].append(c4.log)
        speed = parse_speed(c4.stdout + c4.stderr)
        if speed is not None:
            row["performance_available"] = 1; row["score"] = speed
        else:
            row["candidate_error"] = "correctness passed, but official performance score was unavailable"
    row["artifacts"] = [str(source_file), str(target_file)]
    row["wall_s"] = time.time() - start
    return row


def patch_pareval_repo(repo: Path) -> list[str]:
    """ParEval's run-all.py uses contextlib.chdir (Python 3.11+). This box runs
    3.10, so inject an idempotent backport shim right after the imports."""
    patched: list[str] = []
    shim_marker = "# unified_bench: contextlib.chdir 3.10 backport"
    shim = (
        f"\n{shim_marker}\n"
        "import contextlib as _ub_contextlib, os as _ub_os\n"
        "if not hasattr(_ub_contextlib, 'chdir'):\n"
        "    class _ub_chdir(_ub_contextlib.AbstractContextManager):\n"
        "        def __init__(self, path): self._path = path\n"
        "        def __enter__(self): self._old = _ub_os.getcwd(); _ub_os.chdir(self._path)\n"
        "        def __exit__(self, *exc): _ub_os.chdir(self._old)\n"
        "    _ub_contextlib.chdir = _ub_chdir\n"
    )
    for name in ["drivers/run-all.py"]:
        path = repo / name
        if not path.exists():
            continue
        text = path.read_text(errors="ignore")
        if "contextlib.chdir" not in text or shim_marker in text:
            continue
        backup = path.with_suffix(path.suffix + ".bak_official_matrix")
        if not backup.exists():
            backup.write_text(text, errors="ignore")
        lines = text.splitlines(keepends=True)
        insert_at = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                insert_at = i + 1
        lines.insert(insert_at, shim)
        path.write_text("".join(lines), errors="ignore")
        patched.append(f"{path}:contextlib_chdir")
    return patched


def patch_sol_repo(repo: Path) -> list[str]:
    """SOL-ExecBench's timing.py hard-imports the `cupti` python bindings, which are
    not pip-installable and absent here. It also ships a cupti-free `cuda_events`
    timing path. Make the cupti import optional and default timing to cuda_events."""
    patched: list[str] = []
    timing = repo / "src/sol_execbench/core/bench/timing.py"
    if not timing.exists():
        return patched
    text = timing.read_text(errors="ignore")
    marker = "# unified_bench: cupti optional"
    if marker in text:
        return patched
    original = text
    text = text.replace(
        "from cupti import cupti\n",
        "try:  " + marker + "\n"
        "    from cupti import cupti\n"
        "except Exception:\n"
        "    cupti = None\n",
    )
    text = text.replace(
        'methodology: Literal["cuda_events", "cupti"] = "cupti",',
        'methodology: Literal["cuda_events", "cupti"] = "cuda_events",',
    )
    if text == original:
        return patched
    backup = timing.with_suffix(timing.suffix + ".bak_official_matrix")
    if not backup.exists():
        backup.write_text(original, errors="ignore")
    timing.write_text(text, errors="ignore")
    patched.append(f"{timing}:cupti_optional+cuda_events")
    return patched


def infer_candidate_language(candidate: Path) -> str:
    text = candidate.read_text(errors="ignore").lower()
    return "triton" if "@triton.jit" in text or "import triton" in text else "cuda"


def evaluate_multikernel(root: Path, task: Task, candidate: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    row = result_base("multikernelbench", task, candidate)
    repo = repo_for(root, "multikernelbench")
    if not repo:
        row["infrastructure_error"] = "MultiKernelBench repo missing"; return row
    evaluator = repo / "eval_single_runner.py"
    if not evaluator.exists():
        row["infrastructure_error"] = "eval_single_runner.py missing"; return row
    response = out_dir / "generated_response.txt"; response.parent.mkdir(parents=True, exist_ok=True)
    response.write_text(candidate.read_text(errors="ignore"), errors="ignore")
    result_path = out_dir / "official_result.json"
    result = run_cmd([sys.executable, str(evaluator), "--input", str(response), "--op", str(task.metadata["op"]), "--language", infer_candidate_language(candidate), "--result", str(result_path)], repo, out_dir / "official_eval.log.txt", args.eval_timeout, {"CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONPATH": f"{repo}:{root}:{os.environ.get('PYTHONPATH', '')}"})
    row["official_attempted"] = 1; row["evaluator_rc"] = result.rc; row["logs"] = [result.log]
    data = read_json(result_path, None)
    if isinstance(data, dict):
        row["official_eval"] = 1
        row["correct"] = int(bool(parse_bool(data, ["correct", "correctness", "passed", "success", "verified"])))
        score = parse_number(data, ["speedup", "performance", "score", "best_score"])
        if score is not None:
            row["performance_available"] = 1; row["score"] = score
        row["artifacts"] = [str(result_path)]
        if not row["correct"]:
            row["candidate_error"] = str(data.get("error") or data.get("message") or "")
    else:
        row["infrastructure_error"] = (result.stdout + result.stderr)[-3000:]
    row["wall_s"] = result.wall_s
    return row


def candidate_backend_wrapper(op: str, candidate_code: str) -> str:
    candidate_literal = repr(candidate_code)
    return f"""
import types
_CANDIDATE_CODE = {candidate_literal}
_candidate = types.ModuleType('_official_agent_candidate')
_candidate_error = None
try:
    exec(compile(_CANDIDATE_CODE, '<agent_candidate>', 'exec'), _candidate.__dict__)
except Exception as exc:
    _candidate_error = exc

def _call_candidate(*args, **kwargs):
    if _candidate_error is not None:
        raise RuntimeError('candidate import failed') from _candidate_error
    for name in ['{op}_kernel_impl', '{op}', 'kernel', 'run', 'forward', 'solution', 'call']:
        fn = getattr(_candidate, name, None)
        if callable(fn):
            return fn(*args, **kwargs)
    cls = getattr(_candidate, 'ModelNew', None)
    if cls is not None:
        module = cls()
        return module(*args, **kwargs)
    raise RuntimeError('candidate exposes no supported callable')

def {op}_kernel_impl(*args, **kwargs):
    return _call_candidate(*args, **kwargs)
""".strip() + "\n"


BACKEND_OP_NAME_KEYS = ["op_name", "operator", "op", "name", "operator_name", "test_name"]
BACKEND_CORRECT_KEYS = ["is_correct", "correct", "correctness", "passed", "success"]


def find_backend_result(data: Any, target_op: str) -> Optional[dict[str, Any]]:
    if isinstance(data, dict):
        name = first_string(data, BACKEND_OP_NAME_KEYS)
        if name and normalize_backend_op(name) == target_op:
            return data
        for value in data.values():
            found = find_backend_result(value, target_op)
            if found:
                return found
    elif isinstance(data, list):
        for value in data:
            found = find_backend_result(value, target_op)
            if found:
                return found
    return None


def collect_backend_rows(data: Any, target_op: str) -> list[dict[str, Any]]:
    """All leaf dict rows in the official results whose op-name maps to target_op.

    BackendBench writes one row per (op, test-case); an op is correct only if
    every one of its correctness rows passed, so we must aggregate, not take
    the first match."""
    rows: list[dict[str, Any]] = []
    if isinstance(data, dict):
        name = first_string(data, BACKEND_OP_NAME_KEYS)
        if name and normalize_backend_op(name) == target_op:
            rows.append(data)
        else:
            for value in data.values():
                rows.extend(collect_backend_rows(value, target_op))
    elif isinstance(data, list):
        for value in data:
            rows.extend(collect_backend_rows(value, target_op))
    return rows


def evaluate_backendbench(root: Path, task: Task, candidate: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    row = result_base("backendbench", task, candidate)
    repo = repo_for(root, "backendbench")
    if not repo:
        row["infrastructure_error"] = "BackendBench repo missing"; return row
    main = repo / "BackendBench/scripts/main.py"
    if not main.exists():
        row["infrastructure_error"] = "BackendBench official CLI missing"; return row
    target_op = str(task.metadata["op"])
    smoke_ops = [normalize_backend_op(x) for x in task.metadata.get("smoke_ops", KNOWN_BACKENDBENCH_OPS)]
    smoke_ops = [x for x in smoke_ops if x]
    if target_op not in smoke_ops:
        smoke_ops.append(target_op)
    ops_dir = out_dir / "ops"; clean_directory(ops_dir)
    candidate_code = candidate.read_text(errors="ignore")
    for op in smoke_ops:
        directory = ops_dir / op; directory.mkdir(parents=True, exist_ok=True)
        code = candidate_backend_wrapper(op, candidate_code) if op == target_op else eager_backend_impl(op)
        (directory / f"{op}_implementation_1.py").write_text(code, errors="ignore")
        (directory / "README.md").write_text(f"# {op}\n", errors="ignore")
    start = time.time(); row["official_attempted"] = 1; all_text = ""; suites_ran = False
    # opinfo covers compute-bound ops (mm/bmm/gelu/relu/...). Overhead-dominated
    # elementwise ops (add/mul/sub/div) are excluded by opinfo and are only testable
    # via the torchbench suite with --check-overhead-dominated-ops (the flag errors on
    # any other suite). Try opinfo first, then torchbench for the overhead ops.
    for index, (suite, extra) in enumerate([("opinfo", []), ("torchbench", ["--check-overhead-dominated-ops"]), ("smoke", [])]):
        log_dir = out_dir / f"results_{index}_{suite}"; clean_directory(log_dir)
        result = run_cmd([sys.executable, str(main), "--suite", suite, "--backend", "directory", "--ops-directory", str(ops_dir), "--log-dir", str(log_dir)] + extra, repo, out_dir / f"official_{index}_{suite}.log.txt", args.eval_timeout, {"CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONPATH": f"{repo}:{root}:{os.environ.get('PYTHONPATH', '')}"})
        row["logs"].append(result.log); all_text += result.stdout + result.stderr + "\n"
        full_path = log_dir / "full_results.json"; full = read_json(full_path, None)
        if not isinstance(full, list) or not full:
            continue
        suites_ran = True  # the suite executed and produced a results file
        matched = collect_backend_rows(full, target_op)
        if not matched:
            continue
        row["official_eval"] = 1; row["evaluator_rc"] = result.rc
        # Aggregate correctness across every correctness row for this op.
        row_flags = [parse_bool(r, BACKEND_CORRECT_KEYS) for r in matched]
        row_flags = [f for f in row_flags if f is not None]
        if row_flags:
            correct = all(row_flags)
        else:
            failed = read_json(log_dir / "failed_tests.json", [])
            correct = not collect_backend_rows(failed, target_op)
        row["correct"] = int(bool(correct))
        scores = [parse_number(r, ["speedup", "performance", "score"]) for r in matched]
        scores = [s for s in scores if s is not None]
        if scores:
            row["performance_available"] = 1; row["score"] = max(scores)
        row["artifacts"] = [str(full_path), str(log_dir / "operator_summary.csv"), str(log_dir / "OVERALL_SUMMARY.md")]
        if not row["correct"]:
            bad = [r for r, f in zip(matched, [parse_bool(r, BACKEND_CORRECT_KEYS) for r in matched]) if f is False]
            row["candidate_error"] = json.dumps(bad or matched, ensure_ascii=False)[-3000:]
        break
    if not row["official_eval"]:
        if suites_ran:
            # The candidate loaded and the suites ran, but this op has no test case in
            # any locally-available BackendBench suite (opinfo excludes it and no
            # torchbench trace data covers it). Report the coverage gap explicitly so
            # it is not confused with an evaluator/infrastructure failure.
            row["infrastructure_error"] = (
                f"op '{target_op}' not covered by locally-available BackendBench suites "
                f"(opinfo/torchbench); no official test case exists for it here"
            )
        else:
            row["infrastructure_error"] = all_text[-3000:]
    row["wall_s"] = time.time() - start
    return row


def evaluate_pareval(root: Path, task: Task, candidate: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    row = result_base("pareval", task, candidate)
    repo = repo_for(root, "pareval")
    if not repo:
        row["infrastructure_error"] = "ParEval repo missing"; return row
    run_all = repo / "drivers/run-all.py"
    if not run_all.exists():
        row["infrastructure_error"] = "ParEval run-all.py missing"; return row
    patch_pareval_repo(repo)
    prompt = dict(task.metadata.get("prompt_obj", {}))
    prompt["name"] = prompt.get("name") or task.task_id
    prompt["parallelism_model"] = "cuda"
    prompt["language"] = "cpp"
    prompt["outputs"] = [candidate.read_text(errors="ignore")]
    input_path, result_path = out_dir / "pareval_input.json", out_dir / "pareval_result.json"
    write_json(input_path, [prompt])
    config_paths: dict[str, Path] = {}
    for name in ["launch-configs.json", "build-configs.json", "problem-sizes.json"]:
        paths = [repo / "drivers" / name, repo / name] + list(repo.rglob(name))
        path = next((x for x in paths if x.exists() and x.is_file()), None)
        if path is None:
            row["infrastructure_error"] = f"ParEval official config missing: {name}"; return row
        config_paths[name] = path
    cmd = [sys.executable, str(run_all), str(input_path), "--launch-configs", str(config_paths["launch-configs.json"]), "--build-configs", str(config_paths["build-configs.json"]), "--problem-sizes", str(config_paths["problem-sizes.json"]), "--include-models", "cuda", "--problem", str(prompt["name"]), "-o", str(result_path), "--yes-to-all", "--hide-progress", "--early-exit-runs", "--build-timeout", str(args.pareval_build_timeout), "--run-timeout", str(args.pareval_run_timeout)]
    result = run_cmd(cmd, repo, out_dir / "official_eval.log.txt", args.eval_timeout, {"CUDA_VISIBLE_DEVICES": str(args.gpu), "PAREVAL_ROOT": str(repo), "PYTHONPATH": f"{repo / 'drivers'}:{repo}:{root}:{os.environ.get('PYTHONPATH', '')}"})
    row["official_attempted"] = 1; row["evaluator_rc"] = result.rc; row["logs"] = [result.log]
    data = read_json(result_path, None)
    if isinstance(data, list) and data and isinstance(data[0], dict):
        outputs = data[0].get("outputs")
        if isinstance(outputs, list) and outputs and all(isinstance(x, dict) for x in outputs):
            row["official_eval"] = 1
            # ParEval's pass signal is `are_all_valid` (built, ran, and every run's
            # output matched the reference). The generic key search misses it because
            # the field is named are_all_valid, not "valid".
            valid_flags = [o.get("are_all_valid") for o in outputs
                           if isinstance(o, dict) and "are_all_valid" in o]
            if valid_flags:
                row["correct"] = int(all(bool(v) for v in valid_flags))
            else:
                row["correct"] = int(bool(parse_bool(outputs, ["are_all_valid", "correct", "passed", "success", "valid"])))
            score = parse_number(outputs, ["speedup", "performance", "score", "runtime", "time"])
            if score is not None:
                row["performance_available"] = 1; row["score"] = score
            row["artifacts"] = [str(result_path)]
            if not row["correct"]:
                row["candidate_error"] = json.dumps(outputs, ensure_ascii=False)[-3000:]
    if not row["official_eval"]:
        row["infrastructure_error"] = (result.stdout + result.stderr)[-3000:]
    row["wall_s"] = result.wall_s
    return row


def infer_sol_language(candidate: Path) -> str:
    text = candidate.read_text(errors="ignore"); low = text.lower()
    if candidate.suffix in {".cu", ".cpp", ".cc"} or "__global__" in text or "#include <torch/extension.h>" in text:
        return "cuda_cpp"
    if "@triton.jit" in low or "import triton" in low:
        return "triton"
    return "pytorch"


def valid_identifier(name: str) -> str:
    value = re.sub(r"\W+", "_", name)
    if not value or value[0].isdigit():
        value = "_" + value
    return value


def sol_wrapper_source(definition: dict[str, Any], candidate_code: str) -> str:
    inputs = list((definition.get("inputs") or {}).keys())
    outputs = list((definition.get("outputs") or {}).keys())
    input_args = [valid_identifier(x) for x in inputs]
    output_args = [valid_identifier(x) for x in outputs]
    signature = ", ".join(input_args + output_args)
    input_list = ", ".join(input_args)
    candidate_literal = repr(candidate_code)
    if len(output_args) == 1:
        copy_logic = f"\n    if result is not None:\n        {output_args[0]}.copy_(result)\n"
    elif len(output_args) > 1:
        copy_lines = "\n".join(f"        {out}.copy_(result[{index}])" for index, out in enumerate(output_args))
        copy_logic = f"\n    if result is not None:\n        if not isinstance(result, (tuple, list)):\n            raise RuntimeError('candidate must return {len(output_args)} outputs')\n{copy_lines}\n"
    else:
        copy_logic = "\n"
    return f"""
import types
_CANDIDATE_CODE = {candidate_literal}
_candidate = types.ModuleType('_sol_agent_candidate')
_candidate_error = None
try:
    exec(compile(_CANDIDATE_CODE, '<agent_candidate>', 'exec'), _candidate.__dict__)
except Exception as exc:
    _candidate_error = exc

def _call_candidate(*args):
    if _candidate_error is not None:
        raise RuntimeError('candidate import failed') from _candidate_error
    for name in ['run', 'forward', 'kernel', 'solution', 'call']:
        fn = getattr(_candidate, name, None)
        if callable(fn):
            return fn(*args)
    cls = getattr(_candidate, 'ModelNew', None)
    if cls is not None:
        module = cls()
        return module(*args)
    raise RuntimeError('candidate exposes no callable run/forward/kernel/ModelNew')

def run({signature}):
    result = _call_candidate({input_list})
{copy_logic}
""".strip() + "\n"


def make_sol_solution(task: Task, candidate: Path, out_dir: Path, agent_name: str) -> Path:
    definition = task.metadata.get("definition", {})
    definition_name = definition.get("name") if isinstance(definition, dict) else None
    definition_name = definition_name or task.task_id
    language = infer_sol_language(candidate)
    candidate_code = candidate.read_text(errors="ignore")
    if language in {"pytorch", "triton"}:
        wrapper = sol_wrapper_source(definition if isinstance(definition, dict) else {}, candidate_code)
        sources = [{"path": "kernel.py", "content": wrapper}]
        spec = {"languages": [language], "target_hardware": ["LOCAL"], "entry_point": "kernel.py::run", "dependencies": ["torch"] + (["triton >= 2.3"] if language == "triton" else []), "destination_passing_style": True}
    else:
        sources = [{"path": "kernel.cu", "content": candidate_code}]
        spec = {"languages": ["cuda_cpp"], "target_hardware": ["LOCAL"], "entry_point": "kernel.cu::run", "dependencies": [], "destination_passing_style": True, "binding": "torch", "compile_options": {"cuda_cflags": ["-O3", "--use_fast_math", "-std=c++17"]}}
    solution = {"name": f"{slug(agent_name)}_{slug(task.task_id)}", "definition": str(definition_name), "author": agent_name, "description": "Generated by unified_bench official matrix.", "spec": spec, "sources": sources}
    path = out_dir / "solution.json"; write_json(path, solution); return path


def parse_json_lines(path: Path) -> list[Any]:
    items: list[Any] = []
    if not path.exists():
        return items
    for line in path.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            items.append(json.loads(line))
        except Exception:
            pass
    return items


def evaluate_sol(root: Path, task: Task, candidate: Path, out_dir: Path, args: argparse.Namespace, agent_name: str) -> dict[str, Any]:
    row = result_base("sol_execbench", task, candidate)
    repo = repo_for(root, "sol_execbench")
    if not repo:
        row["infrastructure_error"] = "SOL-ExecBench repo missing"; return row
    patch_sol_repo(repo)
    problem_dir = Path(str(task.metadata.get("problem_dir", "")))
    if not (problem_dir.is_dir() and (problem_dir / "definition.json").exists() and (problem_dir / "workload.jsonl").exists()):
        row["infrastructure_error"] = "invalid SOL problem directory"; return row
    solution = make_sol_solution(task, candidate, out_dir, agent_name)
    config = out_dir / "bench_config.json"
    write_json(config, {"warmup_runs": args.sol_warmup, "iterations": args.sol_iterations, "lock_clocks": False, "benchmark_reference": False, "seed": args.seed})
    trace_path = out_dir / "trace.jsonl"
    env = {"CUDA_VISIBLE_DEVICES": str(args.gpu), "PYTHONPATH": f"{repo / 'src'}:{repo}:{root}:{os.environ.get('PYTHONPATH', '')}"}
    cmd = [sys.executable, "-m", "sol_execbench.cli.main", str(problem_dir), "--solution", str(solution), "--config", str(config), "--compile-timeout", str(args.sol_compile_timeout), "--timeout", str(args.sol_run_timeout), "-o", str(trace_path), "--json"]
    result = run_cmd(cmd, repo, out_dir / "official_eval.log.txt", args.eval_timeout, env)
    row["official_attempted"] = 1; row["evaluator_rc"] = result.rc; row["logs"] = [result.log]
    traces = parse_json_lines(trace_path)
    if not traces:
        try:
            obj = json.loads(result.stdout.strip()); traces = obj if isinstance(obj, list) else [obj]
        except Exception:
            pass
    if traces:
        row["official_eval"] = 1
        # Each SOL trace line reports a per-workload verdict under
        # evaluation.status ("PASSED"/"FAILED"). The task is correct only if every
        # workload passed. The generic key search misses this because the signal is
        # a nested `status` string, not a top-level "correct"/"passed" bool.
        statuses = [str((tr.get("evaluation") or {}).get("status", "")).upper()
                    for tr in traces if isinstance(tr, dict)]
        statuses = [s for s in statuses if s]
        if statuses:
            row["correct"] = int(all(s == "PASSED" for s in statuses))
        else:
            row["correct"] = int(bool(parse_bool(traces, ["correct", "correctness", "passed", "success", "status"])))
        score = parse_number(traces, ["sol_score", "score", "speedup", "performance", "latency"])
        if score is not None:
            row["performance_available"] = 1; row["score"] = score
        row["artifacts"] = [str(solution), str(trace_path), str(config)]
        if not row["correct"]:
            row["candidate_error"] = json.dumps(traces, ensure_ascii=False)[-3000:]
    else:
        row["infrastructure_error"] = (result.stdout + result.stderr)[-3000:]
    row["wall_s"] = result.wall_s
    return row


def evaluate_candidate(root: Path, agent: Agent, task: Task, candidate: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if task.benchmark in {"tritonbench_t", "tritonbench_g"}:
        row = evaluate_triton(root, task.benchmark, task, candidate, out_dir, args)
    elif task.benchmark == "multikernelbench":
        row = evaluate_multikernel(root, task, candidate, out_dir, args)
    elif task.benchmark == "backendbench":
        row = evaluate_backendbench(root, task, candidate, out_dir, args)
    elif task.benchmark == "pareval":
        row = evaluate_pareval(root, task, candidate, out_dir, args)
    elif task.benchmark == "sol_execbench":
        row = evaluate_sol(root, task, candidate, out_dir, args, agent.name)
    else:
        row = result_base(task.benchmark, task, candidate); row["infrastructure_error"] = f"no native adapter for {task.benchmark}"
    write_json(out_dir / "official_report.json", row)
    return row


def aggregate_candidates(rows: list[dict[str, Any]], task: Task, generation: dict[str, Any]) -> dict[str, Any]:
    if not rows:
        return {"benchmark": task.benchmark, "task_id": task.task_id, "official_task": 1, "generation_ok": int(generation.get("candidate_count", 0) > 0), "candidate_count": generation.get("candidate_count", 0), "official_attempted": 0, "official_eval": 0, "correct": 0, "performance_available": 0, "best_score": 0.0, "infrastructure_error": "no candidate was evaluated", "candidate_error": "", "generation_log": generation.get("log", ""), "official_reports": []}
    official_rows = [r for r in rows if r.get("official_eval")]
    correct_rows = [r for r in official_rows if r.get("correct")]
    scored_rows = [r for r in official_rows if r.get("performance_available")]
    infrastructure_errors = [r.get("infrastructure_error", "") for r in rows if r.get("infrastructure_error")]
    candidate_errors = [r.get("candidate_error", "") for r in rows if r.get("candidate_error")]
    return {"benchmark": task.benchmark, "task_id": task.task_id, "official_task": 1, "generation_ok": int(generation.get("candidate_count", 0) > 0), "candidate_count": generation.get("candidate_count", 0), "official_attempted": int(any(r.get("official_attempted") for r in rows)), "official_eval": int(bool(official_rows)), "correct": int(bool(correct_rows)), "performance_available": int(bool(scored_rows)), "best_score": max([float(r.get("score", 0) or 0) for r in scored_rows] + [0.0]), "infrastructure_error": Counter(infrastructure_errors).most_common(1)[0][0] if infrastructure_errors else "", "candidate_error": Counter(candidate_errors).most_common(1)[0][0] if candidate_errors else "", "generation_log": generation.get("log", ""), "official_reports": []}


def run_native_cell(root: Path, agent: Agent, tasks: list[Task], args: argparse.Namespace, cell_dir: Path) -> list[dict[str, Any]]:
    task_rows: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks):
        task_dir = cell_dir / f"{task_index:03d}_{slug(task.task_id)}"; task_dir.mkdir(parents=True, exist_ok=True)
        candidates, generation = generate_candidates(root, agent, task, task_dir, args)
        candidate_rows = [evaluate_candidate(root, agent, task, candidate, task_dir / "official_eval" / f"candidate_{index:03d}", args) for index, candidate in enumerate(candidates)]
        aggregate = aggregate_candidates(candidate_rows, task, generation)
        aggregate.update({"agent": agent.name, "cell_dir": str(cell_dir), "task_dir": str(task_dir)})
        write_json(task_dir / "task_result.json", aggregate)
        task_rows.append(aggregate)
    return task_rows


def run_one_cell(root: Path, run_root: Path, agent: Agent, benchmark: str, tasks: list[Task], args: argparse.Namespace) -> dict[str, Any]:
    cell_dir = run_root / "cells" / benchmark / agent.name
    if args.force:
        shutil.rmtree(cell_dir, ignore_errors=True)
    cell_dir.mkdir(parents=True, exist_ok=True)
    existing = read_json(cell_dir / "cell_result.json", None)
    if isinstance(existing, dict) and existing.get("valid_official_cell") == 1 and not args.force:
        return existing
    started = time.time()
    if benchmark in KB_BENCHMARKS:
        raw_rows = evaluate_kb_cell(root, run_root, agent, benchmark, tasks, args, cell_dir)
        task_rows = []
        for task, row in zip(tasks, raw_rows):
            task_rows.append({"benchmark": benchmark, "task_id": task.task_id, "agent": agent.name, "official_task": 1, "generation_ok": int(bool(row.get("candidate"))), "candidate_count": int(bool(row.get("candidate"))), "official_attempted": row.get("official_attempted", 0), "official_eval": row.get("official_eval", 0), "correct": row.get("correct", 0), "performance_available": row.get("performance_available", 0), "best_score": row.get("score", 0.0), "infrastructure_error": row.get("infrastructure_error", ""), "candidate_error": row.get("candidate_error", ""), "generation_log": row.get("logs", [""])[0] if row.get("logs") else "", "cell_dir": str(cell_dir), "task_dir": ""})
    else:
        task_rows = run_native_cell(root, agent, tasks, args, cell_dir)
    n_tasks = len(task_rows)
    candidate_count = sum(int(row.get("generation_ok", 0)) for row in task_rows)
    official_count = sum(int(row.get("official_eval", 0)) for row in task_rows)
    correct_count = sum(int(row.get("correct", 0)) for row in task_rows)
    infrastructure = [row.get("infrastructure_error", "") for row in task_rows if row.get("infrastructure_error")]
    cell = {"benchmark": benchmark, "agent": agent.name, "n_tasks": n_tasks, "candidate_task_count": candidate_count, "official_eval_task_count": official_count, "correct_task_count": correct_count, "valid_official_cell": int(n_tasks > 0 and candidate_count == n_tasks and official_count == n_tasks), "all_tasks_official": int(n_tasks > 0 and official_count == n_tasks), "wall_s": time.time() - started, "top_infrastructure_error": Counter(infrastructure).most_common(1)[0][0] if infrastructure else "", "task_results": task_rows, "finished_at": iso_now()}
    write_json(cell_dir / "cell_result.json", cell)
    return cell


def flatten_task_rows(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in cells:
        for row in cell.get("task_results", []):
            flat = dict(row); flat.setdefault("benchmark", cell["benchmark"]); flat.setdefault("agent", cell["agent"]); rows.append(flat)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore"); writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def collect_run(run_root: Path, agents: list[Agent], benchmarks: list[str]) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        for agent in agents:
            path = run_root / "cells" / benchmark / agent.name / "cell_result.json"
            data = read_json(path, None)
            cells.append(data if isinstance(data, dict) else {"benchmark": benchmark, "agent": agent.name, "n_tasks": 0, "candidate_task_count": 0, "official_eval_task_count": 0, "correct_task_count": 0, "valid_official_cell": 0, "wall_s": 0, "top_infrastructure_error": "cell_result.json missing", "task_results": []})
    task_rows = flatten_task_rows(cells)
    write_csv(run_root / "official_matrix_cells.csv", cells, ["benchmark", "agent", "n_tasks", "candidate_task_count", "official_eval_task_count", "correct_task_count", "valid_official_cell", "all_tasks_official", "wall_s", "top_infrastructure_error", "finished_at"])
    write_csv(run_root / "official_matrix_per_task.csv", task_rows, ["benchmark", "agent", "task_id", "official_task", "generation_ok", "candidate_count", "official_attempted", "official_eval", "correct", "performance_available", "best_score", "infrastructure_error", "candidate_error", "generation_log", "cell_dir", "task_dir"])
    by_benchmark: dict[str, Any] = {}
    for benchmark in benchmarks:
        group = [cell for cell in cells if cell["benchmark"] == benchmark]
        by_benchmark[benchmark] = {"agents": len(group), "valid_official_cells": sum(int(cell.get("valid_official_cell", 0)) for cell in group), "candidate_cells": sum(int(cell.get("candidate_task_count", 0) > 0) for cell in group), "official_eval_tasks": sum(int(cell.get("official_eval_task_count", 0)) for cell in group), "correct_tasks": sum(int(cell.get("correct_task_count", 0)) for cell in group), "all_agents_official": int(bool(group) and all(cell.get("valid_official_cell") == 1 for cell in group)), "top_errors": Counter(cell.get("top_infrastructure_error", "") for cell in group if cell.get("top_infrastructure_error")).most_common(5)}
    manifest = {"run_root": str(run_root), "generated_at": iso_now(), "benchmarks": benchmarks, "agents": [agent.name for agent in agents], "expected_cells": len(benchmarks) * len(agents), "observed_cells": len(cells), "valid_official_cells": sum(int(cell.get("valid_official_cell", 0)) for cell in cells), "all_official_benchmarking_normal": int(len(cells) == len(benchmarks) * len(agents) and all(cell.get("valid_official_cell") == 1 for cell in cells)), "by_benchmark": by_benchmark, "strict_definition": "A valid cell has official tasks, a generated candidate for every selected task, and a parsed structured upstream-evaluator result for every task. Correctness may be zero."}
    write_json(run_root / "official_matrix_manifest.json", manifest)
    return manifest


def preflight(root: Path, run_root: Path, agents: list[Agent], benchmarks: list[str], args: argparse.Namespace) -> dict[str, Any]:
    checks: dict[str, Any] = {"root": str(root), "generated_at": iso_now(), "agents": {}, "benchmarks": {}, "commands": {}}
    for agent in agents:
        checks["agents"][agent.name] = {"driver": str(agent.driver), "driver_exists": int(agent.driver.exists()), "model": agent.model, "langs": agent.langs}
    for benchmark in benchmarks:
        repo = repo_for(root, benchmark)
        checks["benchmarks"][benchmark] = {"repo": str(repo or ""), "repo_exists": int(repo is not None)}
    nvidia = run_cmd(["nvidia-smi"], root, run_root / "preflight/nvidia_smi.log.txt", 60)
    health = run_cmd(["bash", "-lc", f"curl -sf http://127.0.0.1:{args.port}/health"], root, run_root / "preflight/llm_health.log.txt", 30)
    checks["commands"]["nvidia_smi"] = {"rc": nvidia.rc, "log": nvidia.log}; checks["commands"]["llm_health"] = {"rc": health.rc, "log": health.log}
    checks["ok"] = int(nvidia.rc == 0 and health.rc == 0 and all(item["driver_exists"] for item in checks["agents"].values()) and all(item["repo_exists"] for item in checks["benchmarks"].values()))
    write_json(run_root / "preflight/preflight_report.json", checks)
    return checks


def run_matrix(root: Path, run_root: Path, agents: list[Agent], benchmarks: list[str], args: argparse.Namespace) -> dict[str, Any]:
    tasks_by_benchmark = {benchmark: load_prepared_tasks(run_root, benchmark) for benchmark in benchmarks}
    missing = [benchmark for benchmark, tasks in tasks_by_benchmark.items() if not tasks]
    if missing:
        raise RuntimeError(f"No prepared official tasks for: {missing}. Run prepare first.")
    total = len(benchmarks) * len(agents); index = 0
    for benchmark in benchmarks:
        tasks = tasks_by_benchmark[benchmark][:args.limit]
        if not tasks:
            raise RuntimeError(f"Official task limit produced zero tasks for {benchmark}")
        for agent in agents:
            index += 1
            print(f"[{index}/{total}] official benchmark={benchmark} agent={agent.name} tasks={len(tasks)}", flush=True)
            original_force = args.force
            cell: dict[str, Any] = {}
            for attempt in range(1, args.cell_attempts + 1):
                cell = run_one_cell(root, run_root, agent, benchmark, tasks, args)
                if cell.get("valid_official_cell") == 1:
                    break
                if attempt < args.cell_attempts:
                    args.force = True
                    print(f"  retry {attempt + 1}/{args.cell_attempts}: {cell.get('top_infrastructure_error', '')[:300]}", flush=True)
            args.force = original_force
            collect_run(run_root, agents, benchmarks)
            print(json.dumps({"benchmark": benchmark, "agent": agent.name, "valid_official_cell": cell.get("valid_official_cell", 0), "official_eval_task_count": cell.get("official_eval_task_count", 0), "n_tasks": cell.get("n_tasks", 0), "correct_task_count": cell.get("correct_task_count", 0)}, ensure_ascii=False), flush=True)
    return collect_run(run_root, agents, benchmarks)


def pack_run(root: Path, run_root: Path) -> Path:
    bundle = root / f"rtx6000_all_official_matrix_v1_feedback_{now_id()}"
    archive = Path(str(bundle) + ".tar.gz")
    shutil.rmtree(bundle, ignore_errors=True)
    (bundle / "results").mkdir(parents=True, exist_ok=True); (bundle / "env").mkdir(parents=True, exist_ok=True); (bundle / "registry").mkdir(parents=True, exist_ok=True); (bundle / "third_party_status").mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_root, bundle / "results" / run_root.name, dirs_exist_ok=True)
    for path in [root / "rtx6000_all_official_matrix_v1.sh", root / "official_all_matrix_v1.py", root / "run_ext_rtx6000.sh"]:
        if path.exists():
            shutil.copy2(path, bundle / "env" / path.name)
    registry = root / "unified_bench_ext/registry"
    if registry.exists():
        shutil.copytree(registry, bundle / "registry", dirs_exist_ok=True)
    for repo in (root / "third_party").glob("*") if (root / "third_party").exists() else []:
        if not (repo / ".git").exists():
            continue
        run_cmd(["bash", "-lc", f"git -C {shlex.quote(str(repo))} remote -v; git -C {shlex.quote(str(repo))} rev-parse HEAD; git -C {shlex.quote(str(repo))} status --short"], root, bundle / "third_party_status" / f"{repo.name}.txt", 60)
    run_cmd(["bash", "-lc", "python3 --version; nvidia-smi; curl -s http://127.0.0.1:8000/health || true; df -hT .; free -h"], root, bundle / "env/system.txt", 120)
    subprocess.run(["tar", "-czf", str(archive), "-C", str(bundle.parent), bundle.name], check=True)
    shutil.rmtree(bundle, ignore_errors=True)
    return archive


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "prepare", "run", "verify", "all", "pack"])
    parser.add_argument("--root", default=os.environ.get("ROOT", "/home/bi_geum/unified_bench"))
    parser.add_argument("--run-id", default=os.environ.get("OFFICIAL_RUN_ID", "official_matrix_v1"))
    parser.add_argument("--benchmarks", default=os.environ.get("BENCHMARKS", "all"))
    parser.add_argument("--agents", default=os.environ.get("AGENTS", "all"))
    parser.add_argument("--limit", type=int, default=int(os.environ.get("OFFICIAL_LIMIT", "1")))
    parser.add_argument("--rounds", type=int, default=int(os.environ.get("ROUNDS", "1")))
    parser.add_argument("--repeat", type=int, default=int(os.environ.get("REPEAT", "1")))
    parser.add_argument("--temp", type=float, default=float(os.environ.get("TEMP", "0.2")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    parser.add_argument("--gpu", default=os.environ.get("GPU", "0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--generation-timeout", type=int, default=int(os.environ.get("GENERATION_TIMEOUT", "1200")))
    parser.add_argument("--eval-timeout", type=int, default=int(os.environ.get("EVAL_TIMEOUT", "1800")))
    parser.add_argument("--cell-timeout", type=int, default=int(os.environ.get("CELL_TIMEOUT", "7200")))
    parser.add_argument("--cell-attempts", type=int, default=int(os.environ.get("CELL_ATTEMPTS", "2")))
    parser.add_argument("--max-candidates", type=int, default=int(os.environ.get("MAX_CANDIDATES", "2")))
    parser.add_argument("--pareval-build-timeout", type=int, default=int(os.environ.get("PAREVAL_BUILD_TIMEOUT", "60")))
    parser.add_argument("--pareval-run-timeout", type=int, default=int(os.environ.get("PAREVAL_RUN_TIMEOUT", "180")))
    parser.add_argument("--sol-warmup", type=int, default=int(os.environ.get("SOL_WARMUP", "1")))
    parser.add_argument("--sol-iterations", type=int, default=int(os.environ.get("SOL_ITERATIONS", "3")))
    parser.add_argument("--sol-compile-timeout", type=int, default=int(os.environ.get("SOL_COMPILE_TIMEOUT", "180")))
    parser.add_argument("--sol-run-timeout", type=int, default=int(os.environ.get("SOL_RUN_TIMEOUT", "600")))
    parser.add_argument("--force", action="store_true", default=os.environ.get("FORCE", "0") == "1")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.root).resolve()
    run_root = root / "results/all_official_matrix_v1" / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)
    benchmarks = selected_benchmarks(args.benchmarks)
    agents = list_agents(root, args.agents)
    config = {"root": str(root), "run_root": str(run_root), "benchmarks": benchmarks, "agents": [dataclasses.asdict(agent) | {"driver": str(agent.driver)} for agent in agents], "args": vars(args), "generated_at": iso_now()}
    write_json(run_root / "run_config.json", config)
    if args.mode == "preflight":
        report = preflight(root, run_root, agents, benchmarks, args); print(json.dumps(report, indent=2, ensure_ascii=False)); return 0 if report["ok"] else 2
    if args.mode == "prepare":
        report = prepare_all(root, run_root, benchmarks); print(json.dumps(report, indent=2, ensure_ascii=False)); return 0 if report["all_have_tasks"] else 2
    if args.mode == "run":
        manifest = run_matrix(root, run_root, agents, benchmarks, args); print(json.dumps(manifest, indent=2, ensure_ascii=False)); return 0 if manifest["all_official_benchmarking_normal"] else 3
    if args.mode == "verify":
        manifest = collect_run(run_root, agents, benchmarks); print(json.dumps(manifest, indent=2, ensure_ascii=False)); return 0 if manifest["all_official_benchmarking_normal"] else 3
    if args.mode == "all":
        pre = preflight(root, run_root, agents, benchmarks, args)
        if not pre["ok"]:
            print(json.dumps(pre, indent=2, ensure_ascii=False)); return 2
        prep = prepare_all(root, run_root, benchmarks)
        if not prep["all_have_tasks"]:
            print(json.dumps(prep, indent=2, ensure_ascii=False)); return 2
        manifest = run_matrix(root, run_root, agents, benchmarks, args); print(json.dumps(manifest, indent=2, ensure_ascii=False)); return 0 if manifest["all_official_benchmarking_normal"] else 3
    if args.mode == "pack":
        archive = pack_run(root, run_root); print(f"UPLOAD_FILE={archive}"); return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
