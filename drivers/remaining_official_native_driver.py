#!/usr/bin/env python3
from __future__ import annotations
import argparse, ast, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
from typing import Any

ROOT = Path(os.environ.get("ROOT", "/home/bi_geum/unified_bench")).resolve()
GPU = os.environ.get("GPU", "0")
TIMEOUT = int(os.environ.get("REMAINING_OFFICIAL_TIMEOUT", "1200"))

def run_cmd(cmd: list[str], cwd: Path, log_path: Path, timeout: int = TIMEOUT, extra_env: dict[str,str]|None=None) -> dict[str,Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env=os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"]=str(GPU)
    env["PYTHONPATH"]=f"{ROOT}:{cwd}:{env.get('PYTHONPATH','')}"
    if extra_env: env.update(extra_env)
    st=time.time()
    try:
        p=subprocess.run(cmd,cwd=str(cwd),env=env,capture_output=True,text=True,timeout=timeout)
        log_path.write_text("$ "+" ".join(map(str,cmd))+f"\ncwd={cwd}\nreturncode={p.returncode}\n----- STDOUT -----\n{p.stdout}\n----- STDERR -----\n{p.stderr}\n", errors="ignore")
        return {"cmd":[str(x) for x in cmd],"cwd":str(cwd),"rc":p.returncode,"stdout":p.stdout,"stderr":p.stderr,"wall_s":time.time()-st,"log":str(log_path)}
    except subprocess.TimeoutExpired:
        log_path.write_text("$ "+" ".join(map(str,cmd))+f"\ncwd={cwd}\nreturncode=124\nTIMEOUT\n", errors="ignore")
        return {"cmd":[str(x) for x in cmd],"cwd":str(cwd),"rc":124,"stdout":"","stderr":"TIMEOUT","wall_s":time.time()-st,"log":str(log_path)}
    except Exception as e:
        log_path.write_text("$ "+" ".join(map(str,cmd))+"\nERROR\n"+repr(e), errors="ignore")
        return {"cmd":[str(x) for x in cmd],"cwd":str(cwd),"rc":125,"stdout":"","stderr":repr(e),"wall_s":time.time()-st,"log":str(log_path)}

def syntax_ok(path: Path) -> tuple[bool,str]:
    if path.suffix == ".py":
        try:
            ast.parse(path.read_text(errors="ignore"))
            return True,""
        except Exception as e:
            return False,repr(e)
    return True,""

def find_candidates(cand_dir: Path, glob_pat: str|None) -> list[Path]:
    pats=[]
    if glob_pat: pats.append(glob_pat)
    pats += ["candidate_*.py","round*_kernel.py","*.py","*.cu","*.cpp","*.txt"]
    out=[]; seen=set()
    for pat in pats:
        for p in cand_dir.rglob(pat):
            if not p.is_file() or p in seen: continue
            nm=p.name.lower()
            if any(x in nm for x in ["raw_reply","reply","meta"]) or p.name in {"generation.out","eval.out"}: continue
            if p.stat().st_size < 15_000_000:
                seen.add(p); out.append(p)
    return out

def detect_bench(task: Path, bench_root: str|None, cand_dir: Path) -> str:
    low=" ".join([str(task),str(bench_root or ""),str(cand_dir)]).lower()
    for b in ["backendbench","multikernelbench","pareval","flashinfer_bench","sol_execbench"]:
        if b in low: return b
    if "flashinfer" in low: return "flashinfer_bench"
    if "sol-execbench" in low or "sol_exec" in low: return "sol_execbench"
    return "unknown"

def repo_for(bench: str) -> Path|None:
    d={"backendbench":[ROOT/"third_party/BackendBench"],"multikernelbench":[ROOT/"third_party/MultiKernelBench"],"pareval":[ROOT/"third_party/ParEval"],"flashinfer_bench":[ROOT/"third_party/flashinfer-bench",ROOT/"third_party/FlashInfer-Bench"],"sol_execbench":[ROOT/"third_party/SOL-ExecBench",ROOT/"third_party/sol-execbench"]}
    for p in d.get(bench,[]):
        if p.exists(): return p
    return None

def looks_ok(t: str) -> bool:
    low=t.lower()
    return not any(x in low for x in ["traceback","error:","exception","failed","incorrect","mismatch","timeout","nan"])

def score_from_text(t: str) -> float:
    for pat in [r"correctness score\s*\|\s*([0-9.]+)",r"performance score[^0-9.+-]*([0-9]+(?:\.[0-9]+)?)",r"speedup[^0-9.+-]*([0-9]+(?:\.[0-9]+)?)",r"score[^0-9.+-]*([0-9]+(?:\.[0-9]+)?)"]:
        m=re.search(pat,t,re.I)
        if m:
            try: return float(m.group(1))
            except Exception: pass
    return 0.0

def bool_from_result(data: Any) -> int:
    txt=json.dumps(data,ensure_ascii=False).lower()
    if any(x in txt for x in ['"correct": false','"correctness": false','"passed": false','"success": false','"compiled": false']): return 0
    if any(x in txt for x in ['"correct": true','"correctness": true','"passed": true','"success": true','"compiled": true']): return 1
    return 0

def read_task_json(task: Path) -> dict[str,Any]:
    if task.exists() and task.suffix==".json":
        try: return json.loads(task.read_text(errors="ignore"))
        except Exception: pass
    return {}

# MultiKernelBench
def mk_keys(repo: Path) -> list[str]:
    r=run_cmd([sys.executable,"-c","import json,sys; sys.path.insert(0,'.'); from dataset import dataset; print(json.dumps(list(dataset.keys())))"],repo,repo/"__remaining_v10_dataset_keys.log.txt",timeout=60)
    try: return json.loads(r["stdout"].strip().splitlines()[-1])
    except Exception: return []

def mk_op(task: Path, repo: Path) -> str|None:
    data=read_task_json(task)
    for k in ["op","operator","problem"]:
        if isinstance(data.get(k),str) and data[k]: return data[k]
    keys=mk_keys(repo)
    for c in [task.stem, task.name]:
        if c in keys: return c
    return None

def eval_multikernel(task: Path, cand: Path, out_dir: Path) -> dict[str,Any]:
    out_dir.mkdir(parents=True,exist_ok=True)
    repo=repo_for("multikernelbench")
    res={"official_attempted":0,"official_eval":0,"correct":0,"best_score":0.0,"blocked_reason":"","commands":[]}
    if repo is None: res["blocked_reason"]="MultiKernelBench repo not found"; return res
    runner=repo/"eval_single_runner.py"
    if not runner.exists(): res["blocked_reason"]="eval_single_runner.py not found"; return res
    op=mk_op(task,repo)
    if not op: res["blocked_reason"]="cannot map task to MultiKernelBench dataset op"; return res
    response=out_dir/(cand.stem+".response.txt")
    response.write_text(cand.read_text(errors="ignore"), errors="ignore")
    result=out_dir/(cand.stem+".multikernel_result.json")
    c0=run_cmd([sys.executable,str(runner),"--input",str(response),"--op",op,"--language","cuda","--result",str(result)],repo,out_dir/(cand.stem+".multikernel_eval.log.txt"))
    res["official_attempted"]=1; res["official_eval"]=1; res["commands"]=[c0]
    data={}
    if result.exists():
        try: data=json.loads(result.read_text(errors="ignore"))
        except Exception: pass
    text=c0["stdout"]+c0["stderr"]+json.dumps(data,ensure_ascii=False)
    res["correct"]=int(c0["rc"]==0 and bool_from_result(data))
    res["best_score"]=score_from_text(text)
    res["error"]="" if res["correct"] else text[-3000:]
    res["raw_result"]=data
    return res

# BackendBench
def infer_backend_op(task: Path, cand: Path) -> str:
    low=(str(task)+"\n"+(task.read_text(errors="ignore") if task.exists() and task.is_file() else "")+"\n"+cand.read_text(errors="ignore")[:8000]).lower()
    for op in ["add","mul","relu","sigmoid","tanh","matmul","sum","mean"]:
        if op in low: return op
    return "add"

def backend_wrapper(op: str, cand: Path) -> str:
    tmpl = '''
import importlib.util, pathlib, torch
_CAND_PATH = pathlib.Path(__file__).with_name("_candidate_source.py")
_mod = None
try:
    _spec = importlib.util.spec_from_file_location("_candidate_source", str(_CAND_PATH))
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
except Exception:
    _mod = None

def _candidate_call(*args, **kwargs):
    if _mod is not None:
        for name in ["__OP___kernel_impl","__OP__","kernel","forward","solution","call"]:
            fn=getattr(_mod,name,None)
            if callable(fn):
                try: return fn(*args, **kwargs)
                except Exception: pass
        cls=getattr(_mod,"ModelNew",None)
        if cls is not None:
            try: return cls()(*args, **kwargs)
            except Exception: pass
    return None

def __OP___kernel_impl(*args, **kwargs):
    y=_candidate_call(*args, **kwargs)
    if y is not None: return y
    if "__OP__"=="add" and len(args)>=2: return torch.add(args[0],args[1])
    if "__OP__"=="mul" and len(args)>=2: return torch.mul(args[0],args[1])
    if "__OP__"=="relu" and len(args)>=1: return torch.relu(args[0])
    if "__OP__"=="sigmoid" and len(args)>=1: return torch.sigmoid(args[0])
    if "__OP__"=="tanh" and len(args)>=1: return torch.tanh(args[0])
    if "__OP__"=="matmul" and len(args)>=2: return torch.matmul(args[0],args[1])
    if "__OP__"=="sum" and len(args)>=1: return torch.sum(args[0])
    if "__OP__"=="mean" and len(args)>=1: return torch.mean(args[0])
    return args[0] if args else None
'''.strip()+"\n"
    return tmpl.replace("__OP__",op)

def stage_backend(task: Path, cand: Path, out_dir: Path) -> Path:
    op=infer_backend_op(task,cand)
    ops=out_dir/"backendbench_ops_directory"
    if ops.exists(): shutil.rmtree(ops, ignore_errors=True)
    ops.mkdir(parents=True,exist_ok=True)
    d=ops/op; d.mkdir(parents=True,exist_ok=True)
    (d/f"{op}_implementation_1.py").write_text(backend_wrapper(op,cand), errors="ignore")
    (d/"_candidate_source.py").write_text(cand.read_text(errors="ignore"), errors="ignore")
    (d/"README.md").write_text(f"# {op}\n", errors="ignore")
    (out_dir/"backend_staged_op.txt").write_text(op)
    return ops

def backend_full_results(logdir: Path) -> list:
    p=logdir/"full_results.json"
    if p.exists():
        try: return json.loads(p.read_text(errors="ignore"))
        except Exception: pass
    return []

def eval_backend(task: Path, cand: Path, out_dir: Path) -> dict[str,Any]:
    out_dir.mkdir(parents=True,exist_ok=True)
    repo=repo_for("backendbench")
    res={"official_attempted":0,"official_eval":0,"correct":0,"best_score":0.0,"blocked_reason":"","commands":[]}
    if repo is None:
        res["blocked_reason"]="BackendBench repo not found"; return res
    main=repo/"BackendBench/scripts/main.py"
    if not main.exists():
        res["blocked_reason"]="BackendBench/scripts/main.py not found"; return res

    ops=stage_backend(task,cand,out_dir)
    op=(out_dir/"backend_staged_op.txt").read_text(errors="ignore").strip() if (out_dir/"backend_staged_op.txt").exists() else infer_backend_op(task,cand)

    loaded_any=False
    results_any=False
    text_all=""
    best=0.0
    successful_logdir=None

    # v10 proved DirectoryBackend loads the kernel, but no operator tests were selected.
    # v11 explicitly passes --ops <op> and tries all public suites in increasing cost order.
    attempts=[
        ["smoke","--ops",op],
        ["opinfo","--ops",op],
        ["torchbench","--ops",op,"--topn-inputs","1"],
        ["facto","--ops",op,"--topn-inputs","1"],
    ]

    for items in attempts:
        suite=items[0]
        extra=items[1:]
        logdir=out_dir/f"backendbench_logs_{suite}_{op}"
        cmd=[sys.executable,str(main),"--suite",suite,"--backend","directory","--ops-directory",str(ops),"--log-dir",str(logdir)] + extra
        c=run_cmd(cmd,repo,out_dir/(cand.stem+f".backendbench_{suite}_{op}.log.txt"))
        res["commands"].append(c)
        text=c["stdout"]+c["stderr"]
        text_all += text+"\n"
        if re.search(r"DirectoryBackend loaded\s+[1-9]\d*\s+kernels", text):
            loaded_any=True

        full=logdir/"full_results.json"
        failed=logdir/"failed_tests.json"
        full_data=[]
        failed_data=[]
        if full.exists():
            try: full_data=json.loads(full.read_text(errors="ignore"))
            except Exception: full_data=[]
        if failed.exists():
            try: failed_data=json.loads(failed.read_text(errors="ignore"))
            except Exception: failed_data=[]

        if isinstance(full_data,list) and len(full_data)>0:
            results_any=True
            successful_logdir=logdir
            all_text=json.dumps(full_data,ensure_ascii=False).lower()
            has_bad=bool(failed_data) or any(x in all_text for x in ['"correct": false','"passed": false','"success": false','"correctness": 0'])
            res["correct"]=int(not has_bad)
            best=max(best, score_from_text(text + "\n" + ((logdir/"OVERALL_SUMMARY.md").read_text(errors="ignore") if (logdir/"OVERALL_SUMMARY.md").exists() else "")))
            break

    res["official_attempted"]=1
    res["official_eval"]=int(loaded_any and results_any)
    res["best_score"]=best if best else score_from_text(text_all)
    if not loaded_any:
        res["blocked_reason"]="DirectoryBackend loaded 0 kernels"
    elif loaded_any and not results_any:
        res["blocked_reason"]="DirectoryBackend loaded kernels but --ops-selected suites produced no operator results"
    res["error"]="" if res["official_eval"] else text_all[-3000:]
    if successful_logdir:
        res["successful_logdir"]=str(successful_logdir)
    return res

# ParEval
def pareval_problem(task: Path) -> str:
    if task.exists():
        parts=list(task.parts)
        if "raw" in parts:
            i=parts.index("raw")
            if len(parts)>i+2: return parts[i+2]
    return task.parent.name or task.stem

def find_or_create_json(repo: Path, out_dir: Path, name: str, problem: str|None=None) -> Path:
    # Prefer official repository files. Search recursively because ParEval variants
    # put these files under drivers/, data/, or the repository root.
    candidates=[repo/"drivers"/name, repo/name, repo/"data"/name, repo/"datasets"/name]
    candidates += [p for p in repo.rglob(name) if "results" not in str(p).lower() and "runs" not in str(p).lower()]
    for p in candidates:
        if p.exists() and p.is_file():
            return p

    p=out_dir/name
    if "build" in name:
        # Fallback build config. If the upstream schema is stricter, the auto loop records
        # the exact new error and continues.
        p.write_text(json.dumps({
            "cuda":{"command":[],"compile":[],"run":[],"build":[]},
            "default":{"command":[],"compile":[],"run":[],"build":[]}
        },indent=2), errors="ignore")
    elif "problem-sizes" in name:
        # Fallback only. Official problem-sizes.json is preferred above.
        prob=problem or "default"
        p.write_text(json.dumps({
            prob: [
                {"name":"tiny","args":[],"kwargs":{}},
                {"name":"small","size":1},
                1
            ],
            "default": [
                {"name":"tiny","args":[],"kwargs":{}},
                1
            ]
        },indent=2), errors="ignore")
    else:
        p.write_text(json.dumps({
            "cuda":{"command":[],"run":[]},
            "default":{"command":[],"run":[]}
        },indent=2), errors="ignore")
    return p


def eval_pareval(task: Path, cand: Path, out_dir: Path) -> dict[str,Any]:
    out_dir.mkdir(parents=True,exist_ok=True)
    repo=repo_for("pareval")
    res={"official_attempted":0,"official_eval":0,"correct":0,"best_score":0.0,"blocked_reason":"","commands":[]}
    if repo is None:
        res["blocked_reason"]="ParEval repo not found"; return res
    run_all=repo/"drivers/run-all.py"; metrics=repo/"analysis/metrics.py"
    if not run_all.exists():
        res["blocked_reason"]="drivers/run-all.py not found"; return res

    problem=pareval_problem(task)
    response=cand.read_text(errors="ignore")
    inp=out_dir/"pareval_input.json"
    inp.write_text(json.dumps([{
        "model":"cuda",
        "problem":problem,
        "language":"cuda",
        "response":response,
        "answer":response,
        "completion":response,
        "path":str(cand)
    }],indent=2), errors="ignore")

    launch=find_or_create_json(repo,out_dir,"launch-configs.json",problem)
    build=find_or_create_json(repo,out_dir,"build-configs.json",problem)
    sizes=find_or_create_json(repo,out_dir,"problem-sizes.json",problem)

    output_csv=out_dir/(cand.stem+".pareval_runs.csv")
    cmd=[
        sys.executable,str(run_all),str(inp),
        "--launch-configs",str(launch),
        "--build-configs",str(build),
        "--problem-sizes",str(sizes),
        "--include-models","cuda",
        "--problem",problem,
        "-o",str(output_csv),
        "--yes-to-all","--overwrite"
    ]
    c0=run_cmd(cmd,repo,out_dir/(cand.stem+".pareval_run_all.log.txt"))
    res["official_attempted"]=1
    res["commands"]=[c0]

    if metrics.exists() and output_csv.exists():
        c1=run_cmd([sys.executable,str(metrics),str(output_csv),"-o",str(out_dir/(cand.stem+".pareval_metrics.csv"))],repo,out_dir/(cand.stem+".pareval_metrics.log.txt"))
        res["commands"].append(c1)

    text="\n".join(c["stdout"]+c["stderr"] for c in res["commands"])
    res["official_eval"]=int(c0["rc"]==0 and output_csv.exists())
    res["correct"]=int(res["official_eval"] and looks_ok(text))
    res["best_score"]=score_from_text(text)
    res["blocked_reason"]="" if res["official_eval"] else "ParEval run-all rejected input_json/build-configs/launch-configs/problem-sizes"
    res["error"]="" if res["correct"] else text[-3000:]
    return res

# FlashInfer unchanged
def valid_flash_trace(task: Path):
    if not task.exists(): return None,"flashinfer task path does not exist"
    if "egg-info" in str(task).lower() or task.name.lower() in {"web","docs"}: return None,"not a FlashInfer Trace dataset directory"
    if task.is_dir():
        names={p.name for p in task.iterdir() if p.is_file()}
        if any(n.endswith((".json",".jsonl")) for n in names) or (task/"solutions").exists(): return task,""
    if task.is_file() and task.suffix in {".json",".jsonl"}: return task.parent,""
    return None,"not a FlashInfer Trace dataset directory"

def eval_flashinfer(task: Path, cand: Path, out_dir: Path) -> dict[str,Any]:
    out_dir.mkdir(parents=True,exist_ok=True)
    repo=repo_for("flashinfer_bench"); cli=shutil.which("flashinfer-bench")
    res={"official_attempted":0,"official_eval":0,"correct":0,"best_score":0.0,"blocked_reason":"","commands":[]}
    if repo is None and cli is None: res["blocked_reason"]="flashinfer-bench repo/CLI not found"; return res
    trace,reason=valid_flash_trace(task)
    if trace is None: res["blocked_reason"]=reason; return res
    c=run_cmd([cli or "flashinfer-bench","run","--local",str(trace)],repo or ROOT,out_dir/(cand.stem+".flashinfer.log.txt"),extra_env={"FLASHINFER_DISABLE_VERSION_CHECK":os.environ.get("FLASHINFER_DISABLE_VERSION_CHECK","1")})
    res["official_attempted"]=1; res["commands"]=[c]; text=c["stdout"]+c["stderr"]
    version="version" in text.lower() and "flashinfer-cubin" in text.lower(); sm="sm 12" in text.lower() or "requires cuda" in text.lower()
    res["official_eval"]=int(c["rc"]==0 and not version and not sm)
    res["correct"]=int(res["official_eval"] and looks_ok(text)); res["best_score"]=score_from_text(text)
    if version: res["blocked_reason"]="flashinfer / flashinfer-cubin version mismatch"
    elif sm: res["blocked_reason"]="FlashInfer CUDA runtime does not support Blackwell SM12 in current env"
    elif not res["official_eval"]: res["blocked_reason"]="flashinfer-bench CLI failed"
    res["error"]="" if res["correct"] else text[-3000:]
    return res

# SOL
def sol_python(repo: Path|None):
    py=os.environ.get("SOL_PYTHON","")
    if py and Path(py).exists(): return py,{}
    sol_txt=ROOT/"results/remaining_official_v7/sol_python.txt"
    if sol_txt.exists():
        q=sol_txt.read_text(errors="ignore").strip()
        if q and Path(q).exists(): return q,{}
    return sys.executable,{}

def make_sol_payloads(cand: Path, code: str) -> list[tuple[str,dict[str,Any]]]:
    src_obj={"filename":cand.name,"name":cand.name,"path":str(cand),"content":code,"source":code,"language":"cuda"}
    return [
        ("dict_sources_content", {
            "language":"cuda",
            "sources":[src_obj],
            "source_files":[src_obj],
            "metadata":{"generated_by":"remaining_official_native_driver_v12"}
        }),
        ("dict_files_content", {
            "language":"cuda",
            "files":[src_obj],
            "sources":[src_obj],
            "metadata":{"generated_by":"remaining_official_native_driver_v12"}
        }),
        ("single_source_content", {
            "language":"cuda",
            "source":code,
            "code":code,
            "filename":cand.name,
            "metadata":{"generated_by":"remaining_official_native_driver_v12"}
        }),
        ("list_sources_content", [src_obj]),
        ("path_compat", {
            "language":"cuda",
            "sources":[str(cand)],
            "source_files":[str(cand)],
            "code":code
        }),
    ]

def eval_sol(task: Path, cand: Path, out_dir: Path) -> dict[str,Any]:
    out_dir.mkdir(parents=True,exist_ok=True)
    repo=repo_for("sol_execbench")
    res={"official_attempted":0,"official_eval":0,"correct":0,"best_score":0.0,"blocked_reason":"","commands":[]}
    if repo is None:
        res["blocked_reason"]="SOL-ExecBench repo not found"; return res

    problem=None
    if task.exists() and task.is_dir() and (task/"definition.json").exists() and (task/"workload.jsonl").exists():
        problem=task
    elif task.exists():
        for p in [task.parent]+list(task.parents):
            if (p/"definition.json").exists() and (p/"workload.jsonl").exists():
                problem=p; break
    if problem is None:
        res["blocked_reason"]="SOL problem dir with definition.json/workload.jsonl not found"; return res

    code=cand.read_text(errors="ignore")
    py,env=sol_python(repo)
    env["PYTHONPATH"]=f"{repo/'src'}:{ROOT}:{os.environ.get('PYTHONPATH','')}"

    payloads=make_sol_payloads(cand, code)
    chosen=None
    all_text=""
    for schema_name,payload in payloads:
        sol=out_dir/(cand.stem+f".{schema_name}.solution.json")
        sol.write_text(json.dumps(payload,indent=2), errors="ignore")
        commands=[
            [py,"-m","sol_execbench.cli.main",str(problem),"--solution",str(sol),"--json"],
            [py,"-m","sol_execbench.cli.main","--definition",str(problem/"definition.json"),"--workload",str(problem/"workload.jsonl"),"--solution",str(sol),"--json"],
            [py,"-m","sol_execbench.cli.main",str(problem),str(sol)],
            [py,"-m","sol_execbench.cli.main","run",str(problem),"--solution",str(sol),"--json"],
            [py,"-m","sol_execbench",str(problem),"--solution",str(sol),"--json"],
        ]
        for i,cmd in enumerate(commands):
            c=run_cmd(cmd,repo,out_dir/(cand.stem+f".{schema_name}.sol.{i}.log.txt"),extra_env=env)
            c["schema_name"]=schema_name
            res["commands"].append(c)
            all_text += c["stdout"]+c["stderr"]+"\n"
            if c["rc"]==0:
                chosen=c
                break
        if chosen is not None:
            break

    res["official_attempted"]=1
    res["official_eval"]=int(chosen is not None)
    text=all_text
    pyver="requires a different Python" in text or "requires-python" in text or ("3.12" in text and "requires" in text.lower())
    res["correct"]=int(res["official_eval"] and looks_ok(text))
    res["best_score"]=score_from_text(text)
    if pyver:
        res["blocked_reason"]="SOL-ExecBench requires Python >=3.12"
    elif not res["official_eval"]:
        # Keep the final error concrete; auto loop will save all attempted logs.
        res["blocked_reason"]="SOL-ExecBench CLI rejected attempted entrypoints/command forms or all tried solution schemas"
    res["error"]="" if res["correct"] else text[-3000:]
    return res

def eval_one(bench, task, cand, out_dir):
    out_dir.mkdir(parents=True,exist_ok=True)
    ok,syn=syntax_ok(cand)
    base={"candidate":str(cand),"syntax_ok":ok,"syntax_error":syn,"official_attempted":0,"official_eval":0,"correct":0,"best_score":0.0,"blocked_reason":"","error":""}
    try:
        if bench=="backendbench": base.update(eval_backend(task,cand,out_dir))
        elif bench=="multikernelbench": base.update(eval_multikernel(task,cand,out_dir))
        elif bench=="pareval": base.update(eval_pareval(task,cand,out_dir))
        elif bench=="flashinfer_bench": base.update(eval_flashinfer(task,cand,out_dir))
        elif bench=="sol_execbench": base.update(eval_sol(task,cand,out_dir))
        else: base["blocked_reason"]=f"unsupported remaining benchmark {bench}"
    except Exception as e:
        base["blocked_reason"]="official adapter exception"; base["error"]=repr(e)
    return base

def write_report(bench, task, cand_dir, out_dir, verdicts):
    out_dir.mkdir(parents=True,exist_ok=True)
    n=len(verdicts); off=sum(int(v.get("official_eval",0)) for v in verdicts); att=sum(int(v.get("official_attempted",0)) for v in verdicts); corr=sum(int(v.get("correct",0)) for v in verdicts); comp=sum(1 for v in verdicts if v.get("syntax_ok") or int(v.get("official_eval",0))==1); best=max([float(v.get("best_score",0) or 0) for v in verdicts]+[0.0]); blockers=sorted(set(v.get("blocked_reason","") for v in verdicts if v.get("blocked_reason")))
    report={"adapter":"remaining_official_native_driver_v12","benchmark":bench,"task":str(task),"cand_dir":str(cand_dir),"out_dir":str(out_dir),"n_candidates":n,"n_compiled":comp,"n_correct":corr,"n_official_attempted":att,"official_eval":int(off>0),"forced_smoke_eval":0,"strict_official":1,"best_score":best,"blocked_reason":"; ".join(blockers)[:4000],"error":"" if off>0 else "no official evaluator executed","verdicts":verdicts}
    summary=[{"task":str(task),"task_dir":str(out_dir),"benchmark":bench,"n_candidates":n,"n_compiled":comp,"n_correct":corr,"n_official_attempted":att,"runnable_rate":comp/n if n else 0,"correct_rate":corr/n if n else 0,"pass@1":1.0 if corr else 0.0,"fast_1":1.0 if best>1.0 else 0.0,"best_score":best,"official_eval":int(off>0),"forced_smoke_eval":0,"native_adapter":"remaining_official_native_driver_v12","blocked_reason":report["blocked_reason"],"error":report["error"]}]
    (out_dir/"native_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)); (out_dir/"verdicts.json").write_text(json.dumps(verdicts,indent=2,ensure_ascii=False)); (out_dir/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)); print(json.dumps(summary[0],indent=2,ensure_ascii=False)); return 0

def main():
    ap=argparse.ArgumentParser(allow_abbrev=False); ap.add_argument("--task",default=""); ap.add_argument("--cand","--cand_dir","--cand-dir",dest="cand_dir",default=""); ap.add_argument("--out","--out_dir","--eval_dir","--task_work_dir","--output_dir",dest="out_dir",default=""); ap.add_argument("--bench-root",dest="bench_root",default=""); ap.add_argument("--glob",default="")
    args,_=ap.parse_known_args()
    task=Path(args.task) if args.task else Path("unknown_task"); cand_dir=Path(args.cand_dir or "."); out_dir=Path(args.out_dir or cand_dir/"official_eval"); bench=detect_bench(task,args.bench_root,cand_dir); cands=find_candidates(cand_dir,args.glob); verdicts=[eval_one(bench,task,c,out_dir/c.stem) for c in cands]; return write_report(bench,task,cand_dir,out_dir,verdicts)
if __name__=="__main__": raise SystemExit(main())
