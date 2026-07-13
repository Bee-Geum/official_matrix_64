#!/usr/bin/env python3
"""One-shot diagnostic: one Qwen generation under AutoKernel playbook -> save -> bench (verbose)."""
import os, re, subprocess, sys
from pathlib import Path
from openai import OpenAI

AK = Path("/home/bi_geum/official_matrix_64/third_party/autokernel")
KB = AK / "kernelbench"
KB_ACTIVE = AK / "workspace" / "kb_active"
KERNEL_PY = AK / "kernel.py"

playbook = (KB / "program_kb.md").read_text()
reference = (KB_ACTIVE / "reference.py").read_text()
starter = KERNEL_PY.read_text()

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
user = f"""You are optimizing a KernelBench problem. Produce a COMPLETE, self-contained
`kernel.py` that defines a `ModelNew(torch.nn.Module)` numerically matching the reference
`Model` (atol=rtol=1e-2) and faster than it. Use CUDA C++ via torch.utils.cpp_extension.load_inline,
or Triton @triton.jit. Return ONE python code block only.

# Reference
```python
{reference}
```
# Current kernel.py
```python
{starter}
```
Return the full new kernel.py now."""
r = client.chat.completions.create(model="qwen14b",
    messages=[{"role": "system", "content": playbook}, {"role": "user", "content": user}],
    max_tokens=6144, temperature=0.2, top_p=0.95)
raw = r.choices[0].message.content or ""
Path("/home/bi_geum/official_matrix_64/logs/_ak_diag_raw.txt").write_text(raw)
m = re.findall(r"```(?:python)?\s*(.*?)```", raw, re.DOTALL)
code = next((b for b in m if "class ModelNew" in b), (max(m, key=len) if m else raw)).strip()
KERNEL_PY.write_text(code, encoding="utf-8")
print("=== EXTRACTED kernel.py (first 40 lines) ===")
print("\n".join(code.splitlines()[:40]))
print("=== ... bench now ===", flush=True)

env = dict(os.environ)
env["TORCH_CUDA_ARCH_LIST"] = "12.0"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
env["TORCH_EXTENSIONS_DIR"] = "/home/bi_geum/official_matrix_64/results/repro/_extdir_ak_diag"
env["PYTHONPATH"] = str(AK) + os.pathsep + env.get("PYTHONPATH", "")
Path(env["TORCH_EXTENSIONS_DIR"]).mkdir(parents=True, exist_ok=True)
p = subprocess.run([sys.executable, str(KB / "bench_kb.py"), "--skip-stability", "--skip-determinism"],
                   cwd=str(AK), env=env, timeout=400, capture_output=True, text=True)
print("=== BENCH STDOUT (tail) ===")
print(p.stdout[-3000:])
print("=== BENCH STDERR (tail) ===")
print(p.stderr[-2000:])
