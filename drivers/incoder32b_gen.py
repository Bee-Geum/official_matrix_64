#!/usr/bin/env python3
"""
incoder32b_gen.py -- FAITHFUL reproduction driver for incoder32b (IndustrialCoder-32B).

IndustrialCoder (Multilingual-Multimodal-NLP/IndustrialCoder, arch IQuestCoderForCausalLM) is a
custom 32B code model; vLLM has no native support, so we load it via HF transformers with
trust_remote_code and generate a CUDA ModelNew for a KernelBench task in-process. Model-only
faithful repro (registry lists incoder32b as a CUDA generator).

Usage:
  python3 incoder32b_gen.py --task <KernelBench_task.py> --out-cand <dir> [--max-new-tokens 4096]
"""
from __future__ import annotations
import argparse, os, re
from pathlib import Path

MODEL_ID = os.environ.get("INCODER_MODEL", "Multilingual-Multimodal-NLP/IndustrialCoder")


def extract_code(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    if blocks:
        for b in blocks:
            if "class ModelNew" in b:
                return b.strip()
        return max(blocks, key=len).strip()
    return text.strip() if "class ModelNew" in text else text.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--out-cand", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.6)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # transformers>=recent removed the "default" key from ROPE_INIT_FUNCTIONS; IQuestCoder's
    # custom modeling code still looks it up. Register the standard RoPE init so the model loads
    # (does not change model weights/behaviour -- restores the classic default rope formula).
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    if "default" not in ROPE_INIT_FUNCTIONS:
        def _default_rope(config, device=None, seq_len=None, **kwargs):
            base = getattr(config, "rope_theta", 10000.0)
            dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
            return inv_freq, 1.0
        ROPE_INIT_FUNCTIONS["default"] = _default_rope

    ref = Path(args.task).read_text(encoding="utf-8")
    user = (
        "You write custom CUDA kernels to replace the pytorch operators in the given architecture "
        "to get speedups. Produce a COMPLETE, self-contained Python module defining a class "
        "`ModelNew(torch.nn.Module)` that numerically matches the reference `Model` (atol=rtol=1e-2), "
        "using CUDA C++ via torch.utils.cpp_extension.load_inline. Return one python code block.\n\n"
        f"Reference model:\n```python\n{ref}\n```"
    )

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    if tok.chat_template:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True
        )
    else:
        prompt = user
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=True,
            temperature=args.temperature, top_p=0.95,
            eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    outd = Path(args.out_cand); outd.mkdir(parents=True, exist_ok=True)
    (outd / "round0_raw.txt").write_text(text, encoding="utf-8")
    code = extract_code(text)
    (outd / "round0_kernel.py").write_text(code, encoding="utf-8")
    print(f"[incoder32b] wrote round0_kernel.py (ModelNew present: {'class ModelNew' in code}, {len(code)} chars)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
