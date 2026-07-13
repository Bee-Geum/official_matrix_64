# Third-Party Components — Attribution, Licenses & Citations

This repository (`official_matrix_64`) is a research reproduction harness. It **vendors
(bundles) upstream open-source projects** under `third_party/` — GPU-kernel benchmarks and
kernel-generation agents. All bundled code **remains the property of its original authors and
is used under its original license**. The `.git` history of each vendored repo was removed to
reduce size; please refer to the upstream repositories below for full sources, licenses, and
citation details.

If you use this harness, please **cite the original benchmark/agent papers** (keys below refer
to the survey's bibliography) and comply with each component's license.

## Benchmarks (`third_party/`)

| Component | Upstream | License (bundled/known) | Cite |
|---|---|---|---|
| KernelBench | https://github.com/ScalingIntelligence/KernelBench | MIT | Ouyang 2025 |
| robust-kbench | https://github.com/SakanaAI/robust-kbench | see upstream | Lange 2025 |
| MultiKernelBench | https://github.com/wzzll123/MultiKernelBench | MIT | Wen 2025 |
| TritonBench (T & G) | https://github.com/thunlp/TritonBench | Apache-2.0 | Li 2025c |
| BackendBench | https://github.com/meta-pytorch/BackendBench | BSD (© Meta 2025) | Saroufim 2025 |
| ParEval | https://github.com/parallelcodefoundry/ParEval | see upstream (© Univ. of Maryland) | Nichols 2024 |
| SOL-ExecBench | https://github.com/sol-execbench/SOL-ExecBench | Apache-2.0 | Lin 2026 |

## Kernel-generation agents (`third_party/`)

| Component | Upstream | License (bundled/known) |
|---|---|---|
| CudaForge | https://github.com/OptimAI-Lab/CudaForge | MIT |
| AutoKernel | https://github.com/rightnow-ai/autokernel | MIT |
| CUDA-L1 | https://github.com/deepreinforce-ai/CUDA-L1 | MIT |
| AutoTriton | https://github.com/AI9Stars/AutoTriton — model: https://huggingface.co/ai9stars/AutoTriton | see upstream |
| Dr.Kernel / KernelGYM | https://github.com/hkust-nlp/KernelGYM — model: https://huggingface.co/hkust-nlp/drkernel-14b | see upstream |
| GEAK-agent | https://github.com/AMD-AGI/GEAK-agent | MIT |
| K-Search | https://github.com/caoshiyi/K-Search | Apache-2.0 |
| CUDA-Agent | https://github.com/BytedTsinghua-SIA/CUDA-Agent | see upstream |
| KernelLLM | https://huggingface.co/facebook/KernelLLM | see upstream (model license) |
| IndustrialCoder (InCoder-32B) | https://github.com/CSJianYang/Industrial-Coder — model: https://huggingface.co/Multilingual-Multimodal-NLP/IndustrialCoder | see upstream (model license) |
| KernelMem | https://github.com/0satan0/KernelMem | see upstream |

> "see upstream" = the bundled copy did not include a standalone `LICENSE` file at vendoring
> time; consult the upstream repository (or model card) for the authoritative license.

## Model weights
Model weights are **not** included in this repository (they live in a shared Hugging Face cache).
Each model is governed by its own license/terms on its Hugging Face model card — see the links above.

## This harness's own code
The reproduction glue in this repo (the runner `official_all_matrix_v1.py`, `telemetry/`, the
drivers in `drivers/`, `repro_env.sh`, and the reports in `docs/`) is authored for this project;
its use is at the discretion of the repository owner. The vendored `third_party/` content is **not**
covered by any license the owner may apply to their own code.
