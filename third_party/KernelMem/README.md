## Project Overview: KernelMem

KernelMem is an **automatic CUDA kernel generation and optimization system based on PyTorch model code, enhanced with a "long–short term memory" mechanism**.  
The core idea is: starting from PyTorch forward code, the system uses an LLM to iteratively generate candidate CUDA kernels, and combines historical optimization experience, performance/correctness feedback, and expert knowledge about kernel optimization to form a "memory loop", continuously evolving faster kernels. The long-term memory component incorporates general knowledge and best practices for kernel optimization, enabling the system to leverage proven optimization strategies across different tasks.

The main entry point of the project is the `main()` function in `main_memory_latest.py` (triggered when the script is run directly).

---

## Key Features

- **Automatic migration from PyTorch operators / models to CUDA kernels**
  - Automatically reads operator / network definitions from PyTorch task scripts.
  - Builds LLM prompts according to the task and asks the model to generate corresponding CUDA kernels.

- **Multi-round self-evolution with “memory”**
  - For each kernel across rounds, the system records:
    - Correctness results (whether it runs, whether it passes numerical checks)
    - Performance metrics (speedup, NVIDIA Nsight Compute / Nsight Systems metrics, etc.)
    - Applied optimization strategies, failure reasons, repair history
  - These are written into `code/`, `evaluation/`, `profile/`, etc., and then fed back as short term memory to guide future kernel generation and repair.

- **Automatic benchmarking and error repair**
  - Uses `utils/compile_and_run.py` to compile and benchmark generated kernels:
    - Compares numerical errors against the reference PyTorch implementation (`tol`).
    - Measures average forward latency and computes **speedup = ref_latency / test_latency**.
  - For compilation errors / runtime errors / accuracy failures:
    - Builds “memory-aware” error analysis and repair prompts via `prompts/judger_repair_memory.py` and `prompts/error_memory.py`.
    - Asks the LLM to generate more reliable kernel versions based on historical error logs and repair records.

- **NCU & NSYS profiling–driven optimization**
  - Invokes NVIDIA Nsight Compute (`ncu`) via `run_ncu_memory.py` to obtain fine-grained performance metrics:
    - Memory efficiency, SM utilization, launch/occupancy, bottleneck stages, etc.
  - Invokes Nsight Systems (`nsys`) via `run_nsys.py` to measure kernel launch counts and runtime behavior.
  - These profiling results are converted into optimization suggestions by `prompts/judger_optimization_memory_latest.py` / `prompts/optimization_memory_latest.py`, then used to drive new kernel generations.

---

## Code Structure

- **`main_memory_latest.py`**: main entry of the project
  - Parses CLI arguments (task selection, GPU, LLM settings, number of rounds, etc.).
  - Calls the LLM to generate / repair / optimize kernels.
  - Orchestrates benchmarking, NCU/NSYS profiling, visualization, and summary.

- **`KernelBench/`**: PyTorch reference tasks
  - `level1`, `level2`: various basic operators and small subnetworks.
  - `level3`: representative deep learning models (ResNet, VGG, LSTM, Transformer, etc.).

- **`prompts/`**: prompt design and “memory mechanism”
  - `generate_custom_cuda_memory.py`: seed prompt for the first-round kernel generation.
  - `optimization_memory_latest.py`: optimization prompts that fuse historical kernels with profiling results.
  - `judger_*_memory*.py`: judge and analysis modules for optimization strategy, compilation timeouts, runtime errors, etc., which then produce repair/optimization suggestions.
  - `few_shot/`: few-shot examples for the LLM.

- **`memorybank/`**:
  - Stores prior knowledge about hardware bottlenecks and kernel structures.
  - These act as “long-term memory” and are injected into prompts to guide better optimization choices.

- **`utils/`**:
  - `compile_and_run.py`: compile, run, compare accuracy, and measure performance.
  - `kernel_io.py`: extract code blocks from LLM replies, save them as Python/CUDA files, and read/write metrics.

- **`agents/query_server.py`**:
  - Unified interface for talking to actual LLM backends (OpenAI, local vLLM/sglang, etc.).

---

## Environment Requirements

It is recommended to run the project on **Linux + NVIDIA GPU** (on Windows you need to prepare the CUDA toolchain and Nsight tools yourself).  
Typical dependencies (for reference; adjust versions to your environment):

- Python 3.9+
- PyTorch (with GPU support)
- CUDA Toolkit and matching drivers
- NVIDIA Nsight Compute (`ncu`) and Nsight Systems (`nsys`)
- Python packages:
  - `matplotlib`
  - `pandas`, `numpy` (for profiling CSV processing if needed)
  - SDK for your LLM service (e.g. `openai` or a custom HTTP client)

Using a virtualenv or Conda environment is strongly recommended.

---

## Quick Start

### 1. Install dependencies

In the project root, create a virtual environment and install required packages, for example:

```bash
conda create -n kernelmem python=3.10 -y
conda activate kernelmem

# Install dependencies as needed (example)
pip install torch matplotlib pandas numpy
# If using OpenAI models, also install: openai
```

Make sure `ncu` and `nsys` are available in your shell.

### 2. Run a single task

The most basic usage is to specify a PyTorch task script as `arch_py`:

```bash
python main_memory_latest.py KernelBench/level1/001_xxx.py \
  --gpu A100-80GB \
  --server_type openai \
  --server_address localhost \
  --server_port 8000 \
  --model_name gpt-5.1-chat \
  --round 10 \
  --work_dir run \
  --device 0
```

Key arguments:

- **`arch_py`**: path to a PyTorch task script, or to a directory containing multiple tasks.
- **`--gpu`**: GPU name used in prompts (does not change the actual device, only informs the LLM of hardware specs).
- **`--server_type` / `--server_address` / `--server_port` / `--model_name`**: LLM backend configuration.
- **`--round`**: total number of rounds per task (including seed generation, repair, and optimization).
- **`--device`**: CUDA device ID.
- **`--warmup` / `--repeat` / `--tol`**: warmup iterations, benchmark repetitions, and error tolerance.

### 3. Batch tasks and filtering

- Randomly sample tasks from a directory:

```bash
python main_memory_latest.py KernelBench/level3 \
  --num_tasks 5 --shuffle_seed 42
```

- Use `summary.json` from a previous run to only re-run tasks whose best kernel is still non-runnable:

```bash
python main_memory_latest.py KernelBench/level3 \
  --filter_from_summary path/to/previous/summary.json
```

---

## Outputs and Visualization


Example structure for a single task:

- `code/`: all kernels generated for this task (Python/CUDA), possibly with optimization/repair history JSON.
- `evaluation/`:
  - `llm_io/`: all prompts and raw LLM replies for each round.
  - Per-round metrics JSON: whether it is runnable, error type, speedup, etc.
- `figures/`:
  - `taskname_score.png`: speedup curve across rounds, with runnable/non-runnable points distinguished.
- `profile/`:
  - `*_ncu*.csv`: Nsight Compute metrics.
  - `*_nsys*.nsys-rep` / `*_nsys*.csv`: Nsight Systems traces and stats.
- `optimization_tree.json`:
  - A “genealogy” of all kernels for the task, with parent–child relationships, speedups, NCU status, and whether an optimization method was matched.
- `usage.csv`:
  - Token usage for all LLM calls, with a total row appended at the end.

For each batch directory, you will also get:

- `summary.json` / `summary.csv`: cross-task summary including average speedup, accuracy, and total tokens.

---

## Long–Short Term Memory Mechanism (Conceptual)

- **Short-term memory (local context)**:
  - Recently generated kernel snippets in the current run, recent error logs, and profiling results.
  - Constructed via helpers such as `_build_history_block` into Markdown code blocks, which are directly embedded into optimization prompts.
  - Historical artifacts such as `optimization_tree.json` and per-round `opt_round_*.json` / `repair_round_*.json`.

- **Long-term memory (cross-round / cross-task experience)**:
  - Prior knowledge stored under `memorybank/` (hardware bottlenecks, common kernel structures, feasible optimization strategies).


When generating, repairing, or optimizing kernels, the LLM consumes this memory as additional context so that it can:

- Avoid repeating the same compilation/runtime mistakes.
- Reuse optimization strategies that have worked in the past.
- Make more targeted design choices for specific hardware and operator patterns.

---

## Notes and Caveats

- The project frequently compiles and runs GPU kernels. Make sure your machine has sufficient GPU memory and proper timeout/monitoring to avoid hangs caused by buggy kernels.
- NCU / NSYS profiling can be time-consuming, especially for large-model tasks in `KernelBench/level3`. It is recommended to first debug the pipeline on small tasks with fewer rounds.


If you want to deploy or extend this project in your environment (e.g. connecting to your own LLM backend, adding new kernel templates / task sets), start by reading and modifying `main_memory_latest.py` and files under `prompts/`.
