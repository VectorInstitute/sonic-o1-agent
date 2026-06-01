# SONIC-O1 Multi-Agent System
A compound multi-agent system for audio-video understanding built on Qwen3-Omni and vLLM.

[![website](https://img.shields.io/badge/website-ff00ff)](https://vectorinstitute.github.io/sonic-o1-agent/)
[![code checks](https://github.com/VectorInstitute/sonic-o1-agent/actions/workflows/code_checks.yml/badge.svg)](https://github.com/VectorInstitute/sonic-o1-agent/actions/workflows/code_checks.yml)
[![unit tests](https://github.com/VectorInstitute/sonic-o1-agent/actions/workflows/unit_tests.yml/badge.svg)](https://github.com/VectorInstitute/sonic-o1-agent/actions/workflows/unit_tests.yml)
[![integration tests](https://github.com/VectorInstitute/sonic-o1-agent/actions/workflows/integration_tests.yml/badge.svg)](https://github.com/VectorInstitute/sonic-o1-agent/actions/workflows/integration_tests.yml)
[![docs](https://github.com/VectorInstitute/sonic-o1-agent/actions/workflows/docs.yml/badge.svg)](https://github.com/VectorInstitute/sonic-o1-agent/actions/workflows/docs.yml)
[![License: Vector Institute](https://img.shields.io/badge/License-Apache2.0-003049.svg)](./LICENSE.md)
[![Contact](https://img.shields.io/badge/Contact-shaina.raza%40vectorinstitute.ai-green)](mailto:shaina.raza@vectorinstitute.ai)

---

## Overview

SONIC-O1 addresses the problem of deep, evidence-grounded understanding of long-form audio-video content. Rather than a single model call, it orchestrates a pipeline of specialized agents - each responsible for a distinct reasoning role — coordinated through a LangGraph workflow.

**Agents:**
- **Planner** — parses temporal references, detects relevant modalities, decides whether to segment the video to a sub-window, and optionally decomposes complex queries into ordered sub-tasks
- **Reasoner** — runs chain-of-thought analysis with explicit reasoning steps and self-verification
- **Reflection** — evaluates response quality, confidence-scores the output, and refines when recommended; supports iterative refinement and hallucination detection
- **Multimodal Engine** — memory-efficient video/audio processing via PyAV; builds a frame-captioned temporal index for longer videos to give the model accurate timestamp grounding

**Stack:** Qwen3-Omni (native video + audio), vLLM (tensor-parallel inference), LangGraph (workflow orchestration), PyAV (media processing).

```
sonic-o1-agent/
├── src/sonic_o1_agent/
│   ├── agents/
│   │   ├── sonic_agent.py         # Main orchestrator
│   │   ├── planner.py             # Planning agent
│   │   ├── planner_advanced.py    # Multi-step decomposition
│   │   ├── reasoner.py            # Chain-of-Thought agent
│   │   └── reflection.py          # Self-reflection agent
│   ├── models/
│   │   └── qwen_model.py          # Qwen3-Omni + vLLM wrapper (embedded & server mode)
│   ├── core/
│   │   ├── multimodal_engine.py   # Orchestration
│   │   ├── video_processor.py     # PyAV video processing
│   │   ├── audio_processor.py     # PyAV audio processing
│   │   └── multimodal_utils.py    # Shared constants & math helpers
│   ├── processors/
│   │   ├── prompt_builder.py      # Dynamic prompt construction
│   │   └── temporal_index.py      # Frame-captioned segment grounding
│   ├── workflows/
│   │   ├── state.py               # SonicState schema
│   │   ├── nodes.py               # LangGraph node functions
│   │   └── graph.py               # StateGraph with conditional edges
│   ├── api.py                     # FastAPI demo server
│   ├── demo/                      # Demo UI (HTML/CSS/JS)
│   └── utils/
│       └── segmenter.py           # ffmpeg-based segmentation
├── configs/
│   └── agent_config.yaml
├── scripts/
│   ├── run_agent.py
│   └── verify_setup.py
├── slurm/
│   └── run_sonic_agent_native.sh
└── tests/
```

---

## Installation

After cloning the repository:

```bash
git clone https://github.com/VectorInstitute/sonic-o1-agent.git
cd sonic-o1-agent
```

Install dependencies with [uv](https://docs.astral.sh/uv/). On network or cluster filesystems, use **`--link-mode=copy`** so uv does not hardlink across mounts (which can cause install errors):

```bash
uv sync --link-mode=copy
```

Install the `sonic-o1-agent` package and CLI entry point:

```bash
uv pip install .
```

Activate the virtual environment:

```bash
source .venv/bin/activate
```

If `which python` does not point to `.venv/bin/python` after activating, prepend the venv to your `PATH`:

```bash
export PATH="$(pwd)/.venv/bin:$PATH"
```

Verify the CLI:

```bash
sonic-o1-agent --help
# or: uv run sonic-o1-agent --help
```

---

## Embedded mode vs. vLLM server

SONIC-O1 supports two inference backends:

| Mode | When to use | Config |
|------|-------------|--------|
| **Embedded** (in-process) | No separate `vllm serve` process; model loads inside the agent | `vllm_base_url: ""` |
| **Server** | A `vllm serve` instance is already running (demo UI, multi-node) | `vllm_base_url: "http://<host>:<port>/v1"` |

**If you do not have a vLLM server running**, use **embedded mode**. Edit `configs/agent_config.yaml` under the `model:` section and set:

```yaml
model:
  vllm_base_url: ""   # empty string → embedded / in-process vLLM
```

The field is **`vllm_base_url`** (not `vllm_path`). It lives in `configs/agent_config.yaml` at `model.vllm_base_url`. You can also override it with the environment variable `VLLM_BASE_URL`.

When `vllm_base_url` is empty or unset, embedded settings apply (`tensor_parallel_size`, `gpu_memory_utilization`, etc.). When it is a URL, the agent sends requests to that server and ignores embedded load settings.

---

## Troubleshooting

### `World size (N) is larger than the number of available GPUs`

Embedded mode uses tensor parallelism across GPUs. The world size must match how many GPUs Slurm (or your session) actually allocated.

1. Check available GPUs:

```bash
nvidia-smi
```

2. Set **`tensor_parallel_size`** in `configs/agent_config.yaml` under `model:` to that count:

```yaml
model:
  tensor_parallel_size: 1   # 1 GPU  (e.g. srun --gres=gpu:a100:1)
  # tensor_parallel_size: 4   # 4 GPUs (e.g. srun --gres=gpu:a100:4)
```

| GPUs visible in `nvidia-smi` | `tensor_parallel_size` |
|------------------------------|-------------------------|
| 1 | `1` |
| 2 | `2` |
| 4 | `4` |

If you request 1 GPU in Slurm but leave `tensor_parallel_size: 4`, vLLM will fail with a world-size error.

---

## Prerequisites

### Model Weights

The model weights (~60GB for Qwen3-Omni-30B) must be available before running. Set the cache directory:

```bash
export HF_HOME=/path/to/model/cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
```

To download the weights:

```bash
huggingface-cli download Qwen/Qwen3-Omni-30B-A3B-Instruct
```

---

## Demo UI

> **[▶ Watch the demo video](https://drive.google.com/file/d/1e1qv4JCKqeDc7UdZTIx1-c39zqXNKWtr/view?usp=sharing)**

The demo provides a web interface for uploading videos and querying the multi-agent pipeline. It streams real-time progress as each agent completes — planning, temporal indexing, inference, and reflection — with a live thinking panel showing intermediate results.

### Running the Demo (vLLM Server Mode)

**1. Start the vLLM server** (on a machine with GPUs):

```bash
uv run vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8080 --host 0.0.0.0 \
  --dtype bfloat16 --max-model-len 32768 \
  --allowed-local-media-path / \
  -tp 4
```

Adjust `-tp` to match your GPU count. The model requires ~60GB VRAM; `-tp 4` (4×A40 or similar) is the recommended minimum.

**2. Start the demo server:**

```bash
cd sonic-o1-agent
PYTHONPATH=src uv run python -c "
from sonic_o1_agent.api import serve
serve(vllm_base_url='http://localhost:8080/v1', port=8000, config_path='configs/agent_config.yaml')
"
```

> **Note:** If the demo server and vLLM run on different machines (e.g. login node vs GPU node), replace `localhost` with the hostname of the machine running vLLM. The port in `vllm_base_url` must match `--port` above.

**3. Open in your browser:**

```
http://localhost:8000
```

If running on a remote server, forward the port:

```bash
ssh -L 8000:<node>:8000 user@server
```

### Server mode (demo / external vLLM)

Point `model.vllm_base_url` at your running server:

```yaml
model:
  vllm_base_url: "http://localhost:8080/v1"
  model_path: "Qwen/Qwen3-Omni-30B-A3B-Instruct"
```

Or via environment variable:

```bash
VLLM_BASE_URL=http://localhost:8080/v1 sonic-o1-agent \
  --config configs/agent_config.yaml \
  --video video.mp4 --query "Summarize the key points" --all-features
```

See [Embedded mode vs. vLLM server](#embedded-mode-vs-vllm-server) above for switching back to embedded mode.

---

## How It Works

```
User Query
    ↓
Planning       — parse temporal refs, detect modality, optionally decompose into sub-tasks
    ↓
Segmentation   — extract time window if query references a specific range
    ↓
Temporal Index — for longer videos: caption N segments → inject timestamped index into prompt
    ↓
Inference      — multi_step | reasoning | direct  (selected based on flags)
    ↓
Reflection     — evaluate quality, refine if needed, optional hallucination check
    ↓
Cleanup → Final Answer
```

**Agent modes:**

| Mode | Use when | Flag |
|---|---|---|
| Direct | Simple queries | *(default)* |
| Reasoning | Step-by-step analysis needed | `--reasoning` |
| Reflective | Quality-critical output | `--reflection` |
| Multi-Step | Comparisons, causal, multi-entity | `--multi-step` |
| Full | Maximum capability | `--all-features` |

---

## Configuration

`configs/agent_config.yaml`:

```yaml
model:
  model_path: "Qwen/Qwen3-Omni-30B-A3B-Instruct"
  use_thinking: false
  # vLLM backend mode (optional):
  #   - Set URL to use server mode (connects to running vllm serve)
  #   - Empty or omitted = embedded mode (loads model in-process)
  # vllm_base_url: "http://localhost:8080/v1"
  # vLLM settings
  gpu_memory_utilization: 0.85
  tensor_parallel_size: 4  # Number of GPUs
  max_num_seqs: 1
  max_model_len: 65536
  # Generation settings
  generation_config:
    temperature: 0.0
    top_p: 0.95
    top_k: 20
    max_new_tokens: 8192
  # Multimodal limits
  limit_mm_per_prompt:
    image: 1
    video: 1
    audio: 1
processing:
  max_frames: 64
  max_audio_chunks: 32
  audio_chunk_duration_sec: 10.0
  min_video_duration_for_segmentation: 300  # 5 minutes
  segment_efficiency_threshold: 0.8
reasoning:
  max_reasoning_steps: 5
  enable_verification: true
reflection:
  confidence_threshold: 0.7
  max_refinement_attempts: 2
  use_iterative_refinement: true
  check_hallucination: true
temporal_index:
  min_duration_sec: 180
  num_segments: 10
  max_frames_per_segment: 16
  caption_max_tokens: 128
planning:
  max_steps: 10
  enable_auto_decompose: true
```

---

## Research Applications

Legal analysis, medical review, education, compliance monitoring, qualitative research.

---

## Quick Start

```bash
# 1. Install (see Installation section)
uv sync --link-mode=copy && uv pip install .
source .venv/bin/activate

# 2. Configure embedded mode + GPU count
vim configs/agent_config.yaml   # vllm_base_url: "" and tensor_parallel_size

# 3. Run locally
sonic-o1-agent --video video.mp4 --audio audio.m4a \
  --query "Summarize the key points" --config configs/agent_config.yaml

# Or submit via SLURM (edit paths first)
vim slurm/run_sonic_agent_native.sh
sbatch slurm/run_sonic_agent_native.sh
tail -f logs/sonic_agent_*.out
```

---

## Usage

```bash
# Basic inference
sonic-o1-agent \
  --video hearing.mp4 --audio hearing.m4a \
  --query "Summarize the key arguments"

# Chain-of-thought reasoning
sonic-o1-agent --video hearing.mp4 \
  --query "Analyze the legal strategy" --reasoning

# Self-reflection
sonic-o1-agent --video hearing.mp4 \
  --query "What inconsistencies exist?" --reflection

# Multi-step decomposition
sonic-o1-agent --video trial.mp4 \
  --query "Compare defense vs prosecution arguments" --multi-step

# All agents + save output
sonic-o1-agent --video complex.mp4 \
  --query "Comprehensive analysis with contradictions" \
  --all-features --verbose --output results.json

# Temporal query (auto-segments the relevant window)
sonic-o1-agent --video hearing.mp4 \
  --query "What happened between minute 5 and 10?"
```

`python scripts/run_agent.py` with the same flags remains supported as a thin wrapper.

---

## Contact

**Shaina Raza** - shaina.raza@vectorinstitute.ai
Vector Institute for Artificial Intelligence

---

## Acknowledgments

Resources provided in part by the Province of Ontario, the Government of Canada through CIFAR, and companies sponsoring the Vector Institute.
Funded by the EU Horizon Europe programme - AIXPERT project (Grant No. 101214389).
Built on [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni), [vLLM](https://github.com/vllm-project/vllm), and the [Vector Institute AI Engineering Template](https://github.com/VectorInstitute/aieng-template-uv).
