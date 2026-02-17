# SONIC-O1 Multi-Agent System

A compound multi-agent system for audio-video understanding built on Qwen3-Omni and vLLM.

---

## Overview

SONIC-O1 addresses the problem of deep, evidence-grounded understanding of long-form audio-video content. Rather than a single model call, it orchestrates a pipeline of specialized agents — each responsible for a distinct reasoning role — coordinated through a LangGraph workflow.

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
│   │   └── qwen_model.py          # Qwen3-Omni + vLLM wrapper
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
  tensor_parallel_size: 4
  max_frames: 128

processing:
  max_audio_chunks: null
  min_video_duration_for_segmentation: 300  # seconds

temporal_index:
  min_duration_sec: 180       # skip indexing for shorter videos
  num_segments: 10
  max_frames_per_segment: 16
  caption_max_tokens: 256

reasoning:
  max_reasoning_steps: 5
  enable_verification: true

reflection:
  confidence_threshold: 0.7
  max_refinement_attempts: 2
  use_iterative_refinement: false   # loop until ACCEPT or threshold
  check_hallucination: false        # post-response hallucination check

planning:
  max_steps: 10
```

---

## Research Applications

Legal analysis, medical review, education, compliance monitoring, qualitative research.

---

## Quick Start

```bash
# 1. Set your model path
vim configs/agent_config.yaml

# 2. Edit paths in the SLURM script
vim slurm/run_sonic_agent_native.sh

# 3. Submit
sbatch slurm/run_sonic_agent_native.sh

# 4. Monitor
tail -f logs/sonic_agent_*.out
```

---

## Usage

```bash
# Basic inference
python scripts/run_agent.py \
  --video hearing.mp4 --audio hearing.m4a \
  --query "Summarize the key arguments"

# Chain-of-thought reasoning
python scripts/run_agent.py --video hearing.mp4 \
  --query "Analyze the legal strategy" --reasoning

# Self-reflection
python scripts/run_agent.py --video hearing.mp4 \
  --query "What inconsistencies exist?" --reflection

# Multi-step decomposition
python scripts/run_agent.py --video trial.mp4 \
  --query "Compare defense vs prosecution arguments" --multi-step

# All agents + save output
python scripts/run_agent.py --video complex.mp4 \
  --query "Comprehensive analysis with contradictions" \
  --all-features --verbose --output results.json

# Temporal query (auto-segments the relevant window)
python scripts/run_agent.py --video hearing.mp4 \
  --query "What happened between minute 5 and 10?"
```

---

## Citation

```bibtex
@software{sonic_o1_multi_agent,
  author    = {Radwan, Ahmed Y.},
  title     = {Sonic O1: A Multi-Agent System for Audio-Video Understanding},
  year      = {2026},
  publisher = {Vector Institute},
}
```

---

## Contact

**Ahmed Y. Radwan** — ahmed.radwan@vectorinstitute.ai
Vector Institute for Artificial Intelligence


---

## Acknowledgments

Resources provided in part by the Province of Ontario, the Government of Canada through CIFAR, and companies sponsoring the Vector Institute.
Funded by the EU Horizon Europe programme — AIXPERT project (Grant No. 101214389).

Built on [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni), [vLLM](https://github.com/vllm-project/vllm), and the [Vector Institute AI Engineering Template](https://github.com/VectorInstitute/aieng-template-uv).
