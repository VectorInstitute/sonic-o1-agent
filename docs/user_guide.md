# User Guide

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPUs (4x recommended for tensor parallelism)
- [uv](https://docs.astral.sh/uv/) package manager (optional; you can use pip in a container)

### Setup

```bash
# Clone the repository
git clone https://github.com/VectorInstitute/sonic-o1-agent.git
cd sonic-o1-agent

# Install dependencies (with uv)
uv sync --all-extras --group docs --group test

# Or with pip (e.g. inside a Singularity container)
pip install -e ".[docs]"   # installs package + docs deps

# Verify setup
python scripts/verify_setup.py
```

### Building the documentation

From the project root, ensure the package is importable when MkDocs runs (required for API reference generation):

```bash
# With uv
uv run mkdocs build

# With system/conda Python: add src to PYTHONPATH
PYTHONPATH=src python -m mkdocs build
# Or: PYTHONPATH=src python -m mkdocs serve
```

---

## Usage Examples

### Basic Inference (Direct Mode)

The simplest mode uses the Planner and Temporal Index to produce a
grounded response in a single pass.

```bash
python scripts/run_agent.py \
  --video courtroom.mp4 --audio courtroom.m4a \
  --query "Summarize the key arguments"
```

### Chain-of-Thought Reasoning

Enables the Reasoner agent for step-by-step analysis with explicit
reasoning traces.

```bash
python scripts/run_agent.py \
  --video hearing.mp4 \
  --query "Analyze the legal strategy used" \
  --reasoning
```

### Self-Reflection

Enables the Reflection agent to evaluate response quality and
iteratively refine until the confidence threshold is met.

```bash
python scripts/run_agent.py \
  --video deposition.mp4 \
  --query "What inconsistencies exist in the testimony?" \
  --reflection
```

### Multi-Step Decomposition

Enables the advanced Planner to automatically decompose complex queries
into sequential sub-tasks.

```bash
python scripts/run_agent.py \
  --video trial.mp4 \
  --query "Compare the defense attorney's arguments vs the prosecutor's" \
  --multi-step
```

### Full Multi-Agent System

Enables all agents working together for maximum capability.

```bash
python scripts/run_agent.py \
  --video complex_case.mp4 \
  --query "Provide a comprehensive analysis with key contradictions" \
  --all-features \
  --verbose \
  --output results.json
```

### Temporal Queries (Auto-Segmentation)

The Planner agent automatically detects temporal references and segments
the video to process only the relevant window.

```bash
python scripts/run_agent.py \
  --video hearing.mp4 \
  --query "What happened between minute 5 and 10?"
```

---

## Configuration

All settings are in `configs/agent_config.yaml`.

### Model Settings

```yaml
model:
  model_path: "Qwen/Qwen3-Omni-30B-A3B-Instruct"
  use_thinking: false
  gpu_memory_utilization: 0.85
  tensor_parallel_size: 4        # Number of GPUs
  max_num_seqs: 1
  max_model_len: 65536

  generation_config:
    temperature: 0.0             # Greedy decoding
    top_p: 0.95
    top_k: 20
    max_new_tokens: 8192

  max_frames: 128
  min_frames: 64
```

### Processing Settings

```yaml
processing:
  max_frames: 128
  max_audio_chunks: null         # null = no chunking
  audio_chunk_duration_sec: 10.0
  min_video_duration_for_segmentation: 300  # 5 minutes
  segment_efficiency_threshold: 0.8
```

### Temporal Index Settings

Controls the frame-captioning index that provides temporal grounding.
See [Architecture -- Temporal Grounding](architecture.md#temporal-grounding)
for details on how this works.

```yaml
temporal_index:
  min_duration_sec: 180          # Skip indexing for short videos
  num_segments: 10               # Segments to split video into
  max_frames_per_segment: 16     # Frames per segment caption
  caption_max_tokens: 256        # Max tokens per caption
```

!!! note "Short videos"
    Videos shorter than `min_duration_sec` skip the temporal indexing step
    entirely and rely on sampling-context hints instead. The model can
    process the entire clip in a single pass without needing segment-level
    grounding.

### Reasoning Settings

```yaml
reasoning:
  max_reasoning_steps: 5
  enable_verification: true
```

### Reflection Settings

```yaml
reflection:
  confidence_threshold: 0.7      # Minimum confidence to accept
  max_refinement_attempts: 2
  use_iterative_refinement: false  # true = loop until ACCEPT or threshold
  check_hallucination: false       # true = run hallucination check after response
```

### Planning Settings

```yaml
planning:
  max_steps: 10   # Max sub-tasks when multi-step decomposes (e.g. compare/contrast)
```

---

## SLURM Submission

For cluster environments, use the provided SLURM script:

```bash
# Edit the script with your paths and resource requirements
vim slurm/run_sonic_agent_native.sh

# Submit
sbatch slurm/run_sonic_agent_native.sh

# Monitor output
tail -f logs/sonic_agent_*.out
```

---

## CLI Reference

```text
usage: run_agent.py [-h] [--config CONFIG] [--video VIDEO] [--audio AUDIO]
                    --query QUERY [--max-frames MAX_FRAMES]
                    [--max-audio-chunks MAX_AUDIO_CHUNKS]
                    [--reasoning] [--reflection] [--multi-step]
                    [--all-features] [--output OUTPUT] [--verbose] [--stream]

Arguments:
  --config        Path to config YAML (default: configs/agent_config.yaml)
  --video         Path to video file
  --audio         Path to audio file
  --query         Query/question to ask (required)
  --max-frames    Override max frames to process
  --max-audio-chunks  Override max audio chunks

Agent Modes:
  --reasoning     Enable Chain-of-Thought reasoning
  --reflection    Enable self-reflection and refinement
  --multi-step    Enable multi-step task decomposition
  --all-features  Enable all advanced features

Output:
  --output        Save results to JSON file
  --verbose       Show detailed reasoning trace
  --stream        Stream workflow progress (print each node as it runs; useful in job logs)
```

---

## Example Output

```text
======================================================================
SONIC-O1 AGENT RESPONSE
======================================================================

Query: Compare the defense attorney's arguments vs the prosecutor's

Mode: chain_of_thought

Evidence:
  video:
    duration_sec: 922.22
    frames_analyzed: 128
    sampling_interval_sec: 7.26
    coverage_sec: [0.0, 922.19]
  audio:
    duration_sec: 922.3
    chunks_analyzed: 93
    chunk_duration_sec: 10.0
    coverage_sec: [0.0, 922.3]

Reflection:
  Confidence: 0.85
  Scores: {'completeness': 9, 'accuracy': 8, 'clarity': 9, 'evidence': 8}

Response:
  Around 30s, the defense attorney opens by...
  Between 120s and 180s, the prosecutor counters with...
  [Detailed comparative analysis with timestamped evidence...]

======================================================================
```
