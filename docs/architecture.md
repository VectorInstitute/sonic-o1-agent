# Architecture

## Overview

SONIC-O1 is a **compound multi-agent system** built on
[LangGraph](https://github.com/langchain-ai/langgraph). Each component is a
specialised node in a state-machine workflow. The graph executes sequentially
with conditional branching based on user-selected agent modes.

---

## Multi-Agent Roles

| Agent | Module | Responsibility |
|-------|--------|----------------|
| **Planner** | `agents/planner.py` | Parse temporal references, detect query type, determine segmentation, compute video duration |
| **Temporal Index Builder** | `processors/temporal_index.py` | Split video into segments, caption each via the VLM, assemble a timestamped text index |
| **Reasoner** | `agents/reasoner.py` | Chain-of-Thought: understand query, plan approach, execute analysis, verify, refine |
| **Reflection** | `agents/reflection.py` | Evaluate response quality, score confidence, iteratively refine if below threshold |
| **Prompt Builder** | `processors/prompt_builder.py` | Construct query-type-aware prompts with temporal grounding directives |
| **Multimodal Engine** | `core/multimodal_engine.py` | Orchestration functions (backward compatible) |
| **Video Processor** | `core/video_processor.py` | Video frame sampling (PyAV), metadata extraction |
| **Audio Processor** | `core/audio_processor.py` | Audio loading with time-range slicing, chunking |
| **Multimodal Utils** | `core/multimodal_utils.py` | Shared utilities, constants, math helpers |

---

## Workflow Graph

```text
                    ┌──────────────┐
                    │   Planning   │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │ Segmentation │
                    └──────┬───────┘
                           │
                    ┌──────▼───────────┐
                    │ Temporal Indexing│
                    └──────┬───────────┘
                           │
            ┌──────────────┼──────────────┐
            │  route: multi_step /        │
            │  reasoning / direct         │
            └──┬───────────┬──────────┬───┘
               │           │          │
        ┌──────▼──┐  ┌─────▼────┐  ┌──▼──────┐
        │MultiStep│  │ Reasoning│  │  Direct │
        └──────┬──┘  └─────┬────┘  └──┬──────┘
               │           │          │
               └───────────┼──────────┘
                          │
                  ┌───────┴───────┐
                  │use_reflection?│
                  └───┬───────┬───┘
                      │       │
              ┌───────▼──┐  ┌─▼──────┐
              │Reflection│  │ Cleanup│
              └───────┬──┘  └───┬────┘
                      │         │
                  ┌───▼──────────▼───┐
                  │     Cleanup      │
                  └────────┬─────────┘
                           │
                          END
```

### State

All nodes read from and write to a shared `SonicState` TypedDict that flows
through the graph. Key fields include:

| Field | Type | Set By |
|-------|------|--------|
| `query` | `str` | User input |
| `plan` | `dict` | Planning node |
| `actual_video_path` | `str` | Segmentation node |
| `temporal_index` | `str` | Temporal indexing node |
| `response` | `str` | Reasoning or Direct node |
| `evidence` | `dict` | Direct node |
| `reflection` | `dict` | Reflection node |

---

## Temporal Grounding

Temporal grounding ensures the model cites **accurate timestamps in seconds**
rather than vague references. The strategy depends on video length.

### Short Videos (< `min_duration_sec`)

For videos below the configured threshold (default 180 s), the temporal
indexing step is skipped. The prompt builder injects a **sampling-context
hint** telling the model the video duration, frame count, and sampling
interval so it can estimate timestamps from visual progression.

### Long Videos (>= `min_duration_sec`)

A **frame-captioning index** is built before inference:

1. **Segment computation** -- The video is divided into N equal-length
   segments (capped by `num_segments`, minimum 30 s per segment).

2. **Per-segment captioning** -- For each segment, the VLM receives only
   the time-sliced video frames (`video_start` / `video_end`) **and** the
   matching audio chunk (`audio_start` / `audio_end`). This avoids loading
   the full file for every segment.

3. **Index assembly** -- Captions are combined into a plain-text index:

    ```text
    [0s - 92s] Speaker introduces the case and outlines the charges.
    [92s - 184s] Defense attorney presents opening statement.
    [184s - 277s] Prosecution shows exhibit A, a security recording.
    ...
    ```

4. **Prompt injection** -- The index is placed at the top of the main
   inference prompt, before the user query, with a directive:

    > *IMPORTANT: A timestamped content index of the video is provided
    > below. Use it to cite accurate timestamps in seconds when describing
    > events.*

This approach is inspired by the retrieve-then-read paradigm: expensive
multimodal perception is performed once per segment (cheap, parallel-ready),
and the downstream reasoning step operates on text -- which LLMs handle
with high fidelity.

### Audio Slicing

Both video and audio support time-range parameters:

- **Video**: `video_start` / `video_end` restrict frame decoding via
  `container.seek()` in PyAV.
- **Audio**: `audio_start` / `audio_end` map to the existing `offset` /
  `duration` parameters in `load_audio_pyav`, loading only the relevant
  samples.

When no time range is specified (the main inference pass), the full file
is loaded as before.

---

## Directory Structure

```text
sonic-o1-agent/
├── src/sonic_o1_agent/
│   ├── agents/                       # Multi-Agent System
│   │   ├── sonic_agent.py            # Main orchestrator
│   │   ├── planner.py                # Planning agent
│   │   ├── planner_advanced.py       # Multi-step decomposition
│   │   ├── reasoner.py               # Chain-of-Thought agent
│   │   └── reflection.py             # Self-reflection agent
│   ├── models/                       # Model wrappers
│   │   └── qwen_model.py             # Qwen3-Omni + vLLM
│   ├── core/                         # Multimodal processing
│   │   ├── multimodal_engine.py      # Orchestration (backward compatible)
│   │   ├── video_processor.py        # Video processing (PyAV)
│   │   ├── audio_processor.py       # Audio processing (PyAV)
│   │   └── multimodal_utils.py     # Shared utilities & constants
│   ├── processors/                   # Prompt engineering & indexing
│   │   ├── prompt_builder.py         # Dynamic prompt construction
│   │   └── temporal_index.py         # Frame-captioned segment grounding
│   ├── workflows/                    # LangGraph orchestration
│   │   ├── state.py                  # SonicState (workflow state schema)
│   │   ├── nodes.py                  # All workflow node functions
│   │   └── graph.py                  # StateGraph with conditional edges
│   └── utils/                        # Utilities
│       └── segmenter.py              # Video/audio segmentation (ffmpeg)
├── configs/
│   └── agent_config.yaml             # System configuration
├── scripts/
│   ├── run_agent.py                  # CLI entry point
│   └── verify_setup.py              # Setup verification
├── slurm/
│   └── run_sonic_agent_native.sh     # SLURM job script
├── docs/                             # MkDocs documentation
└── tests/                            # Unit & integration tests
```

---

## Model Backend

SONIC-O1 uses **Qwen3-Omni** served through **vLLM** for efficient
multi-GPU inference.

| Feature | Detail |
|---------|--------|
| Model | Qwen3-Omni-30B-A3B-Instruct |
| Serving | vLLM with tensor parallelism |
| Context | Up to 65 536 tokens |
| Modalities | Native video + audio (no transcription needed) |
| Decoding | Greedy (`temperature=0.0`) for deterministic output |
| Caching | Prefix caching enabled for follow-up efficiency |

The model wrapper (`qwen_model.py`) handles:

- Lazy loading and engine recovery after crashes or OOM
- Chat template formatting via `Qwen3OmniMoeProcessor`
- Text-only generation for intermediate reasoning steps
- Multimodal generation with time-range-aware video and audio slicing
