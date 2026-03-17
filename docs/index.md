# SONIC-O1 Multi-Agent System

**A compound multi-agent system for audio-video understanding with Qwen3-Omni.**

SONIC-O1 is **not a single model** -- it is a **coordinated system of specialized agents**
that plan, reason, ground, and reflect to produce accurate, temporally-grounded answers
from video and audio content.

---

## Demo

<video src="https://drive.google.com/file/d/1e1qv4JCKqeDc7UdZTIx1-c39zqXNKWtr/preview" width="100%" controls></video>

> **[▶ Watch the full demo video](https://drive.google.com/file/d/1e1qv4JCKqeDc7UdZTIx1-c39zqXNKWtr/view?usp=sharing)** — Upload a video, ask a question, and watch the multi-agent pipeline analyze it in real-time with planning, temporal indexing, chain-of-thought reasoning, and self-reflection.

The demo UI connects to a vLLM server and streams progress as each agent completes — see the [User Guide](user_guide.md#demo-ui) for setup instructions.

---

## Key Features

<div class="grid cards" markdown>

- :material-account-group: **Multi-Agent Coordination**

    ---
    Specialized agents for planning, reasoning, and reflection work together
    through a LangGraph workflow with conditional branching.

- :material-brain: **Chain-of-Thought Reasoning**

    ---
    Step-by-step analysis with explicit reasoning traces, self-verification
    at each step, and confidence scoring.

- :material-clock-outline: **Temporal Grounding**

    ---
    Frame-captioning index splits video into segments, captions each with
    time-sliced video and audio, and injects a timestamped index into
    prompts for accurate second-level citations.

- :material-magnify: **Self-Reflection**

    ---
    Automatic quality assessment with iterative refinement until a
    confidence threshold is met. Detects gaps and hallucinations.

- :material-format-list-numbered: **Multi-Step Planning**

    ---
    Automatic decomposition of complex queries into sub-tasks with
    sequential execution and context passing.

- :material-lightning-bolt: **Efficient Processing**

    ---
    vLLM backend with tensor parallelism, prompt caching, smart video
    segmentation, and per-segment audio/video slicing. Supports both
    embedded inference and a decoupled server mode with parallel
    temporal indexing.

</div>

---

## Quick Start

```bash
# Edit config with your model path
vim configs/agent_config.yaml

# Edit SLURM script with your video/audio paths
vim slurm/run_sonic_agent_native.sh

# Submit job
sbatch slurm/run_sonic_agent_native.sh

# Monitor
tail -f logs/sonic_agent_*.out
```

See the [User Guide](user_guide.md) for detailed usage examples and
configuration options, or the [Architecture](architecture.md) page for
a deep dive into the multi-agent workflow and temporal grounding strategy.

---

## Agent Modes

| Mode | Agents Active | Use Case | Flag |
|------|--------------|----------|------|
| **Direct** | Planner + Temporal Index | Simple queries | *(default)* |
| **Reasoning** | Planner + Temporal Index + Reasoner | Complex analysis | `--reasoning` |
| **Reflective** | Planner + Temporal Index + Reflection | Quality-critical | `--reflection` |
| **Multi-Step** | Planner (Advanced) + Temporal Index | Comparisons, decomposition | `--multi-step` |
| **Full** | All agents | Maximum capability | `--all-features` |

---

## Citation

```bibtex
@software{sonic_o1_multi_agent,
  author = {Radwan, Ahmed Y.},
  title = {Sonic O1: A Multi-Agent System for Audio-Video Understanding},
  year = {2026},
  publisher = {Vector Institute},
  note = {Compound AI system with planning, reasoning, and reflection agents}
}
```

---

## Acknowledgments

Resources used in preparing this research were provided, in part, by the Province
of Ontario, the Government of Canada through CIFAR, and companies sponsoring the
Vector Institute.

This research was funded by the European Union's Horizon Europe research and
innovation programme under the AIXPERT project (Grant Agreement No. 101214389).

Built on:

- [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni) -- Multimodal foundation model
- [vLLM](https://github.com/vllm-project/vllm) -- Efficient LLM inference
- [LangGraph](https://github.com/langchain-ai/langgraph) -- Workflow orchestration
- [Vector Institute AI Engineering Template](https://github.com/VectorInstitute/aieng-template-uv)