# SONIC-O1 Multi-Agent System

**Author:** Ahmed Y. Radwan, SONIC-O1 Team

**A compound multi-agent system** for audio-video understanding with Qwen3-Omni. Features intelligent planning, chain-of-thought reasoning, self-reflection, and multi-step task decomposition.

---

## 🧠 Multi-Agent Architecture

SONIC-O1 is **not a single model** — it's a **coordinated system of specialized agents** working together:

```
┌─────────────────────────────────────────────────────────────┐
│                     SONIC-O1 ORCHESTRATOR                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│   │   Planner    │  │   Reasoner   │  │  Reflection  │      │
│   │   Agent      │  │    Agent     │  │    Agent     │      │
│   ├──────────────┤  ├──────────────┤  ├──────────────┤      │
│   │ • Parse time │  │ • Chain of   │  │ • Evaluate   │      │
│   │ • Detect     │  │   Thought    │  │ • Refine     │      │
│   │   modality   │  │ • Multi-step │  │ • Verify     │      │ 
│   │ • Segment    │  │   reasoning  │  │   quality    │      │
│   └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                             │
│   ┌──────────────────────────────────────────────────────┐  │
│   │        Qwen3-Omni Model (vLLM Backend)               │  │
│   │        Native Video + Audio Processing               │  │
│   └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### **Agent Roles:**

1. **Planner Agent** - Task decomposition, temporal parsing, segmentation
2. **Reasoner Agent** - Chain-of-Thought, step-by-step analysis
3. **Reflection Agent** - Self-critique, confidence scoring, refinement
4. **Multimodal Engine** - Video/audio processing with PyAV

---

## 🚀 Quick Start

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

---

## 📖 Usage Examples

### **1. Basic Inference (Single Agent)**
```bash
python scripts/run_agent.py \
  --video courtroom.mp4 --audio courtroom.m4a \
  --query "Summarize the key arguments"
```

### **2. Chain-of-Thought Reasoning (Reasoner Agent)**
```bash
python scripts/run_agent.py \
  --video hearing.mp4 \
  --query "Analyze the legal strategy used" \
  --reasoning
```

### **3. Self-Reflection (Reflection Agent)**
```bash
python scripts/run_agent.py \
  --video deposition.mp4 \
  --query "What inconsistencies exist in the testimony?" \
  --reflection
```

### **4. Multi-Step Decomposition (Planner Agent)**
```bash
python scripts/run_agent.py \
  --video trial.mp4 \
  --query "Compare the defense attorney's arguments vs the prosecutor's" \
  --multi-step
```

### **5. Full Multi-Agent System (All Enabled)**
```bash
python scripts/run_agent.py \
  --video complex_case.mp4 \
  --query "Provide a comprehensive analysis with key contradictions" \
  --all-features \
  --verbose \
  --output results.json
```

### **6. Temporal Queries (Auto-Segmentation)**
```bash
python scripts/run_agent.py \
  --video hearing.mp4 \
  --query "What happened between minute 5 and 10?"
```
*Agent automatically segments and processes only the relevant 5-minute window*

---

## 🎯 How It Works

### **Multi-Agent Workflow (LangGraph):**

```
User Query
    ↓
┌───────────────────────────────────────┐
│  1. Planning                          │
│  • Parse temporal references          │
│  • Detect modalities & query type     │
│  • If --multi-step: decompose into    │
│    sub-tasks (comparison, causal, …)  │
└───────────────────────────────────────┘
                    ↓
┌───────────────────────────────────────┐
│  2. Segmentation                      │
│  • Extract time window if query       │
│    references e.g. "minute 5 to 10"   │
└───────────────────────────────────────┘
                    ↓
┌───────────────────────────────────────┐
│  3. Temporal Index (optional)         │
│  • Split video into N segments        │
│  • Caption each via VLM (time-sliced) │
│  • Skipped for short videos           │
└───────────────────────────────────────┘
                    ↓
┌───────────────┬───────────────┬───────────────┐
        ↓               ↓               ↓
    multi_step      reasoning        direct
    (if planned)    (--reasoning)    (default)
└───────────────┴───────────────┴───────────────┘
                    ↓
┌───────────────────────────────────────┐
│  4. Reflection (if --reflection)      │
│  • Evaluate quality & confidence      │
│  • One-shot refine or iterative       │
│    (config: use_iterative_refinement) │
│  • Optional hallucination check       │
│    (config: check_hallucination)      │
└───────────────────────────────────────┘
                    ↓
         Cleanup (temp files) → Final Answer
```

---

## 🏗️ System Architecture

```
sonic-o1-agent/
├── src/sonic_o1_agent/
│   ├── agents/                    # Multi-Agent System
│   │   ├── sonic_agent.py         # Main orchestrator
│   │   ├── planner.py             # Planning agent
│   │   ├── planner_advanced.py    # Multi-step decomposition
│   │   ├── reasoner.py            # Chain-of-Thought agent
│   │   └── reflection.py          # Self-reflection agent
│   ├── models/                    # Model wrappers
│   │   └── qwen_model.py          # Qwen3-Omni + vLLM
│   ├── core/                      # Multimodal processing
│   │   ├── multimodal_engine.py   # Orchestration (backward compatible)
│   │   ├── video_processor.py     # Video processing (PyAV)
│   │   ├── audio_processor.py    # Audio processing (PyAV)
│   │   └── multimodal_utils.py   # Shared utilities & constants
│   ├── processors/                # Prompt engineering & indexing
│   │   ├── prompt_builder.py      # Dynamic prompt construction
│   │   └── temporal_index.py      # Frame-captioned segment grounding
│   ├── workflows/                 # LangGraph orchestration
│   │   ├── state.py               # SonicState (workflow state schema)
│   │   ├── nodes.py               # All workflow node functions
│   │   └── graph.py               # StateGraph with conditional edges
│   └── utils/                     # Utilities
│       └── segmenter.py           # Video/audio segmentation (ffmpeg)
├── configs/
│   └── agent_config.yaml          # System configuration
├── scripts/
│   ├── run_agent.py               # CLI entry point
│   └── verify_setup.py            # Setup verification
├── slurm/
│   └── run_sonic_agent_native.sh  # SLURM job script
└── tests/                         # Unit & integration tests
```

---

## ⚙️ Configuration

Edit `configs/agent_config.yaml`:

```yaml
# Model settings
model:
  model_path: "Qwen/Qwen3-Omni-30B-A3B-Instruct"
  tensor_parallel_size: 4  # GPUs
  max_frames: 128

# Processing settings
processing:
  max_audio_chunks: null
  min_video_duration_for_segmentation: 300  # 5 min

# Temporal index (frame-captioned segment grounding)
temporal_index:
  min_duration_sec: 180    # Skip indexing for videos shorter than this
  num_segments: 10         # Segments to split the video into
  max_frames_per_segment: 16   # Frames per segment caption
  caption_max_tokens: 256  # Max tokens per segment caption

# Agent settings
reasoning:
  max_reasoning_steps: 5
  enable_verification: true

reflection:
  confidence_threshold: 0.7
  max_refinement_attempts: 2
  use_iterative_refinement: false  # true = loop until ACCEPT or threshold
  check_hallucination: false        # true = run hallucination check after response

planning:
  max_steps: 10   # Max sub-tasks when multi-step decomposes (e.g. compare/contrast)
```

---

## ✨ Key Features

### **🎭 Multi-Agent Coordination**
- Specialized agents for different aspects of understanding
- Agents can work independently or collaboratively
- Dynamic task routing based on query complexity

### **🧠 Chain-of-Thought Reasoning**
- Step-by-step analysis with explicit reasoning traces
- Self-verification at each step
- Confidence scoring

### **🔍 Self-Reflection**
- Automatic quality assessment and one-shot refine when recommended
- **Iterative refinement** (config `use_iterative_refinement: true`): loop until ACCEPT or confidence threshold
- **Hallucination check** (config `check_hallucination: true`): optional post-response verification

### **📋 Multi-Step Planning**
- When `--multi-step` is used and the query is complex (compare, contrast, both, why, …), the **advanced planner** decomposes it into sub-tasks
- Sequential execution with context passing between steps
- Handles comparisons, causality, multi-entity analysis; result includes `steps_executed` and `multi_step_plan`

### **🕐 Temporal Grounding**
- For longer videos (configurable threshold), a **frame-captioning index** is built before inference: the video is split into N segments, each captioned by the VLM using time-sliced video frames **and** audio, producing a timestamped text index
- The index is injected into the main prompt so the model can cite accurate timestamps in seconds (e.g., "Around 45s, ...")
- Audio and video are sliced per-segment (`audio_start`/`audio_end`, `video_start`/`video_end`) so only the relevant chunk is loaded -- not the full file each time
- Short videos skip indexing entirely and rely on sampling-context hints
- Compact evidence metadata: coverage range, sampling interval, frame/chunk counts

### **⚡ Efficient Processing**
- Prompt caching (90% cost reduction for follow-ups)
- Smart video segmentation (only process relevant parts)
- Memory-optimized with vLLM

---

## 🔬 Agent Modes

| Mode | Agents Active | Use Case | Command |
|------|--------------|----------|---------|
| **Direct** | Planner + Temporal Index | Simple queries | Default |
| **Reasoning** | Planner + Temporal Index + Reasoner | Complex analysis | `--reasoning` |
| **Reflective** | Planner + Temporal Index + Reflection | Quality-critical | `--reflection` |
| **Multi-Step** | Planner (Advanced) + Temporal Index + step-wise inference | Comparisons, decomposition | `--multi-step` |
| **Full** | All agents | Maximum capability | `--all-features` |

---

## 📊 Example Output

The agent returns a dict with `response`, `plan`, `reasoning_mode`, and optionally: `reasoning_chain`, `reasoning_trace`, `steps_executed`, `multi_step_plan`, `reflection`, `refinement_history`, `hallucination_assessment`, `evidence`.

```
======================================================================
SONIC-O1 AGENT RESPONSE
======================================================================

Query: Compare the defense attorney's arguments vs the prosecutor's

Mode: multi_step

Multi-Step Plan (5 steps):
  1. Identify first entity/person/topic
  2. Analyze first entity
  3. Identify second entity/person/topic
  4. Analyze second entity
  5. Compare and contrast both entities

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
  ✓ Response was refined

Response:
  Around 30s, the defense attorney opens by...
  Between 120s and 180s, the prosecutor counters with...
  [Detailed comparative analysis with timestamped evidence...]

======================================================================
```

---

## 🎓 Research Applications

SONIC-O1's multi-agent architecture is designed for:

- **Legal Analysis** - Courtroom proceedings, depositions, testimonies
- **Medical Review** - Clinical consultations, surgical recordings
- **Education** - Lecture analysis, student presentations
- **Compliance** - Meeting analysis, policy adherence
- **Research** - Interview transcription, qualitative analysis

---

## 📚 Citation

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

## 📧 Contact

**Ahmed Y. Radwan**  
Vector Institute for Artificial Intelligence  
ahmed.radwan@vectorinstitute.ai

---

## 📄 License

Apache-2.0 - See [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

Resources used in preparing this research were provided, in part, by the Province of Ontario, the Government of Canada through CIFAR, and companies sponsoring the Vector Institute.

This research was funded by the European Union's Horizon Europe research and innovation programme under the AIXPERT project (Grant Agreement No. 101214389).

Built on:
- [Qwen3-Omni](https://github.com/QwenLM/Qwen3-Omni) - Multimodal foundation model
- [vLLM](https://github.com/vllm-project/vllm) - Efficient LLM inference
- [Vector Institute AI Engineering Template](https://github.com/VectorInstitute/aieng-template-uv)
