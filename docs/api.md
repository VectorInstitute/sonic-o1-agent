# API Reference

## Top Level

::: sonic_o1_agent
    options:
      show_root_heading: true
      show_root_full_path: true

---

## Agents

### SonicAgent (Orchestrator)

::: sonic_o1_agent.agents.sonic_agent
    options:
      show_root_heading: true
      show_root_full_path: true

### AgentPlanner

::: sonic_o1_agent.agents.planner
    options:
      show_root_heading: true
      show_root_full_path: true

### ChainOfThoughtReasoner

::: sonic_o1_agent.agents.reasoner
    options:
      show_root_heading: true
      show_root_full_path: true

### SelfReflection

::: sonic_o1_agent.agents.reflection
    options:
      show_root_heading: true
      show_root_full_path: true

### MultiStepPlanner

::: sonic_o1_agent.agents.planner_advanced
    options:
      show_root_heading: true
      show_root_full_path: true

---

## Models

### Qwen3OmniModel

::: sonic_o1_agent.models.qwen_model
    options:
      show_root_heading: true
      show_root_full_path: true

---

## API Server

### Demo Server

::: sonic_o1_agent.api
    options:
      show_root_heading: true
      show_root_full_path: true

---

## Processors

### PromptBuilder

::: sonic_o1_agent.processors.prompt_builder
    options:
      show_root_heading: true
      show_root_full_path: true

### TemporalIndexBuilder

::: sonic_o1_agent.processors.temporal_index
    options:
      show_root_heading: true
      show_root_full_path: true

---

## Core

### Multimodal Engine

::: sonic_o1_agent.core.multimodal_engine
    options:
      show_root_heading: true
      show_root_full_path: true

**Note:** The multimodal engine has been refactored into focused modules:
- `multimodal_engine.py` - Orchestration functions (maintains backward compatibility)
- `video_processor.py` - Video processing functions
- `audio_processor.py` - Audio processing functions
- `multimodal_utils.py` - Shared utilities and constants

All functions remain accessible through `multimodal_engine` for backward compatibility.

---

## Workflows

### Graph

::: sonic_o1_agent.workflows.graph
    options:
      show_root_heading: true
      show_root_full_path: true

### Nodes

::: sonic_o1_agent.workflows.nodes
    options:
      show_root_heading: true
      show_root_full_path: true

### State

::: sonic_o1_agent.workflows.state
    options:
      show_root_heading: true
      show_root_full_path: true

---

## Utilities

### VideoSegmenter

::: sonic_o1_agent.utils.segmenter
    options:
      show_root_heading: true
      show_root_full_path: true
