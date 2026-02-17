"""Main Sonic O1 Agent with LangGraph integration.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

import logging
from typing import Any, Dict, Iterator, Optional

from sonic_o1_agent.workflows.graph import build_sonic_workflow

logger = logging.getLogger(__name__)


class SonicAgent:
    """Sonic O1 Agent with LangGraph workflow orchestration."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize Sonic Agent with LangGraph.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Build LangGraph workflow
        logger.info("Building LangGraph workflow...")
        workflow = build_sonic_workflow(config)
        self.app = workflow.compile()

        logger.info("Sonic O1 Agent initialized with LangGraph")

    def process(
        self,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        query: str = "",
        max_frames: Optional[int] = None,
        max_audio_chunks: Optional[int] = None,
        use_reasoning: bool = False,
        use_reflection: bool = False,
        use_multi_step: bool = False,
    ) -> Dict[str, Any]:
        """Process video/audio with user query using LangGraph.

        Args:
            video_path: Path to video file.
            audio_path: Path to audio file.
            query: User's query.
            max_frames: Override max frames.
            max_audio_chunks: Override max audio chunks.
            use_reasoning: Enable Chain-of-Thought.
            use_reflection: Enable self-reflection.
            use_multi_step: Enable multi-step planning.

        Returns:
            Dict with keys: response, plan, reasoning_mode, modalities_used,
            and optionally reasoning_chain, reflection, steps_executed, etc.

        Raises:
            ValueError: If query is empty or both video_path and audio_path
                are None.
        """
        if not query:
            raise ValueError("Query cannot be empty")

        if video_path is None and audio_path is None:
            raise ValueError("At least one of video_path or audio_path required")

        # Prepare initial state
        initial_state: Dict[str, Any] = {
            "query": query,
            "video_path": video_path,
            "audio_path": audio_path,
            "max_frames": max_frames,
            "max_audio_chunks": max_audio_chunks,
            "use_reasoning": use_reasoning,
            "use_reflection": use_reflection,
            "use_multi_step": use_multi_step,
            "temp_files": [],
        }

        # Run workflow
        logger.info("Running LangGraph workflow...")
        final_state = self.app.invoke(initial_state)

        # Extract result
        result = {
            "response": final_state["response"],
            "plan": final_state["plan"],
            "reasoning_mode": final_state["reasoning_mode"],
            "modalities_used": {
                "video": video_path is not None,
                "audio": audio_path is not None,
            },
        }

        # Add optional fields
        if "reasoning_chain" in final_state:
            result["reasoning_chain"] = final_state["reasoning_chain"]
            result["confidence"] = final_state["reasoning_confidence"]
        if "reasoning_trace" in final_state:
            result["reasoning_trace"] = final_state["reasoning_trace"]

        if "steps_executed" in final_state:
            result["steps_executed"] = final_state["steps_executed"]
        if "multi_step_plan" in final_state and final_state["multi_step_plan"]:
            result["multi_step_plan"] = final_state["multi_step_plan"]
        if "evidence" in final_state:
            result["evidence"] = final_state["evidence"]

        if "reflection" in final_state:
            result["reflection"] = final_state["reflection"]
            result["was_refined"] = final_state.get("was_refined", False)
            if final_state.get("original_response"):
                result["original_response"] = final_state["original_response"]
        if "refinement_history" in final_state and final_state["refinement_history"]:
            result["refinement_history"] = final_state["refinement_history"]
        if "hallucination_assessment" in final_state:
            result["hallucination_assessment"] = final_state["hallucination_assessment"]

        logger.info("LangGraph workflow complete")
        return result

    def process_stream(
        self,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        query: str = "",
        max_frames: Optional[int] = None,
        max_audio_chunks: Optional[int] = None,
        use_reasoning: bool = False,
        use_reflection: bool = False,
        use_multi_step: bool = False,
    ) -> Iterator[Dict[str, Any]]:
        """Run the workflow and stream progress events (node name + state update).

        Yields dicts with keys "node" (str) and "state" (partial state update).
        After the last event, accumulate state to build the same result as
        process(), or consume state["response"] as it appears after inference.

        Args:
            video_path: Path to video file.
            audio_path: Path to audio file.
            query: User's query.
            max_frames: Override max frames.
            max_audio_chunks: Override max audio chunks.
            use_reasoning: Enable Chain-of-Thought.
            use_reflection: Enable self-reflection.
            use_multi_step: Enable multi-step planning.

        Yields:
            Dict with "node" (name of the node that just ran) and "state"
            (that node's partial state update).

        Raises:
            ValueError: If query is empty or both video_path and audio_path
                are None.
        """
        if not query:
            raise ValueError("Query cannot be empty")
        if video_path is None and audio_path is None:
            raise ValueError("At least one of video_path or audio_path required")
        initial_state: Dict[str, Any] = {
            "query": query,
            "video_path": video_path,
            "audio_path": audio_path,
            "max_frames": max_frames,
            "max_audio_chunks": max_audio_chunks,
            "use_reasoning": use_reasoning,
            "use_reflection": use_reflection,
            "use_multi_step": use_multi_step,
            "temp_files": [],
        }
        for event in self.app.stream(
            initial_state,
            stream_mode="updates",
        ):
            for node_name, state_update in event.items():
                yield {"node": node_name, "state": state_update}

    def get_model_info(self) -> Dict[str, Any]:
        """Return backend and workflow info.

        Returns:
            Dict with keys "backend" and "workflow".
        """
        return {
            "backend": "LangGraph + vLLM",
            "workflow": "enabled",
        }

    def __enter__(self) -> "SonicAgent":
        """Enter context manager. Returns self."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager. No-op."""
        pass
