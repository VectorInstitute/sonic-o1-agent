"""LangGraph nodes for Sonic O1 workflow.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

import logging
import tempfile
from pathlib import Path
from typing import Dict, Optional, cast

from sonic_o1_agent.agents.planner import AgentPlanner
from sonic_o1_agent.agents.planner_advanced import MultiStepPlanner
from sonic_o1_agent.agents.reasoner import ChainOfThoughtReasoner
from sonic_o1_agent.agents.reflection import SelfReflection
from sonic_o1_agent.models.qwen_model import Qwen3OmniModel
from sonic_o1_agent.processors.prompt_builder import PromptBuilder
from sonic_o1_agent.processors.temporal_index import TemporalIndexBuilder
from sonic_o1_agent.utils.segmenter import VideoSegmenter
from sonic_o1_agent.workflows.state import SonicState

logger = logging.getLogger(__name__)


class SonicNodes:
    """Container for LangGraph workflow node functions and shared components."""

    def __init__(self, config: Dict) -> None:
        """Initialize node container and lightweight components.

        Args:
            config: Agent configuration dict. Model, reasoner, and reflection
                are created lazily on first use (load_model).
        """
        self.config = config

        # Initialize components (lazy load model)
        self.model: Optional[Qwen3OmniModel] = None
        self.planner = AgentPlanner()
        self.multi_step_planner = MultiStepPlanner(config.get("planning", {}))
        self.prompt_builder = PromptBuilder()
        self.segmenter = VideoSegmenter()
        self.temporal_index_builder = TemporalIndexBuilder(
            config.get("temporal_index", {})
        )
        self.reasoner: Optional[ChainOfThoughtReasoner] = None
        self.reflection: Optional[SelfReflection] = None

    def load_model(self) -> None:
        """Load model, reasoner, and reflection if not already loaded."""
        if self.model is None:
            logger.info("Loading model...")
            self.model = Qwen3OmniModel(self.config.get("model", {}))
            self.model.load()

            self.reasoner = ChainOfThoughtReasoner(
                self.model, self.config.get("reasoning", {})
            )
            self.reflection = SelfReflection(
                self.model, self.config.get("reflection", {})
            )

    def planning_node(self, state: SonicState) -> SonicState:
        """Plan processing: modalities, segmentation, query type, multi-step.

        Args:
            state: Current state with query, video_path, audio_path, flags.

        Returns:
            Partial state: plan, should_segment, time_range, query_type,
            multi_step_plan (if use_multi_step and query decomposable).
        """
        logger.info("Node: Planning")

        _vp = state.get("video_path")
        _ap = state.get("audio_path")
        video_path = Path(_vp) if _vp else None
        audio_path = Path(_ap) if _ap else None

        plan = self.planner.plan_processing(
            video_path,
            audio_path,
            state["query"],
            config=self.config.get("processing", {}),
        )

        # Multi-step: decompose query when requested and applicable
        multi_step_plan = None
        if state.get("use_multi_step") and self.multi_step_planner.should_decompose(
            state["query"]
        ):
            multi_step_plan = self.multi_step_planner.decompose_query(state["query"])
            logger.info("Multi-step plan: %d steps", len(multi_step_plan))

        return {
            "plan": plan,
            "should_segment": plan["should_segment"],
            "time_range": plan.get("time_range"),
            "query_type": plan["query_type"],
            "multi_step_plan": multi_step_plan,
        }

    def segmentation_node(self, state: SonicState) -> SonicState:
        """Segment video/audio to time range when planning requested it.

        Args:
            state: Must contain should_segment, time_range, video_path,
                audio_path.

        Returns:
            Partial state: actual_video_path, actual_audio_path, temp_files.
        """
        logger.info("Node: Segmentation")

        if not state["should_segment"] or not state.get("time_range"):
            # No segmentation needed
            return {
                "actual_video_path": state.get("video_path"),
                "actual_audio_path": state.get("audio_path"),
                "temp_files": [],
            }

        # Segment
        _vp2 = state.get("video_path")
        _ap2 = state.get("audio_path")
        video_path = Path(_vp2) if _vp2 else None
        audio_path = Path(_ap2) if _ap2 else None
        time_range_seg = state["time_range"]
        assert time_range_seg is not None
        start_time, end_time = time_range_seg

        temp_files = []
        temp_video = None
        temp_audio = None

        if video_path and video_path.exists():
            temp_video = Path(tempfile.mktemp(suffix=".mp4", prefix="sonic_seg_"))
            self.segmenter.extract_video_segment(
                video_path, start_time, end_time, temp_video
            )
            temp_files.append(str(temp_video))

        if audio_path and audio_path.exists():
            audio_format = audio_path.suffix[1:]
            temp_audio = Path(
                tempfile.mktemp(suffix=f".{audio_format}", prefix="sonic_seg_")
            )
            self.segmenter.extract_audio_segment(
                audio_path, start_time, end_time, temp_audio, audio_format
            )
            temp_files.append(str(temp_audio))

        return {
            "actual_video_path": str(temp_video)
            if temp_video
            else state.get("video_path"),
            "actual_audio_path": str(temp_audio)
            if temp_audio
            else state.get("audio_path"),
            "temp_files": temp_files,
        }

    def temporal_indexing_node(self, state: SonicState) -> SonicState:
        """Build a timestamped caption index of the video for grounding.

        Splits the video into segments, generates a short VLM caption per
        segment, and assembles a text index injected into downstream prompts.

        Args:
            state: Must contain plan (duration_seconds), actual_video_path.

        Returns:
            Partial state: temporal_index (str or None).
        """
        logger.info("Node: Temporal Indexing")

        duration = state["plan"].get("duration_seconds")
        video_path = state.get("actual_video_path")

        if not video_path or not duration:
            return {"temporal_index": None}

        self.load_model()
        assert self.model is not None

        index_text = self.temporal_index_builder.build_index(
            model=self.model,
            video_path=video_path,
            audio_path=state.get("actual_audio_path"),
            duration_sec=duration,
        )

        return {"temporal_index": index_text or None}

    def reasoning_node(self, state: SonicState) -> SonicState:
        """Run Chain-of-Thought reasoning over video/audio and query.

        Args:
            state: Query, paths, plan, temporal_index, time_range, etc.

        Returns:
            Partial state: response, reasoning_chain, reasoning_confidence,
            reasoning_mode, reasoning_trace.
        """
        logger.info("Node: Reasoning")

        self.load_model()
        assert self.reasoner is not None

        context = {
            "max_frames": state.get("max_frames") or state["plan"]["max_frames"],
            "max_audio_chunks": state.get("max_audio_chunks")
            or state["plan"].get("max_audio_chunks"),
            "query_type": state["query_type"],
            "duration_seconds": state["plan"].get("duration_seconds"),
            "temporal_index": state.get("temporal_index"),
        }

        time_range_r = state.get("time_range")
        if time_range_r:
            context["segment_start"] = time_range_r[0]
            context["segment_end"] = time_range_r[1]

        reasoning_result = self.reasoner.reason(
            query=state["query"],
            video_path=state.get("actual_video_path"),
            audio_path=state.get("actual_audio_path"),
            context=context,
        )

        reasoning_trace = self.reasoner.get_reasoning_trace(reasoning_result)

        return {
            "response": reasoning_result["final_answer"],
            "reasoning_chain": reasoning_result["reasoning_chain"],
            "reasoning_confidence": reasoning_result["confidence"],
            "reasoning_mode": "chain_of_thought",
            "reasoning_trace": reasoning_trace,
        }

    def multi_step_node(self, state: SonicState) -> SonicState:
        """Execute multi-step plan via executor_fn (direct inference per step).

        Args:
            state: Must contain multi_step_plan, actual paths, plan, etc.

        Returns:
            Partial state: response, reasoning_mode "multi_step",
            steps_executed. Falls back to direct_inference_node if no plan.
        """
        logger.info("Node: Multi-Step Execution")

        self.load_model()
        assert self.model is not None
        model = self.model

        multi_step_plan = state["multi_step_plan"]
        if not multi_step_plan:
            # Fallback: single direct inference
            return self.direct_inference_node(state)

        max_frames = state.get("max_frames") or state["plan"]["max_frames"]
        max_audio_chunks = state.get("max_audio_chunks") or state["plan"].get(
            "max_audio_chunks"
        )
        video_path = state.get("actual_video_path")
        audio_path = state.get("actual_audio_path")
        base_context = {
            "query_type": state["query_type"],
            "max_frames": max_frames,
            "duration_seconds": state["plan"].get("duration_seconds"),
            "temporal_index": state.get("temporal_index"),
        }
        time_range_ms = state.get("time_range")
        if time_range_ms:
            base_context["segment_start"] = time_range_ms[0]
            base_context["segment_end"] = time_range_ms[1]

        def executor_fn(
            query: str,
            video_path_arg=None,
            audio_path_arg=None,
            context=None,
        ):
            context = context or {}
            ctx = {**base_context}
            if context:
                ctx["previous_steps"] = "\n".join(
                    f"Step {k}: {v}"
                    for k, v in sorted(context.items())
                    if isinstance(v, str)
                )
            prompt = self.prompt_builder.build_prompt(
                query,
                context=ctx,
                query_type=state["query_type"],
            )
            if ctx.get("previous_steps"):
                prompt = (
                    "Context from previous steps:\n"
                    f"{ctx['previous_steps']}\n\n"
                    f"Current sub-task: {prompt}"
                )
            response_text, _ = model.generate(
                video_path=video_path_arg or video_path,
                audio_path=audio_path_arg or audio_path,
                prompt=prompt,
                max_frames=max_frames,
                max_audio_chunks=max_audio_chunks,
            )
            return response_text

        result = self.multi_step_planner.execute_plan(
            multi_step_plan,
            executor_fn,
            video_path=video_path,
            audio_path=audio_path,
        )

        return {
            "response": result["final_answer"],
            "reasoning_mode": "multi_step",
            "steps_executed": result["steps_executed"],
        }

    def direct_inference_node(self, state: SonicState) -> SonicState:
        """Run single-pass inference with temporal grounding and evidence.

        Args:
            state: Query, actual paths, plan, temporal_index, time_range.

        Returns:
            Partial state: response, reasoning_mode "direct", evidence.
        """
        logger.info("Node: Direct Inference")

        self.load_model()
        assert self.model is not None

        max_frames = state.get("max_frames") or state["plan"]["max_frames"]
        max_audio_chunks = state.get("max_audio_chunks") or state["plan"].get(
            "max_audio_chunks"
        )

        # Build prompt with temporal grounding context
        context = {
            "query_type": state["query_type"],
            "max_frames": max_frames,
            "duration_seconds": state["plan"].get("duration_seconds"),
            "temporal_index": state.get("temporal_index"),
        }
        time_range_di = state.get("time_range")
        if time_range_di:
            context["segment_start"] = time_range_di[0]
            context["segment_end"] = time_range_di[1]

        prompt = self.prompt_builder.build_prompt(
            state["query"],
            context=context,
            query_type=state["query_type"],
        )

        # Generate
        response, metadata = self.model.generate(
            video_path=state.get("actual_video_path"),
            audio_path=state.get("actual_audio_path"),
            prompt=prompt,
            max_frames=max_frames,
            max_audio_chunks=max_audio_chunks,
        )

        # Compact evidence summary
        evidence = self._format_evidence(metadata)

        return {
            "response": response,
            "reasoning_mode": "direct",
            "evidence": evidence,
        }

    def _format_evidence(self, metadata: dict) -> dict:
        """Format model metadata into a compact evidence summary.

        Returns only aggregate stats (duration, counts, coverage) in
        seconds; no per-frame or per-chunk indices.

        Args:
            metadata: Dict with optional video_metadata and audio_metadata
                lists (from model.generate).

        Returns:
            Dict with optional "video" and "audio" keys and aggregate stats.
        """
        evidence = {}

        if metadata.get("video_metadata"):
            vm = metadata["video_metadata"][0]
            evidence["video"] = {
                "duration_sec": vm["duration_sec"],
                "frames_analyzed": vm["frames_sampled"],
                "sampling_interval_sec": vm["sampling_interval_sec"],
                "coverage_sec": vm["coverage_sec"],
            }

        if metadata.get("audio_metadata"):
            am = metadata["audio_metadata"][0]
            evidence["audio"] = {
                "duration_sec": am["duration_sec"],
                "chunks_analyzed": am["chunks_analyzed"],
                "chunk_duration_sec": am["chunk_duration_sec"],
                "coverage_sec": am["coverage_sec"],
            }

        return evidence

    def reflection_node(self, state: SonicState) -> SonicState:
        """Evaluate and optionally refine response; optional hallucination check.

        Uses single evaluate+refine or iterative_refinement and optionally
        detect_hallucination based on config (reflection section).

        Args:
            state: Must contain query, response (from inference node).

        Returns:
            Partial state: response (possibly refined), reflection,
            was_refined, refinement_history, hallucination_assessment.
        """
        logger.info("Node: Reflection")

        self.load_model()
        assert self.reflection is not None

        ref_config = self.config.get("reflection", {})
        use_iterative = ref_config.get("use_iterative_refinement", False)
        check_hallucination = ref_config.get("check_hallucination", False)

        video_path = state.get("actual_video_path")
        audio_path = state.get("actual_audio_path")
        max_frames = state.get("max_frames") or state["plan"]["max_frames"]

        if use_iterative:
            # Iterative refinement until ACCEPT, REJECT, or confidence threshold
            iter_result = self.reflection.iterative_refinement(
                state["query"],
                state["response"],
                video_path=video_path,
                audio_path=audio_path,
                max_frames=max_frames,
            )
            result = {
                "response": iter_result["final_response"],
                "reflection": {
                    "final_confidence": iter_result["final_confidence"],
                    "refinement_history": iter_result["refinement_history"],
                    "total_attempts": iter_result["total_attempts"],
                },
                "refinement_history": iter_result["refinement_history"],
                "was_refined": iter_result["total_attempts"] > 1,
            }
            if iter_result["refinement_history"]:
                result["original_response"] = iter_result["refinement_history"][0].get(
                    "response", state["response"]
                )
        else:
            # Single evaluate + optional one-shot refine
            reflection_result = self.reflection.evaluate_response(
                state["query"], state["response"]
            )
            result = {
                "reflection": {
                    "confidence": reflection_result["confidence"],
                    "scores": reflection_result["scores"],
                    "strengths": reflection_result["strengths"],
                    "weaknesses": reflection_result["weaknesses"],
                },
            }
            if reflection_result["recommendation"] == "REFINE":
                logger.info("Refining response...")
                refined = self.reflection.refine_response(
                    state["query"],
                    state["response"],
                    reflection_result,
                    video_path,
                    audio_path,
                    max_frames,
                )
                result["original_response"] = state["response"]
                result["response"] = refined
                result["was_refined"] = True
            else:
                result["was_refined"] = False

        # Optional hallucination check on final response
        if check_hallucination:
            response_to_check = result.get("response", state["response"])
            halluc = self.reflection.detect_hallucination(
                response_to_check,
                video_path=video_path,
                audio_path=audio_path,
            )
            result["hallucination_assessment"] = halluc

        return cast(SonicState, result)

    def cleanup_node(self, state: SonicState) -> SonicState:
        """Remove temporary files created by segmentation.

        Args:
            state: May contain temp_files list.

        Returns:
            Empty dict (no state update).
        """
        logger.info("Node: Cleanup")

        for temp_file in state.get("temp_files", []):
            temp_path = Path(temp_file)
            if temp_path.exists():
                temp_path.unlink()
                logger.debug("Cleaned up: %s", temp_file)

        return {}
