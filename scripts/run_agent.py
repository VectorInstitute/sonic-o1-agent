#!/usr/bin/env python3
"""Entry point for Sonic O1 Agent.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sonic_o1_agent import SonicAgent

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

# Suppress verbose vLLM logging
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("vllm.engine").setLevel(logging.WARNING)
logging.getLogger("vllm.worker").setLevel(logging.WARNING)
logging.getLogger("vllm.multiproc_executor").setLevel(logging.WARNING)
logging.getLogger("vllm.parallel_state").setLevel(logging.WARNING)
logging.getLogger("vllm.config").setLevel(logging.WARNING)
logging.getLogger("vllm.engine.arg_utils").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return dict(yaml.safe_load(f))


def _state_to_result(final_state: dict, video_path: str, audio_path: str) -> dict:
    """Build the same result dict as SonicAgent.process() from merged state."""
    result = {
        "response": final_state["response"],
        "plan": final_state["plan"],
        "reasoning_mode": final_state["reasoning_mode"],
        "modalities_used": {
            "video": video_path is not None,
            "audio": audio_path is not None,
        },
    }
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
    return result


def _run_with_stream(
    agent,
    video_path: str,
    audio_path: str,
    query: str,
    max_frames: int,
    max_audio_chunks: int,
    use_reasoning: bool,
    use_reflection: bool,
    use_multi_step: bool,
) -> dict:
    """Run workflow with process_stream(); print progress and return result."""
    initial_state = {
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
    merged = dict(initial_state)
    logger.info("Workflow progress (streaming):")
    for event in agent.process_stream(
        video_path=video_path,
        audio_path=audio_path,
        query=query,
        max_frames=max_frames,
        max_audio_chunks=max_audio_chunks,
        use_reasoning=use_reasoning,
        use_reflection=use_reflection,
        use_multi_step=use_multi_step,
    ):
        node = event["node"]
        state_update = event.get("state")
        if state_update is not None and isinstance(state_update, dict):
            merged = {**merged, **state_update}
        # Progress line visible in sbatch log
        print(f"  → {node}", flush=True)
        logger.info("  → %s", node)
    return _state_to_result(merged, video_path, audio_path)


def main():
    """Run the Sonic O1 Agent."""
    parser = argparse.ArgumentParser(
        description="Sonic O1 Agent - Multimodal Video/Audio Processing"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/agent_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to audio file",
    )
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="Query/question to ask",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process",
    )
    parser.add_argument(
        "--max-audio-chunks",
        type=int,
        default=None,
        help="Maximum audio chunks to process",
    )

    # Advanced features
    parser.add_argument(
        "--reasoning",
        action="store_true",
        help="Enable Chain-of-Thought reasoning",
    )
    parser.add_argument(
        "--reflection",
        action="store_true",
        help="Enable self-reflection and refinement",
    )
    parser.add_argument(
        "--multi-step",
        action="store_true",
        help="Enable multi-step task decomposition",
    )
    parser.add_argument(
        "--all-features",
        action="store_true",
        help="Enable all advanced features (reasoning + reflection + multi-step)",
    )

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed reasoning trace",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream workflow progress (print each node as it runs); "
        "shows in job log when using sbatch",
    )

    args = parser.parse_args()

    # Validate inputs
    if args.video is None and args.audio is None:
        parser.error("At least one of --video or --audio is required")

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)

    # Handle --all-features flag
    use_reasoning = args.reasoning or args.all_features
    use_reflection = args.reflection or args.all_features
    use_multi_step = args.multi_step or args.all_features

    # Initialize agent
    logger.info("Initializing Sonic O1 Agent...")
    agent = SonicAgent(config)

    try:
        # Process query
        logger.info(f"Processing query: {args.query}")

        if use_reasoning:
            logger.info("→ Chain-of-Thought reasoning: ENABLED")
        if use_reflection:
            logger.info("→ Self-reflection: ENABLED")
        if use_multi_step:
            logger.info("→ Multi-step planning: ENABLED")

        if args.stream:
            result = _run_with_stream(
                agent=agent,
                video_path=args.video,
                audio_path=args.audio,
                query=args.query,
                max_frames=args.max_frames,
                max_audio_chunks=args.max_audio_chunks,
                use_reasoning=use_reasoning,
                use_reflection=use_reflection,
                use_multi_step=use_multi_step,
            )
        else:
            result = agent.process(
                video_path=args.video,
                audio_path=args.audio,
                query=args.query,
                max_frames=args.max_frames,
                max_audio_chunks=args.max_audio_chunks,
                use_reasoning=use_reasoning,
                use_reflection=use_reflection,
                use_multi_step=use_multi_step,
            )

        # Display results
        print("\n" + "=" * 70)
        print("SONIC O1 AGENT RESPONSE")
        print("=" * 70)
        print(f"\nQuery: {args.query}")
        print(f"\nMode: {result.get('reasoning_mode', 'direct')}")

        # Show plan
        if args.verbose:
            print(f"\nPlan: {json.dumps(result['plan'], indent=2)}")

        # Show multi-step execution
        if "multi_step_plan" in result and args.verbose:
            print(f"\nMulti-Step Plan ({len(result['multi_step_plan'])} steps):")
            for step in result["multi_step_plan"]:
                print(f"  {step['step_id']}. {step['description']}")

        # Show reasoning chain
        if "reasoning_chain" in result and args.verbose:
            print("\nReasoning Chain:")
            for step in result["reasoning_chain"]:
                print(f"  Step {step['step']}: {step['action']}")
                print(f"    → {step['thought'][:100]}...")

        if "reflection" in result:
            r = result["reflection"]
            confidence = r.get("final_confidence") or r.get("confidence")
            lines = ["\nReflection:"]
            if confidence is not None:
                lines.append(f"  Confidence: {confidence:.2f}")
            if args.verbose:
                if r.get("scores") is not None:
                    lines.append(f"  Scores: {r['scores']}")
                if result.get("was_refined"):
                    lines.append("  ✓ Response was refined")
                if r.get("total_attempts") is not None:
                    lines.append(f"  Total refinement attempts: {r['total_attempts']}")
            print("\n".join(lines))

        # Show final response
        print(f"\nResponse:\n{result['response']}")
        print("\n" + "=" * 70)

        # Save to file if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

            logger.info(f"Results saved to {args.output}")

    finally:
        logger.info("Agent shutdown complete")


if __name__ == "__main__":
    main()
