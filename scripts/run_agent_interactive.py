#!/usr/bin/env python3
"""Interactive CLI for Sonic O1 Agent.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sonic_o1_agent import SonicAgent

# Setup logging
logging.basicConfig(
    level=logging.WARNING,  # Only show WARNING and above by default
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

# Suppress transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Suppress workflow node INFO messages (keep only WARNING/ERROR)
logging.getLogger("sonic_o1_agent.workflows.nodes").setLevel(logging.WARNING)
logging.getLogger("sonic_o1_agent.agents.planner").setLevel(logging.WARNING)
logging.getLogger("sonic_o1_agent.agents.sonic_agent").setLevel(logging.WARNING)

# Keep our own logger at INFO for important messages
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class InteractiveCLI:
    """Interactive command-line interface for Sonic O1 Agent."""

    def __init__(self):
        """Initialize interactive CLI."""
        self.agent = None
        self.config = None

    def clear_screen(self):
        """Clear terminal screen."""
        import os

        os.system("cls" if os.name == "nt" else "clear")

    def print_header(self):
        """Print application header."""
        print("\n" + "=" * 70)
        print("🎬 SONIC O1 MULTI-AGENT SYSTEM - Interactive Mode")
        print("=" * 70)
        print("Author: Ahmed Y. Radwan, SONIC-O1 Team")
        print("=" * 70 + "\n")

    def get_yes_no(self, prompt: str, default: Optional[bool] = None) -> bool:
        """Get yes/no input with validation.

        Args:
            prompt: Prompt to display
            default: Default value if user presses Enter (None = required)

        Returns:
            True for yes, False for no
        """
        while True:
            if default is True:
                prompt_text = f"{prompt} (Y/n): "
            elif default is False:
                prompt_text = f"{prompt} (y/N): "
            else:
                prompt_text = f"{prompt} (y/n): "

            response = input(prompt_text).strip().lower()

            # Handle default
            if not response:
                if default is not None:
                    return default
                else:
                    print("  ❌ Please enter 'y' or 'n'")
                    continue

            # Validate input
            if response in ["y", "yes"]:
                return True
            elif response in ["n", "no"]:
                return False
            else:
                print(f"  ❌ Invalid input '{response}'. Please enter 'y' or 'n'")

    def print_menu(self, title: str, options: list) -> str:
        """Print menu and get user choice.

        Args:
            title: Menu title
            options: List of (key, description) tuples

        Returns:
            Selected option key
        """
        print(f"\n{title}")
        print("-" * len(title))
        for key, description in options:
            print(f"  [{key}] {description}")
        print()

        while True:
            choice = input("Select option: ").strip().lower()
            valid_keys = [k for k, _ in options]
            if choice in valid_keys:
                return choice
            print(f"  ❌ Invalid choice. Please select from: {', '.join(valid_keys)}")

    def get_file_path(self, file_type: str, optional: bool = False) -> Optional[str]:
        """Get file path from user with validation.

        Args:
            file_type: Type of file (video/audio)
            optional: Whether file is optional

        Returns:
            File path or None if skipped
        """
        while True:
            if optional:
                path = input(
                    f"\n{file_type.capitalize()} file path (or press Enter to skip): "
                ).strip()
                if not path:
                    if self.get_yes_no(f"Skip {file_type}?"):
                        print(f"  ✓ Skipped {file_type}")
                        return None
                    else:
                        continue
            else:
                path = input(f"\n{file_type.capitalize()} file path: ").strip()

            if path:
                file_path = Path(path)
                if file_path.exists():
                    print(f"  ✓ Found {file_type}: {file_path.name}")
                    return str(file_path)
                else:
                    print(f"  ❌ File not found: {path}")
                    if not self.get_yes_no("Try again?"):
                        if optional:
                            return None
                        else:
                            print("\n  Exiting...")
                            sys.exit(1)
            elif not optional:
                print("  ❌ File path is required")

    def get_query(self) -> str:
        """Get user query."""
        print("\n" + "=" * 70)
        print("Enter your question about the video/audio:")
        print("=" * 70)
        query = input("\nQuery: ").strip()

        if not query:
            print("❌ Query cannot be empty")
            return self.get_query()

        return query

    def select_features(self) -> dict:
        """Let user select which features to enable.

        Returns:
            Dict with feature flags
        """
        print("\n" + "=" * 70)
        print("🧠 SELECT AGENT FEATURES")
        print("=" * 70)

        features = {}

        # Quick modes
        mode_options = [
            ("1", "Quick Mode (Direct inference - fastest)"),
            ("2", "Smart Mode (with Chain-of-Thought reasoning)"),
            ("3", "Quality Mode (with Self-Reflection)"),
            ("4", "Advanced Mode (with Multi-Step Planning)"),
            ("5", "Full Mode (All features enabled)"),
            ("6", "Custom (Choose individual features)"),
        ]

        mode = self.print_menu("Select Processing Mode:", mode_options)

        if mode == "1":
            features = {
                "reasoning": False,
                "reflection": False,
                "multi_step": False,
            }
        elif mode == "2":
            features = {
                "reasoning": True,
                "reflection": False,
                "multi_step": False,
            }
        elif mode == "3":
            features = {
                "reasoning": False,
                "reflection": True,
                "multi_step": False,
            }
        elif mode == "4":
            features = {
                "reasoning": False,
                "reflection": False,
                "multi_step": True,
            }
        elif mode == "5":
            features = {
                "reasoning": True,
                "reflection": True,
                "multi_step": True,
            }
        elif mode == "6":
            # Custom selection
            print("\nCustom Feature Selection:")

            features["reasoning"] = self.get_yes_no(
                "Enable Chain-of-Thought reasoning?"
            )
            features["reflection"] = self.get_yes_no("Enable Self-Reflection?")
            features["multi_step"] = self.get_yes_no("Enable Multi-Step Planning?")

        # Display selected features
        print("\n✓ Features enabled:")
        if features["reasoning"]:
            print("  • Chain-of-Thought Reasoning")
        if features["reflection"]:
            print("  • Self-Reflection")
        if features["multi_step"]:
            print("  • Multi-Step Planning")
        if not any(features.values()):
            print("  • Direct Inference (baseline)")

        return features

    def load_config(self, config_path: str = "configs/agent_config.yaml"):
        """Load agent configuration."""
        print(f"\n📋 Loading configuration from {config_path}...")
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        print("✓ Configuration loaded")

    def initialize_agent(self):
        """Initialize the agent."""
        print("\n🚀 Initializing Sonic O1 Agent...")
        # Suppress logs during initialization
        workflow_logger = logging.getLogger("sonic_o1_agent.workflows.nodes")
        original_level = workflow_logger.level
        workflow_logger.setLevel(logging.WARNING)

        try:
            self.agent = SonicAgent(self.config)
        finally:
            workflow_logger.setLevel(original_level)

        print("✓ Agent initialized (model will load on first query)")

    def process_query(self, video_path, audio_path, query, features):
        """Process query with the agent.

        Args:
            video_path: Path to video
            audio_path: Path to audio
            query: User query
            features: Feature flags

        Returns:
            Result dict
        """
        print("\n" + "=" * 70)
        print("🎬 PROCESSING...")
        print("=" * 70)
        print("(This may take a few minutes - model is processing your query)")
        print()

        # Temporarily suppress INFO logs during processing
        workflow_logger = logging.getLogger("sonic_o1_agent.workflows.nodes")
        planner_logger = logging.getLogger("sonic_o1_agent.agents.planner")
        agent_logger = logging.getLogger("sonic_o1_agent.agents.sonic_agent")

        original_levels = {
            workflow_logger: workflow_logger.level,
            planner_logger: planner_logger.level,
            agent_logger: agent_logger.level,
        }

        # Set to WARNING during processing
        workflow_logger.setLevel(logging.WARNING)
        planner_logger.setLevel(logging.WARNING)
        agent_logger.setLevel(logging.WARNING)

        try:
            result = self.agent.process(
                video_path=video_path,
                audio_path=audio_path,
                query=query,
                use_reasoning=features["reasoning"],
                use_reflection=features["reflection"],
                use_multi_step=features["multi_step"],
            )
        finally:
            # Restore original log levels
            for logger_obj, level in original_levels.items():
                logger_obj.setLevel(level)

        return result

    def display_results(self, result: dict, verbose: bool = False):
        """Display processing results.

        Args:
            result: Result dict from agent
            verbose: Show detailed information
        """
        print("\n" + "=" * 70)
        print("📊 RESULTS")
        print("=" * 70)

        print(f"\nMode: {result.get('reasoning_mode', 'direct')}")

        # Show reflection if available
        if "reflection" in result:
            print("\n🔍 Reflection:")
            reflection = result["reflection"]
            # Handle both iterative (final_confidence) and single (confidence) modes
            confidence = reflection.get("final_confidence") or reflection.get(
                "confidence"
            )
            if confidence is not None:
                print(f"  Confidence: {confidence:.2%}")
            if verbose and reflection.get("scores"):
                print(f"  Scores: {reflection['scores']}")
            if reflection.get("total_attempts"):
                print(f"  Total refinement attempts: {reflection['total_attempts']}")
            if result.get("was_refined"):
                print("  ✓ Response was refined for better quality")

        # Show multi-step plan if available
        if "multi_step_plan" in result and verbose:
            print(f"\n📋 Multi-Step Plan ({len(result['multi_step_plan'])} steps):")
            for step in result["multi_step_plan"]:
                print(f"  {step['step_id']}. {step['description']}")

        # Show reasoning chain if available
        if "reasoning_chain" in result and verbose:
            print("\n🧠 Reasoning Chain:")
            for step in result["reasoning_chain"]:
                print(f"  Step {step['step']}: {step['action']}")

        # Final answer
        print("\n" + "=" * 70)
        print("💬 ANSWER")
        print("=" * 70)
        print(f"\n{result['response']}\n")
        print("=" * 70)

    def save_results(self, result: dict):
        """Save results to file.

        Args:
            result: Result dict
        """
        if self.get_yes_no("Save results to file?"):
            default_path = "data/outputs/interactive_result.json"
            path = input(f"Output path (default: {default_path}): ").strip()
            if not path:
                path = default_path

            output_path = Path(path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)

            print(f"✓ Results saved to {output_path}")

    def run(self):
        """Run interactive session."""
        self.clear_screen()
        self.print_header()

        try:
            # Load configuration
            self.load_config()

            # Initialize agent
            self.initialize_agent()

            # Main loop
            while True:
                print("\n" + "=" * 70)
                print("📁 INPUT FILES")
                print("=" * 70)

                # Get file paths
                video_path = self.get_file_path("video", optional=True)
                audio_path = self.get_file_path("audio", optional=True)

                if not video_path and not audio_path:
                    print("\n❌ Error: At least one of video or audio is required")
                    if not self.get_yes_no("Try again?"):
                        break
                    continue

                # Get query
                query = self.get_query()

                # Select features
                features = self.select_features()

                # Ask for verbose output
                verbose = self.get_yes_no("Show detailed information?")

                # Process
                result = self.process_query(video_path, audio_path, query, features)

                # Display results
                self.display_results(result, verbose=verbose)

                # Save results
                self.save_results(result)

                # Continue?
                print("\n" + "=" * 70)
                if not self.get_yes_no("Process another query?"):
                    break

            print("\n" + "=" * 70)
            print("👋 Thank you for using Sonic O1 Agent!")
            print("=" * 70 + "\n")

        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logger.exception("Error in interactive session")
        finally:
            if self.agent:
                print("\n🔄 Shutting down agent...")
                print("✓ Agent shutdown complete")


def main():
    """Entry point."""
    cli = InteractiveCLI()
    cli.run()


if __name__ == "__main__":
    main()
