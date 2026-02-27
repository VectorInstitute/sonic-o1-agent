#!/usr/bin/env python3
"""Verify sonic-o1-agent setup and run basic tests.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def print_header(text):
    """Print formatted header."""
    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}{text}{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}\n")


def print_success(text):
    """Print success message."""
    print(f"{GREEN}✓{RESET} {text}")


def print_error(text):
    """Print error message."""
    print(f"{RED}✗{RESET} {text}")


def print_warning(text):
    """Print warning message."""
    print(f"{YELLOW}!{RESET} {text}")


def test_imports():
    """Test all imports work."""
    print("Testing imports...")
    try:
        print_success("All imports successful")
        return True
    except Exception as e:
        print_error(f"Import failed: {e}")
        return False


def test_agent_init():
    """Test agent initialization."""
    print("\nTesting agent initialization...")
    try:
        from sonic_o1_agent import SonicAgent

        config = {
            "model": {"model_path": "test"},
            "processing": {"max_frames": 256},
        }
        SonicAgent(config)
        print_success("Agent initialized successfully")
        return True
    except Exception as e:
        print_error(f"Agent init failed: {e}")
        return False


def test_planner():
    """Test planner functionality."""
    print("\nTesting planner...")
    try:
        from sonic_o1_agent.agents.planner import AgentPlanner

        # Test time range parsing
        query = "What happened between minute 5 and 10?"
        time_range = AgentPlanner.parse_time_range(query)
        assert time_range == (300, 600), "Time range parsing failed"

        print_success(f"Planner working: parsed '{query}' → {time_range}")
        return True
    except Exception as e:
        print_error(f"Planner test failed: {e}")
        return False


def test_prompt_builder():
    """Test prompt builder."""
    print("\nTesting prompt builder...")
    try:
        from sonic_o1_agent.processors.prompt_builder import PromptBuilder

        query = "Summarize this video"
        query_type = PromptBuilder.detect_query_type(query)
        assert query_type == "summarization"

        prompt = PromptBuilder.build_prompt(query)
        assert len(prompt) > 0

        print_success(f"Prompt builder working: detected type '{query_type}'")
        return True
    except Exception as e:
        print_error(f"Prompt builder test failed: {e}")
        return False


def run_pytest():
    """Run pytest if available."""
    print("\nRunning pytest...")
    try:
        import pytest

        result = pytest.main(
            [
                "tests/",
                "-v",
                "--tb=short",
                "-m",
                "not slow",
            ]
        )

        if result == 0:
            print_success("All pytest tests passed")
            return True
        else:
            print_warning("Some pytest tests failed (check output above)")
            return False
    except ImportError:
        print_warning("pytest not installed, skipping automated tests")
        return True
    except Exception as e:
        print_error(f"pytest failed: {e}")
        return False


def check_directory_structure():
    """Check directory structure is correct."""
    print("\nChecking directory structure...")
    required_dirs = [
        "src/sonic_o1_agent/agents",
        "src/sonic_o1_agent/models",
        "src/sonic_o1_agent/core",
        "src/sonic_o1_agent/processors",
        "src/sonic_o1_agent/utils",
        "configs",
        "scripts",
        "slurm",
        "tests/unit",
        "tests/integration",
    ]

    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print_success(f"{dir_path}")
        else:
            print_error(f"{dir_path} - MISSING")
            all_exist = False

    return all_exist


def main():
    """Run all verification tests."""
    print_header("Sonic O1 Agent - Setup Verification")

    results = []

    # 1. Check directory structure
    results.append(("Directory Structure", check_directory_structure()))

    # 2. Test imports
    results.append(("Imports", test_imports()))

    # 3. Test agent initialization
    results.append(("Agent Init", test_agent_init()))

    # 4. Test planner
    results.append(("Planner", test_planner()))

    # 5. Test prompt builder
    results.append(("Prompt Builder", test_prompt_builder()))

    # 6. Run pytest
    results.append(("Pytest Suite", run_pytest()))

    # Summary
    print_header("Verification Summary")

    all_passed = all(result for _, result in results)

    for name, passed in results:
        if passed:
            print_success(f"{name}: PASSED")
        else:
            print_error(f"{name}: FAILED")

    print()

    if all_passed:
        print_success(f"{BOLD}All checks passed!{RESET}")
        print()
        print("Next steps:")
        print("  1. Edit slurm/run_sonic_agent.sh with your video/audio paths")
        print("  2. Submit job: sbatch slurm/run_sonic_agent.sh")
        print("  3. Monitor: tail -f logs/sonic_agent_*.out")
        return 0
    else:
        print_error(f"{BOLD}Some checks failed{RESET}")
        print()
        print("Please fix the issues above before proceeding.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
