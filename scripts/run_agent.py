#!/usr/bin/env python3
"""Backward-compatible wrapper; prefer ``sonic-o1-agent`` after ``pip install -e .``."""

import sys
from pathlib import Path

try:
    from sonic_o1_agent.cli import main
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from sonic_o1_agent.cli import main

if __name__ == "__main__":
    main()
