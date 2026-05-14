"""Pytest configuration and shared fixtures.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

import pytest


@pytest.fixture
def minimal_agent_config() -> dict:
    """Minimal agent config sufficient for workflow init and graph tests.

    Returns:
        Dict with model and processing keys; no real model is loaded
        when only graph structure or validation is tested.
    """
    return {
        "model": {"model_path": "test/model"},
        "processing": {},
    }


@pytest.fixture
def my_test_number() -> int:
    """Sample numeric fixture for placeholder tests.

    Returns:
        int: A test number (42).
    """
    return 42
