"""Unit tests for MultiStepPlanner.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

from sonic_o1_agent.agents.planner_advanced import MultiStepPlanner


class TestShouldDecompose:
    """Test should_decompose keyword detection."""

    def test_compare_triggers_decompose(self):
        """Queries with 'compare' should be decomposed."""
        planner = MultiStepPlanner()
        assert planner.should_decompose("Compare the two approaches") is True

    def test_contrast_triggers_decompose(self):
        """Queries with 'contrast' should be decomposed."""
        planner = MultiStepPlanner()
        assert planner.should_decompose("Contrast A and B") is True

    def test_both_triggers_decompose(self):
        """Queries with 'both' should be decomposed."""
        planner = MultiStepPlanner()
        assert planner.should_decompose("Analyze both speakers") is True

    def test_simple_query_no_decompose(self):
        """Simple summarization query should not be decomposed."""
        planner = MultiStepPlanner()
        assert planner.should_decompose("Summarize this video") is False

    def test_why_triggers_decompose(self):
        """Queries with 'explain why' should be decomposed."""
        planner = MultiStepPlanner()
        assert planner.should_decompose("Explain why this happened") is True


class TestDecomposeQuery:
    """Test decompose_query returns step list."""

    def test_comparison_returns_multiple_steps(self):
        """Comparison query returns multiple steps."""
        planner = MultiStepPlanner()
        steps = planner.decompose_query("Compare defense vs prosecutor")
        assert len(steps) >= 2
        assert all("step_id" in s and "query" in s for s in steps)

    def test_default_returns_single_step(self):
        """Query without complex keywords returns single comprehensive step."""
        planner = MultiStepPlanner()
        steps = planner.decompose_query("Summarize the content")
        assert len(steps) == 1
        assert steps[0]["step_id"] == 1
        assert steps[0]["action"] == "analyze"

    def test_max_steps_limits_result(self):
        """max_steps in config caps number of steps returned."""
        planner = MultiStepPlanner(config={"max_steps": 2})
        steps = planner.decompose_query("Compare A and B")
        assert len(steps) <= 2

    def test_step_has_required_keys(self):
        """Each step has step_id, action, description, query."""
        planner = MultiStepPlanner()
        steps = planner.decompose_query("What happened first, then after?")
        for step in steps:
            assert "step_id" in step
            assert "action" in step
            assert "description" in step
            assert "query" in step


class TestMultiStepPlannerInit:
    """Test MultiStepPlanner initialization."""

    def test_default_max_steps(self):
        """Default max_steps is 10."""
        planner = MultiStepPlanner()
        assert planner.max_steps == 10

    def test_config_max_steps(self):
        """max_steps read from config."""
        planner = MultiStepPlanner(config={"max_steps": 5})
        assert planner.max_steps == 5

    def test_empty_config_ok(self):
        """Accepts None or empty config."""
        planner = MultiStepPlanner(config=None)
        assert planner.config == {}
        assert planner.max_steps == 10
