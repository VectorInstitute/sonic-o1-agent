"""Unit tests for AgentPlanner.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

from pathlib import Path

from sonic_o1_agent.agents.planner import AgentPlanner


class TestTimeRangeParsing:
    """Test time range parsing from queries."""

    def test_parse_minute_to_minute(self):
        """Test 'minute X to Y' pattern."""
        query = "what happened from minute 5 to 10"
        result = AgentPlanner.parse_time_range(query)
        assert result == (300, 600)

    def test_parse_between_minutes(self):
        """Test 'between minute X and Y' pattern."""
        query = "what happened between minute 2 and 3"
        result = AgentPlanner.parse_time_range(query)
        assert result == (120, 180)

    def test_parse_at_minute(self):
        """Test 'at minute X' pattern with window."""
        query = "what happened at minute 5"
        result = AgentPlanner.parse_time_range(query)
        assert result == (270, 330)  # 5*60 ± 30 seconds

    def test_parse_around_minute(self):
        """Test 'around minute X' pattern."""
        query = "around minute 10"
        result = AgentPlanner.parse_time_range(query)
        assert result == (570, 630)

    def test_parse_first_minutes(self):
        """Test 'first X minutes' pattern."""
        query = "summarize the first 10 minutes"
        result = AgentPlanner.parse_time_range(query)
        assert result == (0, 600)

    def test_no_time_range(self):
        """Test query without time range."""
        query = "summarize this video"
        result = AgentPlanner.parse_time_range(query)
        assert result is None


class TestModalityDetection:
    """Test modality detection."""

    def test_both_modalities(self):
        """Test with both video and audio."""
        video = Path(__file__)  # Use test file as dummy
        audio = Path(__file__)
        result = AgentPlanner.determine_modalities(video, audio)
        assert result["has_video"] is True
        assert result["has_audio"] is True

    def test_video_only(self):
        """Test with video only."""
        video = Path(__file__)
        result = AgentPlanner.determine_modalities(video, None)
        assert result["has_video"] is True
        assert result["has_audio"] is False

    def test_audio_only(self):
        """Test with audio only."""
        audio = Path(__file__)
        result = AgentPlanner.determine_modalities(None, audio)
        assert result["has_video"] is False
        assert result["has_audio"] is True

    def test_no_modalities(self):
        """Test with neither."""
        result = AgentPlanner.determine_modalities(None, None)
        assert result["has_video"] is False
        assert result["has_audio"] is False

    def test_nonexistent_files(self):
        """Test with paths that don't exist."""
        video = Path("/nonexistent/video.mp4")
        audio = Path("/nonexistent/audio.m4a")
        result = AgentPlanner.determine_modalities(video, audio)
        assert result["has_video"] is False
        assert result["has_audio"] is False


class TestProcessingPlan:
    """Test complete processing plan generation."""

    def test_plan_basic_query(self):
        """Test plan for basic query."""
        plan = AgentPlanner.plan_processing(
            video_path=None,
            audio_path=None,
            query="summarize this",
            config={"max_frames": 256},
        )
        assert "modalities" in plan
        assert "query_type" in plan
        assert plan["should_segment"] is False

    def test_plan_includes_config(self):
        """Test that config values are included."""
        config = {"max_frames": 128, "max_audio_chunks": 5}
        plan = AgentPlanner.plan_processing(
            video_path=None,
            audio_path=None,
            query="test",
            config=config,
        )
        assert plan["max_frames"] == 128
