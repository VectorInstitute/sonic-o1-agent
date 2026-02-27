"""Unit tests for PromptBuilder.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

from sonic_o1_agent.processors.prompt_builder import PromptBuilder


class TestQueryTypeDetection:
    """Test automatic query type detection."""

    def test_detect_summarization(self):
        """Test summarization query detection."""
        queries = [
            "summarize this video",
            "give me a summary",
            "what happened in this video",
            "provide an overview",
            "describe what happened",
        ]
        for query in queries:
            result = PromptBuilder.detect_query_type(query)
            assert result == "summarization", f"Failed for: {query}"

    def test_detect_temporal(self):
        """Test temporal query detection."""
        queries = [
            "what happened at minute 5",
            "when did they discuss X",
            "between minute 2 and 3",
            "what's the timestamp for Y",
            "at what time did this happen",
        ]
        for query in queries:
            result = PromptBuilder.detect_query_type(query)
            assert result == "temporal", f"Failed for: {query}"

    def test_detect_qa(self):
        """Test Q&A query detection."""
        queries = [
            "what is the main topic",
            "who was speaking",
            "where did this happen",
            "how did they solve the problem",
            "why did this occur",
        ]
        for query in queries:
            result = PromptBuilder.detect_query_type(query)
            assert result == "qa", f"Failed for: {query}"

    def test_detect_generic(self):
        """Test generic query detection."""
        queries = [
            "tell me about this",
            "analyze the content",
            "help me understand",
        ]
        for query in queries:
            result = PromptBuilder.detect_query_type(query)
            assert result == "generic", f"Failed for: {query}"


class TestPromptBuilding:
    """Test prompt construction."""

    def test_build_generic_prompt(self):
        """Test generic prompt building."""
        query = "analyze this video"
        prompt = PromptBuilder.build_prompt(query)
        assert query in prompt
        assert isinstance(prompt, str)

    def test_build_summarization_prompt(self):
        """Test summarization prompt."""
        query = "summarize"
        context = {"duration_seconds": 300}
        prompt = PromptBuilder.build_prompt(
            query, context=context, query_type="summarization"
        )
        assert "summary" in prompt.lower()
        assert "300" in prompt

    def test_build_qa_prompt(self):
        """Test Q&A prompt."""
        query = "What is the main topic?"
        prompt = PromptBuilder.build_prompt(query, query_type="qa")
        assert query in prompt
        assert "answer" in prompt.lower()

    def test_build_temporal_prompt(self):
        """Test temporal prompt with segment info."""
        query = "What happened here?"
        context = {"segment_start": 120, "segment_end": 180}
        prompt = PromptBuilder.build_prompt(
            query, context=context, query_type="temporal"
        )
        assert "120" in prompt
        assert "180" in prompt
        assert "timestamp" in prompt.lower()

    def test_prompt_with_previous_context(self):
        """Test prompt includes conversation history."""
        query = "tell me more"
        context = {
            "previous_interactions": [
                {"query": "what happened", "response": "X occurred"}
            ]
        }
        prompt = PromptBuilder.build_prompt(query, context=context)
        assert "previous" in prompt.lower() or "continuing" in prompt.lower()


class TestPromptDetails:
    """Test specific prompt features."""

    def test_summarization_includes_structure(self):
        """Test summarization prompt has expected structure."""
        prompt = PromptBuilder._build_summarization_prompt("summarize", {})
        assert "summary" in prompt.lower()
        assert any(str(i) in prompt for i in range(1, 4))  # Has numbered points

    def test_temporal_prompt_mentions_timestamps(self):
        """Test temporal prompts ask for timestamps."""
        prompt = PromptBuilder._build_temporal_prompt(
            "when did X happen", {"segment_start": 0, "segment_end": 60}
        )
        assert "timestamp" in prompt.lower()

    def test_qa_prompt_clear_format(self):
        """Test Q&A prompt is clear."""
        prompt = PromptBuilder._build_qa_prompt("who spoke first", {})
        assert "who spoke first" in prompt.lower()
        assert "answer" in prompt.lower()
