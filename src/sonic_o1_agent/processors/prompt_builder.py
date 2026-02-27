"""Dynamic prompt construction for different query types.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Build contextual prompts based on query type and context."""

    @staticmethod
    def build_prompt(
        query: str,
        context: Optional[Dict] = None,
        query_type: Optional[str] = None,
    ) -> str:
        """Build a prompt based on query and context.

        Args:
            query: User's query
            context: Optional context (duration, previous interactions, etc.)
            query_type: Optional hint about query type (summarization, qa, etc.)

        Returns:
            Formatted prompt string
        """
        context = context or {}

        # If query type is explicitly provided, use specialized prompts
        if query_type == "summarization":
            prompt = PromptBuilder._build_summarization_prompt(query, context)
        elif query_type == "qa":
            prompt = PromptBuilder._build_qa_prompt(query, context)
        elif query_type == "temporal":
            prompt = PromptBuilder._build_temporal_prompt(query, context)
        else:
            prompt = PromptBuilder._build_generic_prompt(query, context)

        # Append temporal grounding so the model cites seconds
        prompt = PromptBuilder._append_temporal_grounding(prompt, context)
        return prompt

    @staticmethod
    def _build_summarization_prompt(query: str, context: Dict) -> str:
        """Build prompt for summarization tasks."""
        duration = context.get("duration_seconds")

        base_prompt = "Please analyze this video and provide a comprehensive summary."

        if duration:
            base_prompt += f"\n\nThe video is approximately {duration} seconds long."

        if query and query.lower() != "summarize":
            base_prompt += f"\n\nSpecific request: {query}"

        base_prompt += (
            "\n\nProvide:\n"
            "1. A detailed summary of the key events and content\n"
            "2. Important timestamps and their significance\n"
            "3. Key points or takeaways"
        )

        return base_prompt

    @staticmethod
    def _build_qa_prompt(query: str, context: Dict) -> str:
        """Build prompt for question-answering tasks."""
        base_prompt = (
            "Based on the video and audio content, please answer "
            "the following question:\n\n"
        )

        base_prompt += f"{query}\n\n"
        base_prompt += (
            "Provide a clear and concise answer based on what you observe "
            "in the video and hear in the audio."
        )

        return base_prompt

    @staticmethod
    def _build_temporal_prompt(query: str, context: Dict) -> str:
        """Build prompt for temporal localization tasks."""
        segment_start = context.get("segment_start")
        segment_end = context.get("segment_end")

        base_prompt = query

        if segment_start is not None and segment_end is not None:
            base_prompt += (
                f"\n\nNote: This segment covers the time range "
                f"{segment_start}s to {segment_end}s of the original video."
            )

        base_prompt += (
            "\n\nPlease provide specific timestamps in your answer "
            "when referring to events."
        )

        return base_prompt

    @staticmethod
    def _build_generic_prompt(query: str, context: Dict) -> str:
        """Build generic prompt for any query."""
        previous_context = context.get("previous_interactions")

        base_prompt = query

        if previous_context:
            base_prompt = "Continuing our previous conversation:\n\n" + base_prompt

        return base_prompt

    @staticmethod
    def _append_temporal_grounding(prompt: str, context: Dict) -> str:
        """Prepend temporal grounding directive so the model cites seconds.

        Placed BEFORE the main prompt so the model treats timestamp
        citation as a primary requirement, not an afterthought.
        """
        parts = []

        duration = context.get("duration_seconds")
        max_frames = context.get("max_frames")
        segment_start = context.get("segment_start")
        segment_end = context.get("segment_end")

        if max_frames and duration:
            interval = round(duration / max(max_frames, 1), 1)
            parts.append(
                f"The video is {round(duration, 1)}s long. You are "
                f"viewing {max_frames} uniformly sampled frames "
                f"(one every ~{interval}s). The audio covers the "
                f"full duration."
            )
        elif max_frames:
            parts.append(
                f"You are viewing {max_frames} uniformly sampled frames from the video."
            )

        if segment_start is not None and segment_end is not None:
            parts.append(
                f"This content covers {segment_start}s - "
                f"{segment_end}s of the original video."
            )

        if not parts:
            return prompt

        temporal_index = context.get("temporal_index")

        # When a temporal index is available, use it as the primary
        # grounding source; otherwise fall back to the sampling hint.
        if temporal_index:
            directive = (
                "IMPORTANT: A timestamped content index of the video "
                "is provided below. Use it to cite accurate timestamps "
                "in seconds when describing events. Format: "
                "'Around 30s, ...' or 'Between 120s and 180s, ...'."
            )
            grounding = "\n".join(parts)
            return (
                f"{directive}\n\n{grounding}\n\n"
                f"--- VIDEO INDEX ---\n{temporal_index}\n"
                f"--- END INDEX ---\n\n{prompt}"
            )

        directive = (
            "IMPORTANT: Include approximate timestamps in seconds "
            "for key events or points you describe. Use the audio "
            "and visual progression to estimate when things occur. "
            "Format: 'Around 30s, ...' or 'Between 120s and 180s, "
            "...'."
        )
        grounding = "\n".join(parts)
        return f"{directive}\n\n{grounding}\n\n{prompt}"

    @staticmethod
    def detect_query_type(query: str) -> str:
        """Detect the type of query based on keywords.

        Args:
            query: User's query

        Returns:
            Query type: 'summarization', 'qa', 'temporal', or 'generic'
        """
        query_lower = query.lower()

        # Temporal first so "what happened at minute 5" is temporal, not summarization
        temporal_keywords = [
            "when",
            "at what time",
            "minute",
            "second",
            "timestamp",
            "between",
        ]
        if any(kw in query_lower for kw in temporal_keywords):
            return "temporal"

        # Summarization keywords
        summarization_keywords = [
            "summarize",
            "summary",
            "overview",
            "what happened",
            "describe",
            "explain the video",
        ]
        if any(kw in query_lower for kw in summarization_keywords):
            return "summarization"

        # Question keywords
        qa_keywords = ["what", "who", "where", "why", "how", "which"]
        if any(query_lower.startswith(kw) for kw in qa_keywords):
            return "qa"

        return "generic"
