"""Agent planning and task decomposition.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class AgentPlanner:
    """Plan and decompose tasks for the agent."""

    @staticmethod
    def parse_time_range(query: str) -> Optional[Tuple[float, float]]:
        """Extract time range from user query.

        Args:
            query: User's query text

        Returns:
            Tuple of (start_seconds, end_seconds) or None
        """
        query_lower = query.lower()

        # Pattern: "minute X to Y" or "minutes X to Y"
        pattern1 = r"minute[s]?\s+(\d+)\s+to\s+(\d+)"
        match = re.search(pattern1, query_lower)
        if match:
            start_min = int(match.group(1))
            end_min = int(match.group(2))
            return (start_min * 60, end_min * 60)

        # Pattern: "between minute X and Y"
        pattern2 = r"between\s+minute[s]?\s+(\d+)\s+and\s+(\d+)"
        match = re.search(pattern2, query_lower)
        if match:
            start_min = int(match.group(1))
            end_min = int(match.group(2))
            return (start_min * 60, end_min * 60)

        # Pattern: "at minute X" or "around minute X"
        pattern3 = r"(?:at|around)\s+minute[s]?\s+(\d+)"
        match = re.search(pattern3, query_lower)
        if match:
            target_min = int(match.group(1))
            # Create 1-minute window around the target
            return (max(0, target_min * 60 - 30), (target_min * 60 + 30))

        # Pattern: "first X minutes" or "last X minutes"
        pattern4 = r"(first|last)\s+(\d+)\s+minute[s]?"
        match = re.search(pattern4, query_lower)
        if match:
            position = match.group(1)
            duration_min = int(match.group(2))
            if position == "first":
                return (0, duration_min * 60)
            # For "last", we'll need video duration - handled by caller

        return None

    @staticmethod
    def should_segment_video(
        video_path: Path,
        query: str,
        video_duration: Optional[float] = None,
    ) -> Tuple[bool, Optional[Tuple[float, float]]]:
        """Determine if video should be segmented.

        Args:
            video_path: Path to video file
            query: User's query
            video_duration: Video duration in seconds (optional)

        Returns:
            Tuple of (should_segment, time_range)
        """
        # Parse time range from query
        time_range = AgentPlanner.parse_time_range(query)

        if time_range is None:
            return False, None

        # Get video duration if not provided
        if video_duration is None:
            try:
                from sonic_o1_agent.core.multimodal_engine import (
                    get_video_metadata,
                )

                metadata = get_video_metadata(str(video_path))
                video_duration = metadata.total_frames / metadata.fps
            except Exception as e:
                logger.warning(f"Could not get video duration: {e}")
                return False, None

        # Handle "last X minutes" pattern
        if "last" in query.lower() and time_range[0] == 0:
            duration_requested = time_range[1]
            time_range = (
                max(0, video_duration - duration_requested),
                video_duration,
            )

        segment_duration = time_range[1] - time_range[0]

        # Only segment if:
        # 1. Video is longer than 5 minutes AND
        # 2. Segment is less than 80% of total duration
        if video_duration > 300 and segment_duration < video_duration * 0.8:
            logger.info(
                f"Planning to segment video: {time_range[0]}s - "
                f"{time_range[1]}s (from {video_duration}s total)"
            )
            return True, time_range

        return False, None

    @staticmethod
    def determine_modalities(
        video_path: Optional[Path],
        audio_path: Optional[Path],
    ) -> Dict[str, bool]:
        """Determine which modalities are available.

        Args:
            video_path: Path to video file (optional)
            audio_path: Path to audio file (optional)

        Returns:
            Dict with 'has_video' and 'has_audio' flags
        """
        has_video = video_path is not None and video_path.exists()
        has_audio = audio_path is not None and audio_path.exists()

        return {"has_video": has_video, "has_audio": has_audio}

    @staticmethod
    def plan_processing(
        video_path: Optional[Path],
        audio_path: Optional[Path],
        query: str,
        config: Optional[Dict] = None,
    ) -> Dict:
        """Create processing plan for the agent.

        Args:
            video_path: Path to video file (optional)
            audio_path: Path to audio file (optional)
            query: User's query
            config: Agent configuration

        Returns:
            Processing plan dict with all necessary parameters
        """
        config = config or {}

        # Determine modalities
        modalities = AgentPlanner.determine_modalities(video_path, audio_path)

        # Check if segmentation needed
        should_segment = False
        time_range = None
        if video_path and modalities["has_video"]:
            should_segment, time_range = AgentPlanner.should_segment_video(
                video_path, query
            )

        # Detect query type
        from sonic_o1_agent.processors.prompt_builder import PromptBuilder

        query_type = PromptBuilder.detect_query_type(query)

        # Get video duration for temporal grounding
        duration_seconds = None
        if video_path and modalities["has_video"]:
            try:
                from sonic_o1_agent.core.multimodal_engine import (
                    get_video_metadata,
                )

                meta = get_video_metadata(str(video_path))
                duration_seconds = round(meta.total_frames / meta.fps, 2)
            except Exception:
                pass

        # Build plan
        plan = {
            "modalities": modalities,
            "query_type": query_type,
            "should_segment": should_segment,
            "time_range": time_range,
            "max_frames": config.get("max_frames", 256),
            "max_audio_chunks": config.get("max_audio_chunks", None),
            "duration_seconds": duration_seconds,
            "processing_strategy": "single_pass",  # Can be extended later
        }

        logger.info(f"Generated processing plan: {plan}")

        return plan
