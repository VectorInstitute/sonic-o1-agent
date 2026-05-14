"""Video and audio segmentation utilities.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class VideoSegmenter:
    """Utilities for extracting video and audio segments."""

    def __init__(self):
        """Initialize video segmenter."""
        pass

    def extract_video_segment(
        self,
        input_path: Path,
        start_time: float,
        end_time: float,
        output_path: Path,
    ) -> None:
        """Extract video segment using ffmpeg.

        Args:
            input_path: Path to input video file
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Path to output video file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        duration = end_time - start_time

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_time),
            "-i",
            str(input_path),
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-avoid_negative_ts",
            "make_zero",
            str(output_path),
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logger.debug(
                f"Extracted video segment: {start_time}s-{end_time}s -> {output_path}"
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract video segment: {e.stderr.decode()}")
            raise

    def extract_audio_segment(
        self,
        input_path: Path,
        start_time: float,
        end_time: float,
        output_path: Path,
        output_format: str = "m4a",
    ) -> None:
        """Extract audio segment using ffmpeg.

        Args:
            input_path: Path to input audio file
            start_time: Start time in seconds
            end_time: End time in seconds
            output_path: Path to output audio file
            output_format: Output audio format (m4a, wav, etc.)
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        duration = end_time - start_time

        if output_format == "wav":
            codec = "pcm_s16le"
        elif output_format == "m4a":
            codec = "aac"
        else:
            codec = "copy"

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_time),
            "-i",
            str(input_path),
            "-t",
            str(duration),
            "-c:a",
            codec,
            "-vn",
            str(output_path),
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logger.debug(
                f"Extracted audio segment: {start_time}s-{end_time}s -> {output_path}"
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to extract audio segment: {e.stderr.decode()}")
            raise

    def convert_audio_format(
        self,
        input_path: Path,
        output_path: Path,
        output_format: str = "wav",
    ) -> Path:
        """Convert audio file to different format.

        Args:
            input_path: Path to input audio file
            output_path: Path to output audio file
            output_format: Target format (wav, m4a, etc.)

        Returns:
            Path to converted audio file
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_format == "wav":
            codec = "pcm_s16le"
        elif output_format == "m4a":
            codec = "aac"
        else:
            codec = "copy"

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-c:a",
            codec,
            "-vn",
            str(output_path),
        ]

        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            logger.debug(f"Converted audio: {input_path} -> {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to convert audio: {e.stderr.decode()}")
            raise
