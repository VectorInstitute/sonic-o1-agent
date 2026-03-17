"""Temporal index builder for real timestamp grounding.

Splits sampled video frames into segments, generates a short caption
per segment via the VLM, and assembles a timestamped text index that
downstream prompts can reference for accurate temporal citations.

When the model is running in **server mode** (vLLM HTTP server),
segment captioning is parallelised via ``ThreadPoolExecutor`` so
that vLLM's continuous batching can process them concurrently.
In embedded mode, segments are captioned serially (the in-process
vLLM engine is single-threaded).

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

import logging
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Prompt sent to the model for each segment.
CAPTION_PROMPT = (
    "Briefly describe what is shown and said in this video clip. "
    "Be concise (2-3 sentences). Focus on key events, visual "
    "content, and any spoken words."
)


class TemporalIndexBuilder:
    """Build a timestamped text index from video segments.

    The index is a plain-text string of the form::

        [0s - 90s] Speaker introduces himself and the topic.
        [90s - 180s] Diagram of patient history structure shown.
        ...

    This string is injected into the main prompt so the model can
    cite real timestamps when answering queries.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialise the builder.

        All settings are read from ``config`` (typically the
        ``temporal_index`` section of ``agent_config.yaml``).

        Args:
            config: Configuration dict.  Recognised keys:

                - ``min_duration_sec`` -- skip indexing for videos
                  shorter than this (default ``120``).
                - ``num_segments`` -- how many segments to split the
                  video into (default ``10``).
                - ``max_frames_per_segment`` -- frames sampled per
                  segment caption (default ``16``).
                - ``caption_max_tokens`` -- max tokens per caption
                  (default ``256``).
                - ``max_parallel`` -- max concurrent caption requests
                  in server mode (default ``5``).
        """
        config = config or {}
        self.min_duration_sec = config.get("min_duration_sec", 120)
        self.num_segments = config.get("num_segments", 10)
        self.max_frames_per_segment = config.get("max_frames_per_segment", 16)
        self.caption_max_tokens = config.get("caption_max_tokens", 256)
        self.max_parallel = config.get("max_parallel", 5)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_index(
        self,
        model: Any,
        video_path: Optional[str],
        audio_path: Optional[str],
        duration_sec: float,
    ) -> str:
        """Build the temporal index for a video.

        Videos shorter than ``min_duration_sec`` are skipped (returns
        an empty string) because the model can process the entire clip
        in a single pass without needing segment-level grounding.

        Automatically selects parallel or serial captioning based on
        whether the model is in server mode.

        Args:
            model: A loaded ``Qwen3OmniModel`` (or compatible) instance.
            video_path: Path to the video file.
            audio_path: Path to the audio file (may be ``None``).
            duration_sec: Total video duration in seconds.

        Returns:
            Multi-line timestamped index string.  Returns an empty
            string when the video is ``None``, too short, or
            captioning fails entirely.
        """
        if not video_path:
            return ""

        if duration_sec < self.min_duration_sec:
            logger.info(
                "Skipping temporal index: video %.1fs < min_duration_sec %ds",
                duration_sec,
                self.min_duration_sec,
            )
            return ""

        segments = self._compute_segments(duration_sec)
        logger.info(
            "Building temporal index: %d segments over %.1fs video",
            len(segments),
            duration_sec,
        )

        # Choose parallel vs serial based on model backend
        is_server_mode = getattr(model, "_server_mode", False)

        if is_server_mode and len(segments) > 1:
            entries = self._caption_segments_parallel(
                model, video_path, audio_path, segments
            )
        else:
            entries = self._caption_segments_serial(
                model, video_path, audio_path, segments
            )

        if not entries:
            logger.warning("Temporal index is empty -- captioning failed")
            return ""

        index_text = "\n".join(entries)
        logger.info(
            "Temporal index built: %d / %d segments captioned",
            len(entries),
            len(segments),
        )
        return index_text

    # ------------------------------------------------------------------
    # Serial captioning (embedded mode)
    # ------------------------------------------------------------------
    def _caption_segments_serial(
        self,
        model: Any,
        video_path: str,
        audio_path: Optional[str],
        segments: List[Tuple[float, float]],
    ) -> List[str]:
        """Caption segments one at a time (original behaviour).

        Used in embedded mode where the in-process vLLM engine
        processes one request at a time.
        """
        logger.info("Temporal indexing: serial mode (%d segments)", len(segments))
        entries: List[str] = []

        for start_sec, end_sec in segments:
            caption = self._caption_segment(
                model=model,
                video_path=video_path,
                audio_path=audio_path,
                start_sec=start_sec,
                end_sec=end_sec,
                max_frames=self.max_frames_per_segment,
            )
            if caption:
                label = f"[{start_sec:.0f}s - {end_sec:.0f}s]"
                entries.append(f"{label} {caption}")

        return entries

    # ------------------------------------------------------------------
    # Parallel captioning (server mode)
    # ------------------------------------------------------------------
    def _caption_segments_parallel(
        self,
        model: Any,
        video_path: str,
        audio_path: Optional[str],
        segments: List[Tuple[float, float]],
    ) -> List[str]:
        """Caption segments concurrently via ThreadPoolExecutor.

        Each segment fires a separate HTTP request to the vLLM server.
        vLLM's continuous batching processes them concurrently, giving
        near-linear speedup.

        Results are collected and sorted back to original segment order.
        """
        max_workers = min(self.max_parallel, len(segments))
        logger.info(
            "Temporal indexing: parallel mode (%d segments, %d workers)",
            len(segments),
            max_workers,
        )

        # Each future maps to (segment_index, start_sec, end_sec)
        results: Dict[int, str] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}

            for idx, (start_sec, end_sec) in enumerate(segments):
                future = executor.submit(
                    self._caption_segment,
                    model=model,
                    video_path=video_path,
                    audio_path=audio_path,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    max_frames=self.max_frames_per_segment,
                )
                future_to_idx[future] = (idx, start_sec, end_sec)

            for future in as_completed(future_to_idx):
                idx, start_sec, end_sec = future_to_idx[future]
                try:
                    caption = future.result()
                    if caption:
                        label = f"[{start_sec:.0f}s - {end_sec:.0f}s]"
                        results[idx] = f"{label} {caption}"
                        logger.debug(
                            "Segment %d/%d captioned (%.0fs-%.0fs)",
                            idx + 1,
                            len(segments),
                            start_sec,
                            end_sec,
                        )
                except Exception as exc:
                    logger.warning(
                        "Parallel caption failed for segment %d (%.0fs-%.0fs): %s",
                        idx,
                        start_sec,
                        end_sec,
                        exc,
                    )

        # Return in original segment order
        return [results[i] for i in sorted(results.keys())]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _compute_segments(self, duration_sec: float) -> List[Tuple[float, float]]:
        """Divide the video duration into equal-length segments.

        Args:
            duration_sec: Total video duration in seconds.

        Returns:
            List of ``(start_sec, end_sec)`` tuples.
        """
        n = min(self.num_segments, max(1, int(math.ceil(duration_sec / 30))))
        seg_len = duration_sec / n
        return [
            (round(i * seg_len, 2), round(min((i + 1) * seg_len, duration_sec), 2))
            for i in range(n)
        ]

    def _caption_segment(
        self,
        model: Any,
        video_path: str,
        audio_path: Optional[str],
        start_sec: float,
        end_sec: float,
        max_frames: int,
    ) -> str:
        """Generate a short caption for one video segment.

        Uses the model's ``generate`` method with ``video_start`` /
        ``video_end`` kwargs so only the relevant slice is decoded.

        Args:
            model: Loaded VLM instance.
            video_path: Path to the full video file.
            audio_path: Path to audio (may be ``None``).
            start_sec: Segment start in seconds.
            end_sec: Segment end in seconds.
            max_frames: Frames to sample within this segment.

        Returns:
            Caption string, or empty string on failure.
        """
        try:
            # Slice both video AND audio to this segment's time range
            # so only the relevant chunk is loaded -- not the full file.
            response_text, _ = model.generate(
                video_path=video_path,
                audio_path=audio_path,
                prompt=CAPTION_PROMPT,
                max_frames=max_frames,
                video_start=start_sec,
                video_end=end_sec,
                audio_start=start_sec,
                audio_end=end_sec,
                max_new_tokens=self.caption_max_tokens,
            )
            # Collapse to single line for index readability
            return " ".join(response_text.strip().split())
        except Exception as exc:
            logger.warning(
                "Caption failed for segment %.0fs-%.0fs: %s",
                start_sec,
                end_sec,
                exc,
            )
            return ""
