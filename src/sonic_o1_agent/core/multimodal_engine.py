"""Multimodal processing engine using PyAV.

Production-grade PyAV-based multimodal processing for video, audio, and image data.
Memory-efficient frame sampling and optimized audio loading with chunking.

This module provides orchestration functions that coordinate video and audio processing.
For direct access to video/audio processors, see video_processor.py and audio_processor.py.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

# Import from split modules
from sonic_o1_agent.core.audio_processor import (
    load_audio_pyav,
    process_audio_with_metadata,
)
from sonic_o1_agent.core.multimodal_utils import (
    SAMPLE_RATE,
    ceil_by_factor,
    floor_by_factor,
    get_index,
    round_by_factor,
    smart_nframes,
    smart_resize,
)
from sonic_o1_agent.core.video_processor import (
    VideoMetadata,
    fetch_video_pyav,
    get_video_metadata,
    process_video_with_metadata,
)

# Re-export for backward compatibility
__all__ = [
    # Constants
    "SAMPLE_RATE",
    # Math helpers
    "round_by_factor",
    "ceil_by_factor",
    "floor_by_factor",
    "smart_resize",
    "smart_nframes",
    "get_index",
    # Video functions
    "VideoMetadata",
    "get_video_metadata",
    "fetch_video_pyav",
    "process_video_with_metadata",
    # Audio functions
    "load_audio_pyav",
    "process_audio_with_metadata",
    # Orchestration functions
    "process_audio_info_pyav",
    "process_vision_info_pyav",
    "process_mm_info",
]


def process_audio_info_pyav(
    conversations,
    use_audio_in_video: bool,
    max_audio_duration: Optional[float] = None,
    max_audio_chunks: Optional[int] = None,
    audio_chunk_duration_sec: float = 10.0,
) -> Optional[List[np.ndarray]]:
    """Process audio from conversation structure."""
    audios = []

    if isinstance(conversations[0], dict):
        conversations = [conversations]

    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue

            for ele in message["content"]:
                if ele["type"] != "audio":
                    continue

                if "audio" not in ele and "audio_url" not in ele:
                    continue

                path = ele.get("audio", ele.get("audio_url"))
                audio_start: float = ele.get("audio_start", 0.0)
                audio_end: Optional[float] = ele.get("audio_end", None)

                if isinstance(path, np.ndarray):
                    if path.ndim > 1:
                        raise ValueError("Only mono audio supported")

                    start_idx = int(SAMPLE_RATE * audio_start)
                    end_idx = (
                        None if audio_end is None else int(SAMPLE_RATE * audio_end)
                    )
                    audio = path[start_idx:end_idx]

                    if max_audio_chunks is not None:
                        samples_per_chunk = int(audio_chunk_duration_sec * SAMPLE_RATE)
                        total_samples = len(audio)
                        num_chunks = int(np.ceil(total_samples / samples_per_chunk))

                        if num_chunks > max_audio_chunks:
                            chunks = []
                            for i in range(num_chunks):
                                start_idx = i * samples_per_chunk
                                end_idx = min(
                                    (i + 1) * samples_per_chunk, total_samples
                                )
                                chunks.append(audio[start_idx:end_idx])

                            sample_indices = np.linspace(
                                0, num_chunks - 1, max_audio_chunks, dtype=int
                            )
                            sampled_chunks = [chunks[i] for i in sample_indices]
                            audio = np.concatenate(sampled_chunks, axis=0)
                    elif max_audio_duration is not None:
                        max_samples = int(SAMPLE_RATE * max_audio_duration)
                        if len(audio) > max_samples:
                            audio = audio[:max_samples]

                    audios.append(audio)
                    continue

                if path.startswith("file://"):
                    path = path[7:]

                duration: Optional[float] = (
                    None if audio_end is None else (audio_end - audio_start)
                )

                audio = load_audio_pyav(
                    path,
                    sr=SAMPLE_RATE,
                    offset=audio_start,
                    duration=duration,
                    max_duration=max_audio_duration,
                    max_chunks=max_audio_chunks,
                    chunk_duration_sec=audio_chunk_duration_sec,
                )
                audios.append(audio)

    return audios if audios else None


def process_vision_info_pyav(
    conversations,
) -> Tuple[Optional[List[Image.Image]], Optional[List[torch.Tensor]]]:
    """Process images and videos from conversation structure."""
    videos = []
    images = []

    if isinstance(conversations[0], dict):
        conversations = [conversations]

    for conversation in conversations:
        for message in conversation:
            if not isinstance(message["content"], list):
                continue

            for ele in message["content"]:
                if ele["type"] == "video" and ("video" in ele or "video_url" in ele):
                    videos.append(fetch_video_pyav(ele))
                elif ele["type"] == "image" and "image" in ele:
                    img_path = ele["image"]
                    if img_path.startswith("file://"):
                        img_path = img_path[7:]
                    images.append(Image.open(img_path).convert("RGB"))

    return (images if images else None, videos if videos else None)


def process_mm_info(
    conversations,
    use_audio_in_video: bool = False,
    max_audio_duration: Optional[float] = None,
    max_audio_chunks: Optional[int] = None,
    audio_chunk_duration_sec: float = 10.0,
) -> tuple:
    """Process multimodal info from conversations.

    Returns:
        Tuple of (audios, images, videos, metadata)
        metadata includes frame indices, timestamps, etc.
    """
    audios: List[Any] = []
    images: List[Any] = []
    videos: List[Any] = []
    metadata: Dict[str, List[Any]] = {
        "video_metadata": [],
        "audio_metadata": [],
    }

    # Process conversation content FIRST
    for message in conversations:
        if isinstance(message, dict) and "content" in message:
            for item in message["content"]:
                if isinstance(item, dict):
                    if item["type"] == "video":
                        video_path = item["video"]

                        # Process video and GET metadata
                        video_frames, video_meta = process_video_with_metadata(
                            video_path,
                            max_frames=item.get("max_frames", 256),
                            min_frames=item.get("min_frames", 64),
                            video_start=item.get("video_start"),
                            video_end=item.get("video_end"),
                        )

                        videos.append(video_frames)
                        metadata["video_metadata"].append(video_meta)

                    elif item["type"] == "audio":
                        audio_path = item["audio"]

                        # Process audio and GET metadata
                        audio_chunks, audio_meta = process_audio_with_metadata(
                            audio_path,
                            max_audio_duration,
                            max_audio_chunks,
                            audio_chunk_duration_sec,
                            audio_start=item.get("audio_start"),
                            audio_end=item.get("audio_end"),
                        )

                        audios.append(audio_chunks)
                        metadata["audio_metadata"].append(audio_meta)

                    elif item["type"] == "image":
                        # Handle images if needed
                        pass

    return audios, images, videos, metadata
