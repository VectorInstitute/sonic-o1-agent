"""Shared utilities for multimodal processing.

Math helpers, constants, and common utilities used by video and audio processors.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

import math
import os
from functools import lru_cache
from typing import List, Tuple

# ============================================================================
# Constants
# ============================================================================
IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28
VIDEO_MIN_PIXELS = 128 * 28 * 28
VIDEO_MAX_PIXELS = 768 * 28 * 28
VIDEO_TOTAL_PIXELS = int(
    float(os.environ.get("VIDEO_MAX_PIXELS", 128000 * 28 * 28 * 0.9))
)
FRAME_FACTOR = 2
FPS = 1.0
FPS_MIN_FRAMES = 4
FPS_MAX_FRAMES = 768
SAMPLE_RATE = 16000


# ============================================================================
# Optimized Math Helpers
# ============================================================================
@lru_cache(maxsize=1024)
def round_by_factor(number: int, factor: int) -> int:
    """Return the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


@lru_cache(maxsize=1024)
def ceil_by_factor(number: int, factor: int) -> int:
    """Return the smallest integer >= 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


@lru_cache(maxsize=1024)
def floor_by_factor(number: int, factor: int) -> int:
    """Return the largest integer <= 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor


@lru_cache(maxsize=2048)
def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> Tuple[int, int]:
    """Calculate optimal resize dimensions maintaining aspect ratio."""
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, floor_by_factor(int(height / beta), factor))
        w_bar = max(factor, floor_by_factor(int(width / beta), factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(int(height * beta), factor)
        w_bar = ceil_by_factor(int(width * beta), factor)

    return h_bar, w_bar


def smart_nframes(ele: dict, total_frames: int, video_fps: float) -> int:
    """Calculate optimal number of frames to sample from video."""
    if "nframes" in ele:
        return round_by_factor(ele["nframes"], FRAME_FACTOR)

    fps = ele.get("fps", FPS)
    min_frames = ceil_by_factor(ele.get("min_frames", FPS_MIN_FRAMES), FRAME_FACTOR)
    max_frames = floor_by_factor(
        ele.get("max_frames", min(FPS_MAX_FRAMES, total_frames)), FRAME_FACTOR
    )

    nframes = int(total_frames * fps / video_fps)
    nframes = max(min_frames, min(nframes, max_frames, total_frames))
    nframes = floor_by_factor(nframes, FRAME_FACTOR)

    if not (FRAME_FACTOR <= nframes <= total_frames):
        raise ValueError(
            f"nframes should be in [{FRAME_FACTOR}, {total_frames}], got {nframes}"
        )

    return nframes


def get_index(total_frames: int, num_frames: int) -> List[int]:
    """Calculate frame indices to sample uniformly.

    Args:
        total_frames: Total number of frames in video
        num_frames: Number of frames to sample

    Returns:
        List of frame indices to sample
    """
    import numpy as np

    if num_frames >= total_frames:
        return list(range(total_frames))

    # Uniform sampling
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int32)
    return [int(x) for x in indices]
