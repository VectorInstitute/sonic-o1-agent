"""Video processing using PyAV.

Memory-efficient video frame sampling and processing.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, cast

import av
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from sonic_o1_agent.core.multimodal_utils import (
    FRAME_FACTOR,
    IMAGE_FACTOR,
    VIDEO_MAX_PIXELS,
    VIDEO_MIN_PIXELS,
    VIDEO_TOTAL_PIXELS,
    get_index,
    smart_nframes,
    smart_resize,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Video Metadata
# ============================================================================
@dataclass
class VideoMetadata:
    """Cached video metadata to avoid repeated file opens."""

    total_frames: int
    fps: float
    width: int
    height: int


_video_metadata_cache: Dict[str, VideoMetadata] = {}


def get_video_metadata(video_path: str) -> VideoMetadata:
    """Extract and cache video metadata."""
    if video_path in _video_metadata_cache:
        return _video_metadata_cache[video_path]

    container = av.open(video_path)
    video_stream = container.streams.video[0]

    metadata = VideoMetadata(
        total_frames=video_stream.frames or sum(1 for _ in container.decode(video=0)),
        fps=float(video_stream.average_rate),
        width=video_stream.width,
        height=video_stream.height,
    )

    container.close()
    _video_metadata_cache[video_path] = metadata
    return metadata


# ============================================================================
# Video Processing Functions
# ============================================================================
def fetch_video_pyav(ele: dict, image_factor: int = IMAGE_FACTOR) -> torch.Tensor:
    """Memory-efficient video loading using PyAV."""
    video_path = ele["video"]
    st = time.time()

    try:
        metadata = get_video_metadata(video_path)
        total_frames = metadata.total_frames
        video_fps = metadata.fps
        height = metadata.height
        width = metadata.width
    except Exception:
        container = av.open(video_path)
        video_stream = container.streams.video[0]
        total_frames = video_stream.frames or sum(1 for _ in container.decode(video=0))
        video_fps = float(video_stream.average_rate)
        height = video_stream.height
        width = video_stream.width
        container.close()

    video_start = ele.get("video_start", 0.0)
    video_end = ele.get("video_end")

    start_frame = max(0, int(video_start * video_fps)) if video_start > 0 else 0
    if video_end is not None:
        end_frame = min(int(video_end * video_fps), total_frames - 1)
        total_frames = end_frame - start_frame + 1
    else:
        end_frame = total_frames - 1

    nframes = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)

    if nframes == total_frames:
        indices = list(range(start_frame, end_frame + 1))
    else:
        indices = np.linspace(start_frame, end_frame, nframes, dtype=np.int32).tolist()

    indices_set = set(indices)

    container = av.open(
        video_path,
        options={
            "threads": "auto",
            "thread_queue_size": "512",
        },
    )
    video_stream = container.streams.video[0]
    video_stream.thread_type = "AUTO"

    min_pixels = ele.get("min_pixels", VIDEO_MIN_PIXELS)
    total_pixels = ele.get("total_pixels", VIDEO_TOTAL_PIXELS)
    max_pixels = max(
        min(VIDEO_MAX_PIXELS, total_pixels / nframes * FRAME_FACTOR),
        int(min_pixels * 1.05),
    )
    max_pixels = min(ele.get("max_pixels", max_pixels), max_pixels)

    if "resized_height" in ele and "resized_width" in ele:
        resized_height, resized_width = smart_resize(
            ele["resized_height"], ele["resized_width"], factor=image_factor
        )
    else:
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

    frames_array = np.empty((nframes, resized_height, resized_width, 3), dtype=np.uint8)

    if start_frame > 10:
        try:
            container.seek(
                int(start_frame / video_fps * av.time_base), stream=video_stream
            )
            frame_idx = start_frame
        except Exception:
            frame_idx = 0
    else:
        frame_idx = 0

    collected = 0
    resize_transform = transforms.Resize(
        (resized_height, resized_width),
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )

    for frame in container.decode(video=0):
        if frame_idx in indices_set:
            frame_np = frame.to_ndarray(format="rgb24")
            frame_tensor = torch.from_numpy(frame_np).permute(2, 0, 1)
            frame_resized = resize_transform(frame_tensor)
            frames_array[collected] = frame_resized.permute(1, 2, 0).numpy()

            collected += 1
            if collected >= nframes:
                break

        frame_idx += 1
        if frame_idx > end_frame:
            break

    container.close()

    video = torch.from_numpy(frames_array).permute(0, 3, 1, 2).float()

    logger.info(
        f"  Video: {nframes} frames ({height}x{width} -> {resized_height}x{resized_width}) in {time.time() - st:.3f}s"
    )
    return video


def process_video_with_metadata(
    video_path: str,
    max_frames: int = 256,
    min_frames: int = 64,
    video_start: Optional[float] = None,
    video_end: Optional[float] = None,
) -> tuple:
    """Process video and return frames + compact metadata summary.

    Supports optional ``video_start`` / ``video_end`` (seconds) to
    restrict processing to a time range within the video.  When omitted
    the full video is processed.

    Metadata includes only aggregate stats (duration, frame count,
    sampling interval, coverage range) -- no per-frame indices or
    timestamps, keeping the output concise and actionable.
    """
    container = av.open(video_path)
    video_stream = container.streams.video[0]

    # Get video properties
    fps = float(video_stream.average_rate)
    total_frames_full = video_stream.frames
    if total_frames_full == 0:
        full_duration = float(container.duration / av.time_base)
        total_frames_full = int(full_duration * fps)
    else:
        full_duration = total_frames_full / fps

    # Resolve time-range to frame range
    start_frame = 0
    end_frame = total_frames_full - 1
    if video_start is not None and video_start > 0:
        start_frame = max(0, int(video_start * fps))
    if video_end is not None:
        end_frame = min(int(video_end * fps), total_frames_full - 1)

    range_frames = max(end_frame - start_frame + 1, 1)
    duration_sec = range_frames / fps

    # Calculate number of frames to sample within the range
    num_frames = min(max_frames, range_frames)
    num_frames = max(min_frames, num_frames)

    # Calculate indices to sample (relative to start_frame)
    raw_indices = get_index(range_frames, num_frames)
    indices_set = set(idx + start_frame for idx in raw_indices)

    sampled_count = 0
    first_frame_sec = None
    last_frame_sec = None
    frames = []

    # Seek if the range doesn't start at the beginning.
    # After seek, set frame_idx to start_frame so the manual counter
    # tracks the real position (enumerate resets to 0 after seek).
    if start_frame > 10:
        try:
            container.seek(
                int(start_frame / fps * av.time_base),
                stream=video_stream,
            )
            frame_idx = start_frame
        except Exception:
            frame_idx = 0  # fall back to sequential decode
    else:
        frame_idx = 0

    for frame in container.decode(video=0):
        if frame_idx > end_frame:
            break
        if frame_idx in indices_set:
            timestamp_sec = frame_idx / fps

            if first_frame_sec is None:
                first_frame_sec = timestamp_sec
            last_frame_sec = timestamp_sec

            sampled_count += 1

            # Process frame
            img = frame.to_ndarray(format="rgb24")
            h, w = img.shape[:2]

            resized_h, resized_w = smart_resize(h, w)

            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((resized_w, resized_h), Image.BICUBIC)
            img_resized = np.array(pil_img)

            frames.append(img_resized)

            if len(frames) >= num_frames:
                break

        frame_idx += 1

    container.close()

    # Compact coverage summary -- seconds only
    sampling_interval = (
        round(
            (cast(float, last_frame_sec) - cast(float, first_frame_sec))
            / max(sampled_count - 1, 1),
            2,
        )
        if sampled_count > 1
        else 0.0
    )

    metadata = {
        "duration_sec": round(duration_sec, 2),
        "fps": round(fps, 2),
        "total_frames": range_frames,
        "frames_sampled": sampled_count,
        "sampling_interval_sec": sampling_interval,
        "coverage_sec": [
            round(first_frame_sec or 0.0, 2),
            round(last_frame_sec or 0.0, 2),
        ],
    }

    # Convert frames to tensor
    if frames:
        frames_tensor = torch.from_numpy(np.array(frames)).permute(0, 3, 1, 2).float()
    else:
        frames_tensor = torch.empty(0)

    return frames_tensor, metadata
