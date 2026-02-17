"""Audio processing using PyAV.

Memory-efficient audio loading with chunking and resampling.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

import time
from typing import Optional

import av
import numpy as np

from sonic_o1_agent.core.multimodal_utils import SAMPLE_RATE


# ============================================================================
# Audio Processing Functions
# ============================================================================
def load_audio_pyav(
    audio_path: str,
    sr: int = SAMPLE_RATE,
    offset: float = 0.0,
    duration: Optional[float] = None,
    max_duration: Optional[float] = None,
    max_chunks: Optional[int] = None,
    chunk_duration_sec: float = 10.0,
) -> np.ndarray:
    """Memory-efficient audio loading using PyAV with optional chunking."""
    st = time.time()

    container = av.open(audio_path)

    if len(container.streams.audio) == 0:
        container.close()
        print(f"  Audio: No audio stream found in {audio_path}, returning empty array")
        return np.array([], dtype=np.float32)

    audio_stream = container.streams.audio[0]
    original_sr = audio_stream.sample_rate

    start_sample = int(offset * original_sr)
    end_sample = int((offset + duration) * original_sr) if duration else None

    est_samples = int((duration or 60) * original_sr)
    audio_buffer = np.zeros(est_samples, dtype=np.float32)
    write_pos = 0

    current_sample = 0
    for frame in container.decode(audio=0):
        frame_data = frame.to_ndarray()

        if frame_data.ndim > 1:
            frame_data = frame_data.mean(axis=0, dtype=np.float32)
        else:
            frame_data = frame_data.astype(np.float32)

        frame_samples = len(frame_data)

        if current_sample + frame_samples < start_sample:
            current_sample += frame_samples
            continue

        if end_sample and current_sample >= end_sample:
            break

        frame_start = max(0, start_sample - current_sample)
        frame_end = (
            min(frame_samples, end_sample - current_sample)
            if end_sample
            else frame_samples
        )
        chunk = frame_data[frame_start:frame_end]

        chunk_len = len(chunk)
        if write_pos + chunk_len > len(audio_buffer):
            audio_buffer = np.resize(audio_buffer, write_pos + chunk_len + est_samples)

        audio_buffer[write_pos : write_pos + chunk_len] = chunk
        write_pos += chunk_len
        current_sample += frame_samples

    container.close()

    audio = audio_buffer[:write_pos]

    if len(audio) == 0:
        print(
            f"  Audio: No audio data decoded from {audio_path}, returning empty array"
        )
        return np.array([], dtype=np.float32)

    # Resample if needed
    if original_sr != sr:
        ratio = sr / original_sr
        new_length = int(len(audio) * ratio)

        if 0.9 < ratio < 1.1:
            indices = (np.arange(new_length) / ratio).astype(np.int32)
            indices = np.clip(indices, 0, len(audio) - 1)
            audio = audio[indices]
        else:
            indices = np.linspace(0, len(audio) - 1, new_length)
            audio = np.interp(indices, np.arange(len(audio)), audio)

    # Apply chunking if requested
    if max_chunks is not None:
        samples_per_chunk = int(chunk_duration_sec * sr)
        total_samples = len(audio)
        num_chunks = int(np.ceil(total_samples / samples_per_chunk))

        if num_chunks > max_chunks:
            print(
                f"  Audio chunking: {total_samples} samples -> {num_chunks} chunks, sampling {max_chunks}"
            )

            chunks = []
            for i in range(num_chunks):
                start_idx = i * samples_per_chunk
                end_idx = min((i + 1) * samples_per_chunk, total_samples)
                chunks.append(audio[start_idx:end_idx])

            sample_indices = np.linspace(0, num_chunks - 1, max_chunks, dtype=int)
            sampled_chunks = [chunks[i] for i in sample_indices]
            audio = np.concatenate(sampled_chunks, axis=0)

            print(f"  Final audio: {len(audio)} samples (from {total_samples})")

    elif max_duration is not None:
        max_samples = int(max_duration * sr)
        if len(audio) > max_samples:
            audio = audio[:max_samples]

    actual_duration = len(audio) / sr

    was_chunked = False
    was_truncated = False
    if max_chunks is not None:
        samples_per_chunk = int(chunk_duration_sec * sr)
        num_chunks = int(np.ceil(len(audio_buffer[:write_pos]) / samples_per_chunk))
        was_chunked = num_chunks > max_chunks
    elif max_duration is not None:
        was_truncated = actual_duration >= max_duration * 0.99

    status = " (chunked)" if was_chunked else (" (truncated)" if was_truncated else "")
    print(f"  Audio: {actual_duration:.2f}s{status} in {time.time() - st:.3f}s")

    return audio


def process_audio_with_metadata(
    audio_path: str,
    max_audio_duration: Optional[float] = None,
    max_audio_chunks: Optional[int] = None,
    audio_chunk_duration_sec: float = 10.0,
    audio_start: Optional[float] = None,
    audio_end: Optional[float] = None,
) -> tuple:
    """Process audio and return audio + compact metadata summary.

    Supports optional ``audio_start`` / ``audio_end`` (seconds) to
    restrict loading to a time range within the audio file.  When
    omitted the full audio is loaded.

    Metadata includes only aggregate stats (duration, chunk count,
    chunk size, coverage range) -- no per-chunk sample indices.
    """
    offset = audio_start if audio_start and audio_start > 0 else 0.0
    duration = (audio_end - offset) if audio_end is not None else None

    audio = load_audio_pyav(
        audio_path,
        sr=SAMPLE_RATE,
        offset=offset,
        duration=duration,
        max_duration=max_audio_duration,
        max_chunks=max_audio_chunks,
        chunk_duration_sec=audio_chunk_duration_sec,
    )

    total_duration = round(len(audio) / SAMPLE_RATE, 2)
    chunk_size = int(audio_chunk_duration_sec * SAMPLE_RATE)
    num_chunks = int(np.ceil(len(audio) / chunk_size)) if len(audio) > 0 else 0

    metadata = {
        "duration_sec": total_duration,
        "sample_rate": SAMPLE_RATE,
        "chunks_analyzed": num_chunks,
        "chunk_duration_sec": audio_chunk_duration_sec,
        "coverage_sec": [round(offset, 2), round(offset + total_duration, 2)],
    }

    return audio, metadata
