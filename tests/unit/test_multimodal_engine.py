"""Unit tests for multimodal processing engine.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

import numpy as np

from sonic_o1_agent.core.multimodal_engine import (
    ceil_by_factor,
    floor_by_factor,
    round_by_factor,
    smart_nframes,
    smart_resize,
)


class TestMathHelpers:
    """Test mathematical helper functions."""

    def test_round_by_factor(self):
        """Test rounding to nearest factor (Python round: half to even)."""
        # 25/10 = 2.5 -> round(2.5) = 2 in Python 3 (banker's rounding)
        assert round_by_factor(25, 10) == 20
        assert round_by_factor(24, 10) == 20
        assert round_by_factor(26, 10) == 30
        assert round_by_factor(28, 28) == 28
        assert round_by_factor(30, 28) == 28

    def test_ceil_by_factor(self):
        """Test ceiling division by factor."""
        assert ceil_by_factor(25, 10) == 30
        assert ceil_by_factor(20, 10) == 20
        assert ceil_by_factor(21, 10) == 30
        assert ceil_by_factor(1, 28) == 28

    def test_floor_by_factor(self):
        """Test floor division by factor."""
        assert floor_by_factor(25, 10) == 20
        assert floor_by_factor(29, 10) == 20
        assert floor_by_factor(30, 10) == 30
        assert floor_by_factor(27, 28) == 0


class TestSmartResize:
    """Test smart resize for images/videos."""

    def test_resize_maintains_aspect_ratio(self):
        """Test that resize maintains aspect ratio."""
        h, w = smart_resize(1080, 1920, factor=28)
        aspect_original = 1920 / 1080
        aspect_resized = w / h
        assert abs(aspect_original - aspect_resized) < 0.1

    def test_resize_divisible_by_factor(self):
        """Test output is divisible by factor."""
        h, w = smart_resize(1080, 1920, factor=28)
        assert h % 28 == 0
        assert w % 28 == 0

    def test_resize_respects_min_pixels(self):
        """Test minimum pixel constraint."""
        min_pixels = 4 * 28 * 28
        h, w = smart_resize(100, 100, factor=28, min_pixels=min_pixels)
        assert h * w >= min_pixels

    def test_resize_respects_max_pixels(self):
        """Test maximum pixel constraint."""
        max_pixels = 1000 * 28 * 28
        h, w = smart_resize(4000, 4000, factor=28, max_pixels=max_pixels)
        assert h * w <= max_pixels

    def test_resize_caching(self):
        """Test that results are cached."""
        # Call twice with same params
        result1 = smart_resize(1080, 1920, factor=28)
        result2 = smart_resize(1080, 1920, factor=28)
        assert result1 == result2
        assert result1 is result2  # Same object (cached)


class TestSmartNFrames:
    """Test smart frame count calculation."""

    def test_nframes_with_explicit_value(self):
        """Test when nframes is explicitly provided."""
        ele = {"nframes": 100}
        result = smart_nframes(ele, total_frames=1000, video_fps=30.0)
        assert result == 100

    def test_nframes_respects_min_frames(self):
        """Test minimum frames constraint."""
        ele = {"fps": 0.1, "min_frames": 64}
        result = smart_nframes(ele, total_frames=1000, video_fps=30.0)
        assert result >= 64

    def test_nframes_respects_max_frames(self):
        """Test maximum frames constraint."""
        ele = {"fps": 10.0, "max_frames": 128}
        result = smart_nframes(ele, total_frames=1000, video_fps=30.0)
        assert result <= 128

    def test_nframes_divisible_by_frame_factor(self):
        """Test output divisible by FRAME_FACTOR (2)."""
        ele = {"fps": 1.0}
        result = smart_nframes(ele, total_frames=1000, video_fps=30.0)
        assert result % 2 == 0

    def test_nframes_not_exceed_total(self):
        """Test doesn't exceed total available frames."""
        ele = {"fps": 10.0}
        result = smart_nframes(ele, total_frames=100, video_fps=30.0)
        assert result <= 100

    def test_nframes_fps_calculation(self):
        """Test FPS-based sampling calculation."""
        ele = {"fps": 1.0}  # 1 frame per second
        # 1000 frames at 30fps = 33.33 seconds
        # At 1fps sampling = ~33 frames
        result = smart_nframes(ele, total_frames=1000, video_fps=30.0)
        expected = int(1000 * 1.0 / 30.0)
        # Should be close (accounting for factor rounding)
        assert abs(result - expected) < 10


class TestAudioProcessing:
    """Test audio-related utilities."""

    def test_sample_rate_constant(self):
        """Test SAMPLE_RATE is defined correctly."""
        from sonic_o1_agent.core.multimodal_engine import SAMPLE_RATE

        assert SAMPLE_RATE == 16000

    def test_audio_chunking_logic(self):
        """Test audio chunking calculation."""
        # Simulate 60 seconds of audio at 16kHz
        total_samples = 60 * 16000
        chunk_duration = 10.0  # 10 second chunks
        max_chunks = 3

        samples_per_chunk = int(chunk_duration * 16000)
        num_chunks = int(np.ceil(total_samples / samples_per_chunk))

        assert num_chunks == 6  # 60s / 10s = 6 chunks

        # Should trigger sampling since 6 > 3
        assert num_chunks > max_chunks
