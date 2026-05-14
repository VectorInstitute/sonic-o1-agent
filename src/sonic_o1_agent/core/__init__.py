"""Core multimodal processing engines.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

# Import from multimodal_engine for backward compatibility
# All exports are re-exported from the split modules
from sonic_o1_agent.core.multimodal_engine import (
    SAMPLE_RATE,
    VideoMetadata,
    fetch_video_pyav,
    get_video_metadata,
    load_audio_pyav,
    process_mm_info,
)

__all__ = [
    "load_audio_pyav",
    "fetch_video_pyav",
    "get_video_metadata",
    "process_mm_info",
    "VideoMetadata",
    "SAMPLE_RATE",
]
