"""State definition for LangGraph workflow.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

from typing import Any, Dict, List, Optional, Tuple, TypedDict


class SonicState(TypedDict, total=False):
    """State that flows through the LangGraph workflow.

    All keys are optional (total=False). Nodes read and write partial
    updates; the graph merges them.
    """

    # Input
    query: str
    video_path: Optional[str]
    audio_path: Optional[str]
    max_frames: Optional[int]
    max_audio_chunks: Optional[int]

    # Processing flags
    use_reasoning: bool
    use_reflection: bool
    use_multi_step: bool

    # Planning outputs
    plan: Dict[str, Any]
    should_segment: bool
    time_range: Optional[Tuple[float, float]]
    query_type: str

    # Segmentation outputs
    actual_video_path: Optional[str]
    actual_audio_path: Optional[str]
    temp_files: List[str]

    # Temporal index (frame-captioned segment descriptions)
    temporal_index: Optional[str]

    # Reasoning outputs
    reasoning_chain: Optional[List[Dict]]
    reasoning_confidence: Optional[float]
    reasoning_trace: Optional[str]

    # Reflection outputs
    reflection: Optional[Dict]
    was_refined: bool
    original_response: Optional[str]
    refinement_history: Optional[List[Dict]]
    hallucination_assessment: Optional[Dict]

    # Multi-step outputs
    multi_step_plan: Optional[List[Dict]]
    steps_executed: Optional[Dict]

    # Final output
    response: str
    reasoning_mode: str
    modalities_used: Dict[str, bool]
    evidence: Optional[Dict[str, Any]]

    # Metadata
    error: Optional[str]
