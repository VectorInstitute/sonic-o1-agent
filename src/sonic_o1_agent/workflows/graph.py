"""LangGraph workflow definition.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

from langgraph.graph import END, StateGraph

from sonic_o1_agent.workflows.nodes import SonicNodes
from sonic_o1_agent.workflows.state import SonicState


def should_use_reasoning(state: SonicState) -> str:
    """Return whether to use CoT reasoning (used when not multi-step).

    Args:
        state: Current workflow state.

    Returns:
        "reasoning" if use_reasoning is True, else "direct".
    """
    if state.get("use_reasoning", False):
        return "reasoning"
    return "direct"


def should_route_inference(state: SonicState) -> str:
    """Route to multi-step, reasoning, or direct inference node.

    Args:
        state: Current workflow state.

    Returns:
        "multi_step", "reasoning", or "direct".

    Raises:
        ValueError: If use_multi_step is True but multi_step_plan is missing or empty
            (avoids masking planning failures by silently falling back to reasoning/direct).
    """
    use_multi = state.get("use_multi_step")
    plan = state.get("multi_step_plan")
    if use_multi and (not plan or (isinstance(plan, list) and len(plan) == 0)):
        raise ValueError(
            "use_multi_step is True but multi_step_plan is missing or empty; "
            "planning may have failed."
        )
    if use_multi and plan:
        return "multi_step"
    return should_use_reasoning(state)


def should_use_reflection(state: SonicState) -> str:
    """Return whether to run reflection or go to cleanup.

    Args:
        state: Current workflow state.

    Returns:
        "reflection" if use_reflection is True, else "cleanup".
    """
    if state.get("use_reflection", False):
        return "reflection"
    return "cleanup"


def build_sonic_workflow(config: dict) -> StateGraph:
    """Build the Sonic O1 workflow graph.

    Args:
        config: Agent configuration dict.

    Returns:
        StateGraph (uncompiled) with planning, segmentation, temporal
        indexing, inference branches, reflection, and cleanup nodes.
    """
    nodes = SonicNodes(config)

    # Create graph
    workflow = StateGraph(SonicState)

    # Add nodes
    workflow.add_node("planning", nodes.planning_node)
    workflow.add_node("segmentation", nodes.segmentation_node)
    workflow.add_node("temporal_indexing", nodes.temporal_indexing_node)
    workflow.add_node("multi_step", nodes.multi_step_node)
    workflow.add_node("reasoning", nodes.reasoning_node)
    workflow.add_node("direct", nodes.direct_inference_node)
    workflow.add_node("reflection", nodes.reflection_node)
    workflow.add_node("cleanup", nodes.cleanup_node)

    # Build workflow
    workflow.set_entry_point("planning")

    # Planning → Segmentation (always)
    workflow.add_edge("planning", "segmentation")

    # Segmentation → Temporal Indexing (always)
    workflow.add_edge("segmentation", "temporal_indexing")

    # Temporal Indexing → Multi-step OR Reasoning OR Direct (conditional)
    workflow.add_conditional_edges(
        "temporal_indexing",
        should_route_inference,
        {
            "multi_step": "multi_step",
            "reasoning": "reasoning",
            "direct": "direct",
        },
    )

    # Multi-step → Reflection OR Cleanup (conditional)
    workflow.add_conditional_edges(
        "multi_step",
        should_use_reflection,
        {
            "reflection": "reflection",
            "cleanup": "cleanup",
        },
    )

    # Reasoning → Reflection OR Cleanup (conditional)
    workflow.add_conditional_edges(
        "reasoning",
        should_use_reflection,
        {
            "reflection": "reflection",
            "cleanup": "cleanup",
        },
    )

    # Direct → Reflection OR Cleanup (conditional)
    workflow.add_conditional_edges(
        "direct",
        should_use_reflection,
        {
            "reflection": "reflection",
            "cleanup": "cleanup",
        },
    )

    # Reflection → Cleanup (always)
    workflow.add_edge("reflection", "cleanup")

    # Cleanup → END
    workflow.add_edge("cleanup", END)

    return workflow
