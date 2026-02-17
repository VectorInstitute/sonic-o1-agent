"""Advanced multi-step planning and task decomposition.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class MultiStepPlanner:
    """Decomposes complex queries into sequential sub-tasks."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize multi-step planner.

        Args:
            config: Planner configuration
        """
        self.config = config or {}
        self.max_steps = self.config.get("max_steps", 10)

    def should_decompose(self, query: str) -> bool:
        """Determine if query needs decomposition.

        Args:
            query: User query

        Returns:
            True if query should be decomposed
        """
        # Keywords that indicate complex queries
        complex_keywords = [
            "compare",
            "contrast",
            "difference between",
            "both",
            "each",
            "all",
            "versus",
            "vs",
            "analyze",
            "relationship between",
            "how does",
            "explain why",
        ]

        query_lower = query.lower()
        return any(keyword in query_lower for keyword in complex_keywords)

    def decompose_query(self, query: str) -> List[Dict[str, Any]]:
        """Decompose query into sequential steps.

        Args:
            query: Complex user query

        Returns:
            List of step specifications
        """
        query_lower = query.lower()
        steps = []

        # Pattern 1: Comparison queries
        if any(
            kw in query_lower
            for kw in ["compare", "contrast", "difference", "versus", "vs"]
        ):
            steps = self._decompose_comparison(query)

        # Pattern 2: Sequential analysis
        elif any(kw in query_lower for kw in ["first", "then", "after", "next"]):
            steps = self._decompose_sequential(query)

        # Pattern 3: Multi-entity analysis
        elif any(kw in query_lower for kw in ["both", "each", "all"]):
            steps = self._decompose_multi_entity(query)

        # Pattern 4: Causal/explanatory
        elif any(
            kw in query_lower for kw in ["why", "how does", "what causes", "explain"]
        ):
            steps = self._decompose_causal(query)

        # Default: Single comprehensive step
        else:
            steps = [
                {
                    "step_id": 1,
                    "action": "analyze",
                    "description": "Comprehensive analysis",
                    "query": query,
                }
            ]

        return steps[: self.max_steps]  # Limit number of steps

    def _decompose_comparison(self, query: str) -> List[Dict[str, Any]]:
        """Decompose comparison query."""
        # Extract entities being compared (simplified)
        # In production, use NER or LLM to extract entities

        steps = [
            {
                "step_id": 1,
                "action": "identify_entity_1",
                "description": "Identify first entity/person/topic",
                "query": f"Identify the first subject in: {query}",
            },
            {
                "step_id": 2,
                "action": "analyze_entity_1",
                "description": "Analyze first entity",
                "query": "Analyze what was said/done by the first subject",
                "depends_on": [1],
            },
            {
                "step_id": 3,
                "action": "identify_entity_2",
                "description": "Identify second entity/person/topic",
                "query": f"Identify the second subject in: {query}",
            },
            {
                "step_id": 4,
                "action": "analyze_entity_2",
                "description": "Analyze second entity",
                "query": "Analyze what was said/done by the second subject",
                "depends_on": [3],
            },
            {
                "step_id": 5,
                "action": "compare",
                "description": "Compare and contrast both entities",
                "query": query,
                "depends_on": [2, 4],
            },
        ]

        return steps

    def _decompose_sequential(self, query: str) -> List[Dict[str, Any]]:
        """Decompose sequential analysis query."""
        steps = [
            {
                "step_id": 1,
                "action": "identify_sequence",
                "description": "Identify sequence of events",
                "query": "What is the sequence of events?",
            },
            {
                "step_id": 2,
                "action": "analyze_each",
                "description": "Analyze each event",
                "query": query,
                "depends_on": [1],
            },
            {
                "step_id": 3,
                "action": "synthesize",
                "description": "Synthesize overall answer",
                "query": query,
                "depends_on": [2],
            },
        ]

        return steps

    def _decompose_multi_entity(self, query: str) -> List[Dict[str, Any]]:
        """Decompose multi-entity query."""
        steps = [
            {
                "step_id": 1,
                "action": "identify_all",
                "description": "Identify all entities",
                "query": "Identify all relevant subjects/entities",
            },
            {
                "step_id": 2,
                "action": "analyze_each",
                "description": "Analyze each entity separately",
                "query": query,
                "depends_on": [1],
            },
            {
                "step_id": 3,
                "action": "aggregate",
                "description": "Aggregate results",
                "query": query,
                "depends_on": [2],
            },
        ]

        return steps

    def _decompose_causal(self, query: str) -> List[Dict[str, Any]]:
        """Decompose causal/explanatory query."""
        steps = [
            {
                "step_id": 1,
                "action": "observe",
                "description": "Observe what happened",
                "query": "What events/actions occurred?",
            },
            {
                "step_id": 2,
                "action": "identify_causes",
                "description": "Identify potential causes",
                "query": "What could have caused this?",
                "depends_on": [1],
            },
            {
                "step_id": 3,
                "action": "explain",
                "description": "Explain causal relationships",
                "query": query,
                "depends_on": [1, 2],
            },
        ]

        return steps

    def execute_plan(
        self,
        steps: List[Dict[str, Any]],
        executor_fn: Any,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute multi-step plan.

        Args:
            steps: List of step specifications
            executor_fn: Function to execute each step
            video_path: Path to video
            audio_path: Path to audio

        Returns:
            Dict with results from all steps
        """
        results: dict[str, Any] = {}
        step_outputs: dict[int, Any] = {}

        logger.info(f"Executing {len(steps)}-step plan")

        for step in steps:
            step_id = step["step_id"]
            logger.info(f"Step {step_id}/{len(steps)}: {step['description']}")

            # Build context from dependent steps
            context = {}
            if "depends_on" in step:
                for dep_id in step["depends_on"]:
                    if dep_id in step_outputs:
                        context[f"step_{dep_id}"] = step_outputs[dep_id]

            # Execute step
            try:
                result = executor_fn(
                    query=step["query"],
                    video_path=video_path,
                    audio_path=audio_path,
                    context=context,
                )

                step_outputs[step_id] = result
                results[f"step_{step_id}"] = {
                    "action": step["action"],
                    "description": step["description"],
                    "result": result,
                }

            except Exception as e:
                logger.error(f"Step {step_id} failed: {e}")
                results[f"step_{step_id}"] = {
                    "action": step["action"],
                    "description": step["description"],
                    "error": str(e),
                }

        # Final synthesis
        final_result = step_outputs.get(len(steps), "")

        return {
            "final_answer": final_result,
            "steps_executed": results,
            "total_steps": len(steps),
        }
