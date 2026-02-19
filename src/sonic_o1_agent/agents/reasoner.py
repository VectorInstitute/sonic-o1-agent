"""Chain-of-Thought reasoning module.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class ChainOfThoughtReasoner:
    """Implements multi-step reasoning with explicit thought chains."""

    def __init__(self, model, config: Optional[Dict] = None):
        """Initialize the reasoner.

        Args:
            model: The underlying language model (Qwen3OmniModel)
            config: Reasoning configuration
        """
        self.model = model
        self.config = config or {}
        self.max_reasoning_steps = self.config.get("max_reasoning_steps", 5)
        self.enable_verification = self.config.get("enable_verification", True)

    def reason(
        self,
        query: str,
        video_path: Optional[str],
        audio_path: Optional[str],
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Execute chain-of-thought reasoning.

        Args:
            query: User's query
            video_path: Path to video file
            audio_path: Path to audio file
            context: Additional context

        Returns:
            Dict with reasoning chain and final answer
        """
        context = context or {}
        reasoning_chain = []

        # Step 1: Understand the query
        logger.info("CoT Step 1: Understanding query intent")
        understanding = self._understand_query(query, context)
        reasoning_chain.append(
            {
                "step": 1,
                "action": "understand_query",
                "thought": understanding["thought"],
                "result": understanding["intent"],
            }
        )

        # Step 2: Plan the approach
        logger.info("CoT Step 2: Planning analysis approach")
        plan = self._plan_approach(query, understanding, context)
        reasoning_chain.append(
            {
                "step": 2,
                "action": "plan_approach",
                "thought": plan["thought"],
                "result": plan["strategy"],
            }
        )

        # Step 3: Execute analysis
        logger.info("CoT Step 3: Executing analysis")
        analysis = self._execute_analysis(query, video_path, audio_path, plan, context)
        reasoning_chain.append(
            {
                "step": 3,
                "action": "execute_analysis",
                "thought": analysis["thought"],
                "result": analysis["findings"],
            }
        )

        # Step 4: Verify and refine (if enabled)
        if self.enable_verification:
            logger.info("CoT Step 4: Verifying results")
            verification = self._verify_result(query, analysis, context)
            reasoning_chain.append(
                {
                    "step": 4,
                    "action": "verify_result",
                    "thought": verification["thought"],
                    "result": verification["assessment"],
                }
            )

            # Step 5: Refine if needed
            if verification["needs_refinement"]:
                logger.info("CoT Step 5: Refining answer")
                refinement = self._refine_answer(query, analysis, verification, context)
                reasoning_chain.append(
                    {
                        "step": 5,
                        "action": "refine_answer",
                        "thought": refinement["thought"],
                        "result": refinement["improved_answer"],
                    }
                )
                final_answer = refinement["improved_answer"]
            else:
                final_answer = analysis["findings"]
        else:
            final_answer = analysis["findings"]

        return {
            "final_answer": final_answer,
            "reasoning_chain": reasoning_chain,
            "confidence": self._estimate_confidence(reasoning_chain),
        }

    def _understand_query(self, query: str, context: Dict) -> Dict[str, Any]:
        """Step 1: Understand what the user is asking."""
        understanding_prompt = f"""Analyze this query and explain your understanding:

Query: {query}

Think step by step:
1. What is the user asking for?
2. What type of information do they need?
3. What modalities (video/audio) are most important?
4. Is this a simple or complex question?

Provide your analysis."""

        response_text, _ = self.model.generate(
            video_path=None,
            audio_path=None,
            prompt=understanding_prompt,
            max_frames=0,  # No video needed for understanding
        )

        return {
            "thought": response_text,
            "intent": self._extract_intent(response_text),
        }

    def _extract_intent(self, understanding: str) -> str:
        """Extract the core intent from understanding.

        Uses normalized keyword matching so we don't rely on the LLM
        using exact words (e.g. "summary" vs "summarize", "moment" vs "when").
        """
        text = understanding.lower().strip()
        # Summarization: summary, overview, recap, etc.
        if any(
            kw in text
            for kw in (
                "summarize",
                "summarise",
                "summary",
                "overview",
                "recap",
                "sum up",
            )
        ):
            return "summarization"
        # Temporal: when, time, timestamp, moment, duration, etc.
        if any(
            kw in text
            for kw in (
                "when",
                "time",
                "timestamp",
                "moment",
                "at what point",
                "duration",
                "minute",
                "second",
                "temporal",
            )
        ):
            return "temporal_localization"
        # Comparison: compare, contrast, difference, versus, etc.
        if any(
            kw in text
            for kw in (
                "compare",
                "comparison",
                "contrast",
                "difference",
                "versus",
                "vs.",
            )
        ):
            return "comparison"
        # Analysis: analyze, examine, assess, look at, etc.
        if any(
            kw in text
            for kw in (
                "analyze",
                "analyse",
                "analysis",
                "examine",
                "assess",
                "look at",
                "break down",
            )
        ):
            return "analysis"
        return "general_qa"

    def _plan_approach(
        self, query: str, understanding: Dict, context: Dict
    ) -> Dict[str, Any]:
        """Step 2: Plan how to approach the analysis."""
        intent = understanding["intent"]

        planning_prompt = f"""Given this query and understanding, plan your approach:

Query: {query}
Intent: {intent}
Understanding: {understanding["thought"]}

Plan step by step:
1. What should I focus on in the video?
2. What should I listen for in the audio?
3. Should I analyze the whole video or specific segments?
4. What's the best order to process information?

Provide your plan."""

        response_text, _ = self.model.generate(
            video_path=None,
            audio_path=None,
            prompt=planning_prompt,
            max_frames=0,
        )

        return {
            "thought": response_text,
            "strategy": self._extract_strategy(response_text),
        }

    def _extract_strategy(self, plan: str) -> str:
        """Extract the strategy from the plan.

        Uses normalized keyword matching so we don't rely on exact LLM wording.
        """
        text = plan.lower().strip()

        focused_keywords = (
            "segment",
            "specific part",
            "portion",
            "section",
            "specific range",
            "focus on",
        )
        if any(kw in text for kw in focused_keywords):
            return "focused_analysis"

        comparative_keywords = (
            "compare",
            "comparison",
            "contrast",
            "difference",
            "versus",
        )
        if any(kw in text for kw in comparative_keywords):
            return "comparative_analysis"

        return "comprehensive_analysis"

    def _execute_analysis(
        self,
        query: str,
        video_path: Optional[str],
        audio_path: Optional[str],
        plan: Dict,
        context: Dict,
    ) -> Dict[str, Any]:
        """Step 3: Execute the actual analysis."""
        max_frames = context.get("max_frames", 256)
        max_audio_chunks = context.get("max_audio_chunks", None)
        duration = context.get("duration_seconds")

        # Build temporal grounding preamble
        grounding_parts: List[str] = []
        if max_frames and duration:
            interval = round(duration / max(max_frames, 1), 1)
            grounding_parts.append(
                f"The video is {round(duration, 1)}s long. You are "
                f"viewing {max_frames} uniformly sampled frames "
                f"(one every ~{interval}s). The audio covers the "
                f"full duration."
            )
        elif max_frames:
            grounding_parts.append(
                f"You are viewing {max_frames} uniformly sampled frames from the video."
            )

        segment_start = context.get("segment_start")
        segment_end = context.get("segment_end")
        if segment_start is not None and segment_end is not None:
            grounding_parts.append(
                f"This content covers {segment_start}s - "
                f"{segment_end}s of the original video."
            )

        grounding = "\n".join(grounding_parts) if grounding_parts else ""

        # Use the temporal index if one was built by the indexing node
        temporal_index = context.get("temporal_index")
        if temporal_index:
            index_block = f"--- VIDEO INDEX ---\n{temporal_index}\n--- END INDEX ---"
            analysis_prompt = (
                "IMPORTANT: A timestamped content index of the video "
                "is provided below. Use it to cite accurate timestamps "
                "in seconds when describing events. Format: "
                "'Around 30s, ...' or 'Between 120s and 180s, ...'.\n\n"
                f"{grounding}\n\n{index_block}\n\n"
                f"Query: {query}\n\n"
                "Analyze the video and audio content. For each key "
                "point or event you identify:\n"
                "1. State the time in seconds (from the index above)\n"
                "2. Describe what is shown or said\n"
                "3. Explain its significance\n\n"
                "Structure your response chronologically."
            )
        else:
            analysis_prompt = (
                "IMPORTANT: Your response MUST include approximate "
                "timestamps in seconds for every key point or event "
                "you describe. Use the audio and visual progression "
                "to estimate when things occur in the video. Format: "
                "'Around 30s, ...' or 'Between 120s and 180s, ...'.\n\n"
                f"{grounding}\n\n"
                f"Query: {query}\n\n"
                "Analyze the video and audio content. For each key "
                "point or event you identify:\n"
                "1. State the approximate time in seconds when it "
                "occurs\n"
                "2. Describe what is shown or said\n"
                "3. Explain its significance\n\n"
                "Structure your response chronologically, ordered by "
                "when things appear in the video."
            )

        response_text, _metadata = self.model.generate(
            video_path=video_path,
            audio_path=audio_path,
            prompt=analysis_prompt,
            max_frames=max_frames,
            max_audio_chunks=max_audio_chunks,
        )

        return {
            "thought": "Analyzing video and audio content...",
            "findings": response_text,
        }

    def _verify_result(
        self, query: str, analysis: Dict, context: Dict
    ) -> Dict[str, Any]:
        """Step 4: Verify the analysis is complete and accurate."""
        verification_prompt = f"""Review your analysis for completeness:

Original Query: {query}
Your Analysis: {analysis["findings"]}

Verify step by step:
1. Did you fully answer the question?
2. Is your answer supported by evidence from the video/audio?
3. Are there any contradictions or gaps?
4. What's your confidence level (0-1)?

Provide your verification assessment."""

        response_text, _ = self.model.generate(
            video_path=None,
            audio_path=None,
            prompt=verification_prompt,
            max_frames=0,
        )

        # Simple confidence extraction
        needs_refinement = (
            "incomplete" in response_text.lower()
            or "gap" in response_text.lower()
            or "not fully" in response_text.lower()
        )

        return {
            "thought": response_text,
            "assessment": "needs_refinement" if needs_refinement else "complete",
            "needs_refinement": needs_refinement,
        }

    def _refine_answer(
        self, query: str, analysis: Dict, verification: Dict, context: Dict
    ) -> Dict[str, Any]:
        """Step 5: Refine the answer based on verification."""
        refinement_prompt = f"""Improve your answer based on verification:

Original Query: {query}
Previous Answer: {analysis["findings"]}
Verification Feedback: {verification["thought"]}

Refine your answer:
1. Address any gaps identified
2. Add missing evidence
3. Clarify any contradictions
4. Provide a complete, improved answer

Provide your refined answer."""

        response_text, _ = self.model.generate(
            video_path=None,
            audio_path=None,
            prompt=refinement_prompt,
            max_frames=0,
        )

        return {
            "thought": "Refining answer based on verification...",
            "improved_answer": response_text,
        }

    def _estimate_confidence(self, reasoning_chain: List[Dict]) -> float:
        """Estimate confidence based on reasoning chain."""
        # Simple heuristic - can be improved
        if len(reasoning_chain) >= 5:
            return 0.9  # Went through full verification
        elif len(reasoning_chain) >= 3:
            return 0.75  # Basic reasoning
        else:
            return 0.6  # Minimal reasoning

    def get_reasoning_trace(self, result: Dict) -> str:
        """Format reasoning chain as readable text.

        Args:
            result: Result dict from reason()

        Returns:
            Formatted reasoning trace
        """
        trace = "# Reasoning Trace\n\n"

        for step_info in result["reasoning_chain"]:
            trace += f"## Step {step_info['step']}: {step_info['action']}\n"
            trace += f"**Thought:** {step_info['thought'][:200]}...\n"
            trace += f"**Result:** {step_info['result']}\n\n"

        trace += f"**Final Confidence:** {result['confidence']:.2f}\n"

        return trace
