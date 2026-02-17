"""Self-reflection and confidence estimation module.

Author: Ahmed Y. Radwan, SONIC-O1 Team
"""

import logging
import re
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SelfReflection:
    """Implements self-reflection and confidence assessment."""

    def __init__(self, model, config: Optional[Dict] = None):
        """Initialize self-reflection module.

        Args:
            model: The underlying language model (Qwen3OmniModel)
            config: Reflection configuration
        """
        self.model = model
        self.config = config or {}
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.max_refinement_attempts = self.config.get("max_refinement_attempts", 2)

    def evaluate_response(
        self,
        query: str,
        response: str,
        context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Evaluate response quality and confidence.

        Args:
            query: Original user query
            response: Generated response
            context: Additional context

        Returns:
            Dict with confidence score and assessment
        """
        logger.info("Evaluating response quality...")

        evaluation_prompt = f"""Evaluate this response critically:

Query: {query}
Response: {response}

Evaluate on these criteria:
1. **Completeness**: Does it fully answer the question? (0-10)
2. **Accuracy**: Is the information correct and grounded? (0-10)
3. **Clarity**: Is it clear and well-structured? (0-10)
4. **Evidence**: Is it supported by video/audio evidence? (0-10)

For each criterion, provide:
- Score (0-10)
- Brief justification

Then provide:
- Overall confidence (0.0-1.0)
- Key strengths
- Key weaknesses (if any)
- Recommendation: [ACCEPT, REFINE, REJECT]

Format your response as:
Completeness: [score] - [justification]
Accuracy: [score] - [justification]
Clarity: [score] - [justification]
Evidence: [score] - [justification]
Confidence: [0.0-1.0]
Strengths: [list]
Weaknesses: [list]
Recommendation: [ACCEPT/REFINE/REJECT]
"""

        evaluation, _ = self.model.generate(
            video_path=None,
            audio_path=None,
            prompt=evaluation_prompt,
            max_frames=0,
        )

        # Parse evaluation
        parsed = self._parse_evaluation(evaluation)

        return {
            "raw_evaluation": evaluation,
            "confidence": parsed["confidence"],
            "scores": parsed["scores"],
            "strengths": parsed["strengths"],
            "weaknesses": parsed["weaknesses"],
            "recommendation": parsed["recommendation"],
        }

    def _parse_evaluation(self, evaluation: str) -> Dict[str, Any]:
        """Parse evaluation text into structured format."""
        scores = {}
        confidence = 0.5  # Default
        strengths = []
        weaknesses = []
        recommendation = "ACCEPT"

        # Extract scores
        for criterion in ["Completeness", "Accuracy", "Clarity", "Evidence"]:
            pattern = rf"{criterion}:\s*(\d+)"
            match = re.search(pattern, evaluation, re.IGNORECASE)
            if match:
                scores[criterion.lower()] = int(match.group(1))

        # Extract confidence
        conf_pattern = r"Confidence:\s*(0?\.\d+|1\.0)"
        conf_match = re.search(conf_pattern, evaluation, re.IGNORECASE)
        if conf_match:
            confidence = float(conf_match.group(1))
        else:
            # Fallback: average of scores
            if scores:
                avg_score = sum(scores.values()) / len(scores)
                confidence = avg_score / 10.0

        # Extract strengths
        strengths_pattern = r"Strengths?:\s*(.+?)(?=Weaknesses?:|Recommendation:|$)"
        strengths_match = re.search(
            strengths_pattern, evaluation, re.IGNORECASE | re.DOTALL
        )
        if strengths_match:
            strengths = [
                s.strip()
                for s in strengths_match.group(1).split("\n")
                if s.strip() and not s.strip().startswith("-")
            ]

        # Extract weaknesses
        weaknesses_pattern = r"Weaknesses?:\s*(.+?)(?=Recommendation:|$)"
        weaknesses_match = re.search(
            weaknesses_pattern, evaluation, re.IGNORECASE | re.DOTALL
        )
        if weaknesses_match:
            weaknesses = [
                w.strip()
                for w in weaknesses_match.group(1).split("\n")
                if w.strip() and not w.strip().startswith("-")
            ]

        # Extract recommendation
        rec_pattern = r"Recommendation:\s*(ACCEPT|REFINE|REJECT)"
        rec_match = re.search(rec_pattern, evaluation, re.IGNORECASE)
        if rec_match:
            recommendation = rec_match.group(1).upper()
        else:
            # Fallback based on confidence
            if confidence >= 0.8:
                recommendation = "ACCEPT"
            elif confidence >= 0.5:
                recommendation = "REFINE"
            else:
                recommendation = "REJECT"

        return {
            "confidence": confidence,
            "scores": scores,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "recommendation": recommendation,
        }

    def refine_response(
        self,
        query: str,
        original_response: str,
        evaluation: Dict[str, Any],
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> str:
        """Refine response based on evaluation feedback.

        Args:
            query: Original query
            original_response: Initial response
            evaluation: Evaluation results
            video_path: Path to video (optional, for re-analysis)
            audio_path: Path to audio (optional, for re-analysis)
            max_frames: Max frames for re-analysis

        Returns:
            Refined response
        """
        logger.info("Refining response based on feedback...")

        weaknesses_text = "\n".join(f"- {w}" for w in evaluation["weaknesses"])

        refinement_prompt = f"""Improve your previous response:

Original Query: {query}
Previous Response: {original_response}

Issues Identified:
{weaknesses_text}

Confidence Score: {evaluation["confidence"]:.2f}

Instructions:
1. Address each weakness identified above
2. Maintain the strengths from your previous response
3. Add missing details or evidence
4. Ensure completeness and accuracy
5. Provide a clear, well-structured answer

Provide your improved response:"""

        refined, _ = self.model.generate(
            video_path=video_path,
            audio_path=audio_path,
            prompt=refinement_prompt,
            max_frames=max_frames or 128,  # Re-analyze with fewer frames
        )

        return refined

    def iterative_refinement(
        self,
        query: str,
        initial_response: str,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Iteratively refine response until confidence threshold met.

        Args:
            query: User query
            initial_response: Initial response
            video_path: Path to video
            audio_path: Path to audio
            max_frames: Max frames to process

        Returns:
            Dict with final response and refinement history
        """
        current_response = initial_response
        refinement_history = []
        attempts = 0

        while attempts < self.max_refinement_attempts:
            # Evaluate current response
            evaluation = self.evaluate_response(query, current_response)

            refinement_history.append(
                {
                    "attempt": attempts + 1,
                    "response": current_response,
                    "confidence": evaluation["confidence"],
                    "recommendation": evaluation["recommendation"],
                }
            )

            logger.info(
                f"Refinement attempt {attempts + 1}: "
                f"confidence={evaluation['confidence']:.2f}"
            )

            # Check if we should stop
            if evaluation["recommendation"] == "ACCEPT":
                logger.info("Response accepted")
                break

            if evaluation["recommendation"] == "REJECT":
                logger.warning("Response rejected, stopping refinement")
                break

            if evaluation["confidence"] >= self.confidence_threshold:
                logger.info("Confidence threshold met")
                break

            # Refine
            current_response = self.refine_response(
                query,
                current_response,
                evaluation,
                video_path,
                audio_path,
                max_frames,
            )

            attempts += 1

        final_evaluation = self.evaluate_response(query, current_response)

        return {
            "final_response": current_response,
            "final_confidence": final_evaluation["confidence"],
            "refinement_history": refinement_history,
            "total_attempts": attempts + 1,
        }

    def detect_hallucination(
        self,
        response: str,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Detect potential hallucinations in response.

        Args:
            response: Generated response
            video_path: Path to video for verification
            audio_path: Path to audio for verification

        Returns:
            Dict with hallucination assessment
        """
        logger.info("Checking for hallucinations...")

        hallucination_prompt = f"""Analyze this response for potential hallucinations:

Response: {response}

Check for:
1. Claims made without evidence from video/audio
2. Invented details not present in the source material
3. Contradictions or inconsistencies
4. Overly specific information that can't be verified

For each potential hallucination found:
- Quote the problematic claim
- Explain why it's questionable
- Rate severity (LOW/MEDIUM/HIGH)

If no hallucinations found, state: "No hallucinations detected"

Provide your analysis:"""

        analysis, _ = self.model.generate(
            video_path=None,
            audio_path=None,
            prompt=hallucination_prompt,
            max_frames=0,
        )

        has_hallucination = "no hallucinations detected" not in analysis.lower()

        return {
            "has_hallucination": has_hallucination,
            "analysis": analysis,
            "severity": self._extract_severity(analysis),
        }

    def _extract_severity(self, analysis: str) -> str:
        """Extract overall severity from hallucination analysis."""
        if "HIGH" in analysis:
            return "HIGH"
        elif "MEDIUM" in analysis:
            return "MEDIUM"
        elif "LOW" in analysis:
            return "LOW"
        else:
            return "NONE"
