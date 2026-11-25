"""
Post-Correction Verification Module.

Re-evaluates corrected answers to measure:
1. Removed hallucinations
2. Remaining hallucinations
3. Newly introduced hallucinations (Hallucination Regression)
4. New Hallucination Rate (NHR)

Enforces bounded iterative correction loops (maximum 2 rounds).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

from tacer_rag.utils import get_logger
from tacer_rag.detection.predict import PredictedSpan, predict_single, is_isolated_stopword_or_punct
from tacer_rag.correction.controller import CorrectionController, CorrectionDecision

logger = get_logger("post_verifier")


@dataclass
class PostVerificationResult:
    """Outcome of re-evaluating corrected text."""
    initial_answer: str
    final_answer: str
    rounds_executed: int
    initial_spans: List[PredictedSpan]
    final_spans: List[PredictedSpan]
    removed_count: int
    remaining_count: int
    newly_introduced_count: int
    new_hallucination_rate: float
    has_regression: bool
    reverted_to_original: bool
    decisions_per_round: List[CorrectionDecision] = field(default_factory=list)


class PostCorrectionVerifier:
    """
    Manages post-correction verification and bounded iterative loops.
    """

    def __init__(
        self,
        controller: CorrectionController,
        model,
        tokenizer,
        device,
        max_rounds: int = 2,
        revert_on_regression: bool = True,
    ):
        self.controller = controller
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_rounds = max_rounds
        self.revert_on_regression = revert_on_regression

    def verify_and_refine(
        self,
        context: str,
        answer: str,
        initial_spans: Optional[List[PredictedSpan]] = None,
    ) -> PostVerificationResult:
        """
        Run initial correction, re-detect, and perform a bounded 2nd round if needed.
        """
        # Round 1: detect if initial spans not provided
        if initial_spans is None:
            initial_spans = predict_single(
                self.model, self.tokenizer, context, answer, self.device
            )

        decisions: List[CorrectionDecision] = []

        if not initial_spans:
            # Nothing to correct
            return PostVerificationResult(
                initial_answer=answer,
                final_answer=answer,
                rounds_executed=0,
                initial_spans=[],
                final_spans=[],
                removed_count=0,
                remaining_count=0,
                newly_introduced_count=0,
                new_hallucination_rate=0.0,
                has_regression=False,
                reverted_to_original=False,
                decisions_per_round=[],
            )

        # Execute Round 1 Correction
        r1_decision = self.controller.correct_answer(answer, initial_spans, context)
        r1_decision.rounds_run = 1
        decisions.append(r1_decision)

        current_answer = r1_decision.corrected_answer

        # Re-detect on corrected text
        raw_re_spans = predict_single(
            self.model, self.tokenizer, context, current_answer, self.device
        )
        re_detected_spans = [
            s for s in raw_re_spans
            if s.text.strip() and not is_isolated_stopword_or_punct(s.text)
        ]

        rounds_run = 1

        # Check for newly introduced hallucinations or remaining errors
        new_spans = [
            s for s in re_detected_spans
            if s.text.strip().lower() not in [orig.text.strip().lower() for orig in initial_spans]
        ]
        has_regression = len(new_spans) > 0

        # If errors remain and rounds < max_rounds (e.g. round 2)
        if re_detected_spans and rounds_run < self.max_rounds and not has_regression:
            logger.info("Executing correction Round 2 for %d remaining spans", len(re_detected_spans))
            r2_decision = self.controller.correct_answer(current_answer, re_detected_spans, context)
            r2_decision.rounds_run = 2
            decisions.append(r2_decision)
            current_answer = r2_decision.corrected_answer
            rounds_run += 1

            # Final check after Round 2
            raw_re_spans2 = predict_single(
                self.model, self.tokenizer, context, current_answer, self.device
            )
            re_detected_spans = [
                s for s in raw_re_spans2
                if s.text.strip() and not is_isolated_stopword_or_punct(s.text)
            ]
            new_spans = [
                s for s in re_detected_spans
                if s.text.strip().lower() not in [orig.text.strip().lower() for orig in initial_spans]
            ]
            has_regression = len(new_spans) > 0

        # Calculate metrics
        initial_texts = {s.text for s in initial_spans}
        final_texts = {s.text for s in re_detected_spans}

        removed = len(initial_texts - final_texts)
        remaining = len(initial_texts & final_texts)
        newly_introduced = len(new_spans)
        nhr = round(newly_introduced / max(len(initial_spans), 1), 4)

        reverted = False
        if has_regression and self.revert_on_regression:
            logger.warning(
                "Correction introduced %d new hallucinations. Reverting to original answer.",
                newly_introduced,
            )
            current_answer = answer
            reverted = True

        return PostVerificationResult(
            initial_answer=answer,
            final_answer=current_answer,
            rounds_executed=rounds_run,
            initial_spans=initial_spans,
            final_spans=re_detected_spans,
            removed_count=removed,
            remaining_count=remaining,
            newly_introduced_count=newly_introduced,
            new_hallucination_rate=nhr,
            has_regression=has_regression,
            reverted_to_original=reverted,
            decisions_per_round=decisions,
        )
