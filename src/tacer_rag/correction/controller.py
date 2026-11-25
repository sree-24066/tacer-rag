"""
Type-Aware Correction Controller.

Coordinates detection confidence, evidence verification, and type-specific correction policies.
Supports:
- Sentence-level context evidence retrieval
- NLI-grounded semantic classification across the 4 hallucination categories
- Entailment filtering (prevents modifying factually supported claims)
- Reverse-offset multi-span editing (preserves character indices)
- Full audit trail of all actions taken
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from tacer_rag.utils import get_logger
from tacer_rag.detection.predict import PredictedSpan, is_isolated_stopword_or_punct
from tacer_rag.verification.evidence_selector import (
    EvidenceSelector, EvidenceCandidate, find_enclosing_sentence
)
from tacer_rag.verification.nli_verifier import (
    NLIVerifier, VerificationResult, classify_hallucination_type
)
from tacer_rag.correction.policies import (
    correct_evident_conflict, correct_subtle_conflict,
    correct_evident_baseless, correct_subtle_baseless,
    build_llm_prompt, PolicyResult
)
from tacer_rag.correction.llm_client import LLMCorrectionClient

logger = get_logger("correction_controller")


@dataclass
class SpanAction:
    """Record of an action taken on a specific suspicious span."""
    span_text: str
    start: int
    end: int
    label_type: str
    confidence: float
    nli_label: Optional[str]
    nli_confidence: Optional[float]
    evidence_text: str
    action: str
    explanation: str


@dataclass
class CorrectionDecision:
    """Full outcome of correcting an answer."""
    original_answer: str
    corrected_answer: str
    actions_taken: List[SpanAction] = field(default_factory=list)
    abstained: bool = False
    rounds_run: int = 1
    notes: str = ""


class CorrectionController:
    """
    Decoupled correction controller applying calibrated, type-aware policies.
    """

    def __init__(
        self,
        high_confidence_threshold: float = 0.75,
        low_confidence_threshold: float = 0.35,
        abstain_below: float = 0.25,
        evidence_selector: Optional[EvidenceSelector] = None,
        nli_verifier: Optional[NLIVerifier] = None,
        llm_client: Optional[LLMCorrectionClient] = None,
    ):
        self.high_conf = high_confidence_threshold
        self.low_conf = low_confidence_threshold
        self.abstain_below = abstain_below

        self.evidence_selector = evidence_selector or EvidenceSelector(top_k=2)
        self.nli_verifier = nli_verifier
        self.llm_client = llm_client or LLMCorrectionClient()

    def correct_answer(
        self,
        answer: str,
        spans: List[PredictedSpan],
        context: str,
    ) -> CorrectionDecision:
        """
        Apply type-aware and confidence-calibrated corrections to an answer.
        Spans are sorted in reverse offset order (from end to start) so that
        earlier character offsets remain stable during edits.
        """
        # Filter out invalid or isolated punctuation spans
        valid_spans = [
            s for s in spans
            if s.text.strip() and not is_isolated_stopword_or_punct(s.text)
        ]

        if not valid_spans:
            return CorrectionDecision(
                original_answer=answer,
                corrected_answer=answer,
                actions_taken=[],
                abstained=False,
                notes="No hallucinated spans detected.",
            )

        # Sort spans in reverse order: largest start first
        sorted_spans = sorted(valid_spans, key=lambda s: s.start, reverse=True)
        current_answer = answer
        actions: List[SpanAction] = []
        any_abstained = False

        for span in sorted_spans:
            # 1. Check confidence against thresholds
            if span.confidence < self.abstain_below:
                actions.append(SpanAction(
                    span_text=span.text,
                    start=span.start,
                    end=span.end,
                    label_type=span.label_type,
                    confidence=span.confidence,
                    nli_label=None,
                    nli_confidence=None,
                    evidence_text="",
                    action="abstain",
                    explanation=f"Confidence {span.confidence:.2f} < {self.abstain_below:.2f} (too uncertain).",
                ))
                any_abstained = True
                continue

            # 2. Evidence retrieval using sentence context
            enclosing_sent = find_enclosing_sentence(current_answer, span.start, span.end)
            evidence_cand = self.evidence_selector.get_best_evidence(
                context, span.text, enclosing_sentence=enclosing_sent
            )
            evidence_text = evidence_cand.text if evidence_cand else ""

            # 3. NLI verification and semantic grounding
            nli_label = None
            nli_conf = None
            if self.nli_verifier and evidence_text:
                nli_res = self.nli_verifier.verify(evidence_text, enclosing_sent or span.text)
                nli_label = nli_res.label
                nli_conf = nli_res.confidence

                # If NLI confirms claim is entailed by context, preserve it!
                if nli_label == "entailment" and nli_conf >= 0.70:
                    actions.append(SpanAction(
                        span_text=span.text,
                        start=span.start,
                        end=span.end,
                        label_type="O",
                        confidence=span.confidence,
                        nli_label=nli_label,
                        nli_confidence=nli_conf,
                        evidence_text=evidence_text,
                        action="preserve_entailed",
                        explanation="NLI verified claim is entailed by context; skipped correction.",
                    ))
                    continue

                # Ground the category based on semantic verification
                grounded_cat = classify_hallucination_type(
                    span.text, enclosing_sent, evidence_text, nli_res
                )
                if grounded_cat != "O":
                    span.label_type = grounded_cat

            # 4. Optional LLM completion
            llm_text = None
            if self.llm_client and self.llm_client.backend not in ("none", "rule_based"):
                prompt = build_llm_prompt(span.label_type, current_answer, span.text, evidence_text)
                llm_text = self.llm_client.generate(prompt["system"], prompt["user"])

            # 5. Apply Type-Specific Policy with Confidence Tiering
            policy_res: PolicyResult
            cat = span.label_type.upper()

            # Insufficient evidence handling: cannot safely replace conflict without evidence
            if not evidence_text and "CONFLICT" in cat:
                actions.append(SpanAction(
                    span_text=span.text,
                    start=span.start,
                    end=span.end,
                    label_type=span.label_type,
                    confidence=span.confidence,
                    nli_label=None,
                    nli_confidence=None,
                    evidence_text="",
                    action="abstain",
                    explanation="Insufficient context evidence to resolve conflicting claim; abstaining.",
                ))
                any_abstained = True
                continue

            if "CONFLICT" in cat and "EVIDENT" in cat:
                policy_res = correct_evident_conflict(
                    current_answer, span.text, span.start, span.end, evidence_text, llm_text
                )
            elif "CONFLICT" in cat and "SUBTLE" in cat:
                policy_res = correct_subtle_conflict(
                    current_answer, span.text, span.start, span.end, evidence_text, llm_text
                )
            elif "BASELESS" in cat and "EVIDENT" in cat:
                # High-confidence: excise baseless claim cleanly.
                # Intermediate confidence: qualify rather than hard delete if not verified by NLI.
                if span.confidence < self.high_conf and nli_label != "neutral":
                    policy_res = correct_subtle_baseless(
                        current_answer, span.text, span.start, span.end, evidence_text, llm_text
                    )
                else:
                    policy_res = correct_evident_baseless(
                        current_answer, span.text, span.start, span.end, evidence_text, llm_text
                    )
            else:  # Subtle baseless
                policy_res = correct_subtle_baseless(
                    current_answer, span.text, span.start, span.end, evidence_text, llm_text
                )

            if policy_res.applied:
                current_answer = policy_res.corrected_text
                actions.append(SpanAction(
                    span_text=span.text,
                    start=span.start,
                    end=span.end,
                    label_type=span.label_type,
                    confidence=span.confidence,
                    nli_label=nli_label,
                    nli_confidence=nli_conf,
                    evidence_text=evidence_text,
                    action=policy_res.action,
                    explanation=policy_res.explanation,
                ))

        applied_actions = [a for a in actions if a.action not in ("abstain", "preserve_entailed")]
        return CorrectionDecision(
            original_answer=answer,
            corrected_answer=current_answer,
            actions_taken=actions,
            abstained=any_abstained,
            notes=f"Corrected {len(applied_actions)} spans.",
        )
