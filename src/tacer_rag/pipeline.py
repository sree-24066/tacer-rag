"""
TACER-RAG End-to-End Pipeline.

Integrates:
1. Input processing and validation
2. ModernBERT token-level hallucination detection
3. Confidence calibration (Temperature Scaling)
4. Context evidence selection
5. NLI factuality verification and 4-type semantic classification
6. Type-aware correction controller
7. Post-correction verification and regression detection
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from tacer_rag.utils import get_logger, project_root
from tacer_rag.data.dataset import (
    LABEL_MAP, ID_TO_LABEL, N_LABELS, IGNORE_INDEX
)
from tacer_rag.detection.model import HallucinationDetector, DetectorConfig
from tacer_rag.detection.predict import PredictedSpan, predict_single
from tacer_rag.calibration.calibrate import TemperatureScaler
from tacer_rag.verification.evidence_selector import EvidenceSelector
from tacer_rag.verification.nli_verifier import NLIVerifier, classify_hallucination_type
from tacer_rag.correction.controller import CorrectionController, CorrectionDecision, SpanAction
from tacer_rag.correction.post_verifier import PostCorrectionVerifier, PostVerificationResult

logger = get_logger("pipeline")


@dataclass
class PipelineResult:
    """Output from the full TACER-RAG pipeline."""
    question: str
    context: str
    original_answer: str
    final_answer: str
    detected_spans: List[PredictedSpan]
    actions_taken: List[SpanAction]
    has_hallucinations: bool
    is_corrected: bool
    abstained: bool
    post_verification: Optional[PostVerificationResult] = None
    temperature_applied: float = 1.0


class TACERPipeline:
    """
    End-to-end TACER-RAG inference pipeline.
    """

    def __init__(
        self,
        model: HallucinationDetector,
        tokenizer: AutoTokenizer,
        device: torch.device,
        temperature: float = 1.0,
        high_confidence_threshold: float = 0.75,
        low_confidence_threshold: float = 0.35,
        abstain_below: float = 0.25,
        max_length: int = 512,
        nli_verifier: Optional[NLIVerifier] = None,
        max_correction_rounds: int = 2,
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.max_length = max_length
        self.scaler = TemperatureScaler(initial_temperature=temperature)

        self.evidence_selector = EvidenceSelector(top_k=2)
        self.nli_verifier = nli_verifier

        self.controller = CorrectionController(
            high_confidence_threshold=high_confidence_threshold,
            low_confidence_threshold=low_confidence_threshold,
            abstain_below=abstain_below,
            evidence_selector=self.evidence_selector,
            nli_verifier=self.nli_verifier,
        )

        self.post_verifier = PostCorrectionVerifier(
            controller=self.controller,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            max_rounds=max_correction_rounds,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: Optional[str] = None,
        load_nli: bool = True,
    ) -> "TACERPipeline":
        """Factory method to load pipeline directly from a trained model checkpoint."""
        ckpt_file = Path(checkpoint_path)
        if not ckpt_file.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

        ckpt = torch.load(ckpt_file, map_location="cpu")
        saved_args = ckpt.get("args", {})
        encoder_name = saved_args.get("encoder", "answerdotai/ModernBERT-base")
        max_length = saved_args.get("max_length", 512)

        target_device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        config = DetectorConfig(
            encoder_name=encoder_name,
            num_labels=N_LABELS,
            max_length=max_length,
        )
        model = HallucinationDetector(config)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

        tokenizer = AutoTokenizer.from_pretrained(encoder_name)

        # Look for calibration_params.json
        cal_path = ckpt_file.parent / "calibration_params.json"
        temperature = 1.0
        high_conf = 0.75
        low_conf = 0.35
        abstain_thresh = 0.25

        if cal_path.exists():
            try:
                with open(cal_path, encoding="utf-8") as f:
                    cal_data = json.load(f)
                temperature = cal_data.get("temperature", 1.0)
                selected = cal_data.get("selected_thresholds", {})
                high_conf = selected.get("high_confidence_threshold", 0.75)
                low_conf = selected.get("low_confidence_threshold", 0.35)
                abstain_thresh = selected.get("abstain_below_threshold", 0.25)
                logger.info("Loaded calibration params: T=%.3f, high=%.2f, low=%.2f",
                            temperature, high_conf, low_conf)
            except Exception as e:
                logger.warning("Could not read calibration_params.json (%s), using defaults.", e)

        nli_verifier = None
        if load_nli:
            try:
                nli_verifier = NLIVerifier(device=str(target_device)).load()
            except Exception as e:
                logger.warning("NLI verifier offline (%s); using heuristic verifier.", e)
                nli_verifier = NLIVerifier(device=str(target_device))

        return cls(
            model=model,
            tokenizer=tokenizer,
            device=target_device,
            temperature=temperature,
            high_confidence_threshold=high_conf,
            low_confidence_threshold=low_conf,
            abstain_below=abstain_thresh,
            max_length=max_length,
            nli_verifier=nli_verifier,
        )

    def _nli_sentence_detect(self, context: str, answer: str) -> List[PredictedSpan]:
        """
        NLI-based sentence-level fallback hallucination detector.

        Splits the answer into sentences, verifies each against the full context
        using the NLI verifier, and returns PredictedSpans for any sentence that
        is classified as a hallucination (non-O label).  This is used when the
        fine-tuned token classifier produces no detections.
        """
        if self.nli_verifier is None:
            return []

        # Split on sentence boundaries: '.', '!', '?' optionally followed by space
        sentence_re = re.compile(r"(?<=[.!?])\s+")
        raw_sentences = sentence_re.split(answer.strip())

        spans: List[PredictedSpan] = []
        search_from = 0

        for sent in raw_sentences:
            sent = sent.strip()
            if not sent or len(sent.split()) < 4:
                # Skip very short fragments — too noisy for NLI
                search_from = answer.find(sent, search_from) + len(sent) if sent in answer[search_from:] else search_from
                continue

            sent_start = answer.find(sent, search_from)
            if sent_start == -1:
                continue
            sent_end = sent_start + len(sent)
            search_from = sent_end

            try:
                nli_result = self.nli_verifier.verify(context, sent)
            except Exception as exc:
                logger.warning("NLI fallback failed for sentence (%s). Skipping.", exc)
                continue

            label = classify_hallucination_type(sent, sent, context, nli_result)
            if label == "O":
                continue

            # Use NLI confidence as the span confidence (capped to [0.50, 0.99])
            confidence = round(max(0.50, min(nli_result.confidence, 0.99)), 4)
            spans.append(PredictedSpan(
                start=sent_start,
                end=sent_end,
                text=sent,
                label_type=label,
                confidence=confidence,
            ))
            logger.info(
                "NLI fallback detected '%s...' as %s (conf=%.3f, nli=%s)",
                sent[:40], label, confidence, nli_result.label,
            )

        return spans

    @staticmethod
    def _word_overlap_filter(spans: List[PredictedSpan], context: str) -> List[PredictedSpan]:
        """
        Fast string-based pre-filter: drop any span whose words are ≥80% present
        in the context.  Requires no ML model and catches verbatim / near-verbatim
        false positives that the ML detector misfires on.
        """
        import re as _re
        ctx_clean = _re.sub(r"[^\w\s]", " ", context.lower())
        ctx_words = set(ctx_clean.split())
        kept: List[PredictedSpan] = []
        for span in spans:
            sp_clean = _re.sub(r"[^\w\s]", " ", span.text.lower())
            sp_words = sp_clean.split()
            if not sp_words:
                kept.append(span)
                continue
            overlap = sum(1 for w in sp_words if w in ctx_words) / len(sp_words)
            if overlap >= 0.80:
                logger.info(
                    "Word-overlap filter: dropping '%s...' (%.0f%% words in context)",
                    span.text[:50], overlap * 100,
                )
            else:
                kept.append(span)
        return kept

    def _nli_entailment_filter(self, spans: List[PredictedSpan], context: str) -> List[PredictedSpan]:
        """
        NLI entailment gate: drop any span that is *supported* by the context.

        The fine-tuned ML model is sometimes miscalibrated and flags sentences
        that are verbatim or semantically identical to the context.  Running NLI
        on each candidate span catches these false positives: if entailment
        probability is above the threshold the span is genuine context content
        and must NOT be treated as a hallucination.
        """
        if self.nli_verifier is None:
            return spans

        kept: List[PredictedSpan] = []
        for span in spans:
            try:
                nli_result = self.nli_verifier.verify(context, span.text)
            except Exception as exc:
                logger.warning("NLI gate failed for span (%s); keeping span.", exc)
                kept.append(span)
                continue

            p_entail = nli_result.probabilities.get("entailment", 0.0)

            if p_entail >= 0.40:
                # Context supports this span — it is NOT a hallucination
                logger.info(
                    "NLI gate: dropping false-positive span '%s...' "
                    "(p_entail=%.3f, was %s)",
                    span.text[:50], p_entail, span.label_type,
                )
            else:
                kept.append(span)

        return kept

    def detect_spans(
        self, context: str, answer: str, apply_temperature: bool = True
    ) -> List[PredictedSpan]:
        """Run detection with confidence calibration and word-level span reconstruction.

        Primary detector: fine-tuned ModernBERT token classifier.
        NLI entailment gate: filters out false positives entailed by context.
        Fallback detector: NLI sentence-level verifier (activates when primary finds nothing after filtering).
        """
        applied_temp = self.temperature if apply_temperature else 1.0
        spans = predict_single(
            self.model,
            self.tokenizer,
            context,
            answer,
            self.device,
            max_length=self.max_length,
            temperature=applied_temp,
            min_confidence=self.controller.abstain_below,
        )

        # Apply word-overlap filter FIRST (no ML needed — catches verbatim false positives)
        if spans:
            spans = self._word_overlap_filter(spans, context)

        # Apply NLI entailment gate to remove context-supported false positives
        if spans and self.nli_verifier is not None:
            spans = self._nli_entailment_filter(spans, context)

        if not spans:
            logger.info("Primary detector found no spans — activating NLI sentence-level fallback.")
            spans = self._nli_sentence_detect(context, answer)

        return spans

    def process(
        self,
        context: str,
        answer: str,
        question: str = "",
        run_post_verification: bool = True,
        apply_temperature: bool = True,
    ) -> PipelineResult:
        """
        Run the full TACER-RAG pipeline on an input query, context, and answer.
        """
        applied_temp = self.temperature if apply_temperature else 1.0

        # Early exit: if answer has ≥80% word overlap with context it cannot be a
        # hallucination regardless of what the ML model says.
        import re as _re
        def _word_overlap(a: str, b: str) -> float:
            clean = lambda s: set(_re.sub(r"[^\w\s]", " ", s.lower()).split())
            a_words = clean(a)
            return len(a_words & clean(b)) / max(len(a_words), 1)

        if _word_overlap(answer, context) >= 0.80:
            logger.info("Early exit: answer has ≥80%% word overlap with context — no hallucinations.")
            return PipelineResult(
                question=question,
                context=context,
                original_answer=answer,
                final_answer=answer,
                detected_spans=[],
                actions_taken=[],
                has_hallucinations=False,
                is_corrected=False,
                abstained=False,
                post_verification=None,
                temperature_applied=applied_temp,
            )

        # Step 1: Detect hallucinated spans with calibrated confidence & word merging
        spans = self.detect_spans(context, answer, apply_temperature=apply_temperature)

        if not spans:
            return PipelineResult(
                question=question,
                context=context,
                original_answer=answer,
                final_answer=answer,
                detected_spans=[],
                actions_taken=[],
                has_hallucinations=False,
                is_corrected=False,
                abstained=False,
                post_verification=None,
                temperature_applied=applied_temp,
            )

        # Step 2: Correct answer using type-aware policies and post-verification
        if run_post_verification:
            pv_res = self.post_verifier.verify_and_refine(context, answer, initial_spans=spans)
            final_answer = pv_res.final_answer
            all_actions = []
            for d in pv_res.decisions_per_round:
                all_actions.extend(d.actions_taken)

            abstained = any(d.abstained for d in pv_res.decisions_per_round)
            is_corrected = final_answer != answer

            return PipelineResult(
                question=question,
                context=context,
                original_answer=answer,
                final_answer=final_answer,
                detected_spans=spans,
                actions_taken=all_actions,
                has_hallucinations=True,
                is_corrected=is_corrected,
                abstained=abstained,
                post_verification=pv_res,
                temperature_applied=applied_temp,
            )
        else:
            decision = self.controller.correct_answer(answer, spans, context)
            return PipelineResult(
                question=question,
                context=context,
                original_answer=answer,
                final_answer=decision.corrected_answer,
                detected_spans=spans,
                actions_taken=decision.actions_taken,
                has_hallucinations=True,
                is_corrected=decision.corrected_answer != answer,
                abstained=decision.abstained,
                post_verification=None,
                temperature_applied=applied_temp,
            )
