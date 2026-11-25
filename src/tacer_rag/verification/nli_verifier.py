"""
Natural Language Inference (NLI) and Factuality Verifier.

Grounds suspicious claims against retrieved context evidence without external web knowledge.
Supports cross-encoder NLI models (e.g., cross-encoder/nli-deberta-v3-base) and
provides structured verification results (Entailment, Contradiction, Neutral) and
grounded 4-type classification.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from tacer_rag.utils import get_logger

logger = get_logger("nli_verifier")

# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

# Hedging / speculative language → signals SUBTLE_BASELESS
_HEDGING_RE = re.compile(
    r"\b(probably|likely|possibly|perhaps|maybe|might|could|seems|appears|"
    r"suggests|indirectly|partially|potentially|presumably|arguably|"
    r"theoretically|generally|typically|virtually|approximately|roughly|"
    r"perhaps|conceivably|ostensibly|purportedly|supposedly)\b",
    re.IGNORECASE,
)

# Degree-overstatement words → signals SUBTLE_CONFLICT (semantic distortion)
_OVERSTATEMENT_RE = re.compile(
    r"\b(completely|entirely|totally|absolutely|always|never|all|none|every|"
    r"eliminates|guaranteed|impossible|perfect|certainly|definitely|"
    r"entirely|wholly|utterly)\b",
    re.IGNORECASE,
)


def _has_mid_sentence_entity(text: str) -> bool:
    """
    Return True if *text* contains a proper noun that is NOT the sentence-initial
    word (which is capitalised purely for grammar), OR contains a digit sequence.

    This distinguishes spans like 'Berlin, Germany' (mid-word cap → True)
    from sentences like 'Regular exercise eliminates risk' (only leading cap → False).
    """
    words = text.split()
    # Check words beyond the first token for true proper-noun capitalisation
    if len(words) > 1:
        rest = " ".join(words[1:])
        if re.search(r"\b[A-Z][a-zA-Z]+", rest):
            return True
    # Digit sequences always indicate concrete data
    return bool(re.search(r"\b\d+\b", text))


@dataclass
class VerificationResult:
    """Structured result of evidence verification."""
    label: str                       # 'entailment', 'contradiction', 'neutral'
    probabilities: Dict[str, float]  # {'entailment': float, 'contradiction': float, 'neutral': float}
    confidence: float                # Probability of the predicted label
    evidence_text: str               # The context sentence used as premise
    claim_text: str                  # The suspicious claim verified as hypothesis


def classify_hallucination_type(
    span_text: str,
    enclosing_sentence: str,
    evidence_text: str,
    nli_result: Optional[VerificationResult] = None,
) -> str:
    """
    Classify a suspicious span into one of the 4 grounded categories:
    1. EVIDENT_CONFLICT:  Direct contradiction of a named entity, number, or place.
    2. SUBTLE_CONFLICT:   Semantic distortion / degree-overstatement without a concrete entity swap.
    3. EVIDENT_BASELESS:  Specific concrete claim (entity, fact) absent from context.
    4. SUBTLE_BASELESS:   Vague inference, speculation, or hedge not grounded in context.
    """
    if nli_result is None or not evidence_text:
        # Offline fallback: lexical presence check
        span_lower = span_text.lower().strip()
        ev_lower = evidence_text.lower() if evidence_text else ""
        if span_lower in ev_lower:
            return "O"
        has_entities = _has_mid_sentence_entity(span_text)
        return "EVIDENT_BASELESS" if has_entities else "SUBTLE_BASELESS"

    label = nli_result.label
    p_contra = nli_result.probabilities.get("contradiction", 0.0)
    p_entail = nli_result.probabilities.get("entailment", 0.0)
    p_neutral = nli_result.probabilities.get("neutral", 0.0)

    # 1. Supported — no hallucination
    if label == "entailment" or p_entail >= 0.70:
        return "O"

    # 2. Contradiction → Conflict
    #    EVIDENT  : a specific named entity or number is directly contradicted
    #    SUBTLE   : semantic distortion (degree, scope) without concrete entity swap
    if label == "contradiction" or p_contra >= 0.50:
        if _has_mid_sentence_entity(span_text):
            return "EVIDENT_CONFLICT"
        else:
            return "SUBTLE_CONFLICT"

    # 3. Neutral → Baseless (claim not grounded in context)
    #    SUBTLE   : hedging / speculative language present ("probably", "likely", …)
    #    EVIDENT  : definitive concrete claim with no support in context
    if _HEDGING_RE.search(span_text):
        return "SUBTLE_BASELESS"
    else:
        return "EVIDENT_BASELESS"


class NLIVerifier:
    """
    NLI Verifier for factuality checking.

    Premise: Retrieved context evidence sentence
    Hypothesis: Suspicious claim from generated response
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = None
        self.model = None
        self.id2label = {}

    def load(self) -> "NLIVerifier":
        """Load tokenizer and model weights."""
        if self.model is not None:
            return self

        logger.info("Loading NLI verifier: %s (device: %s)", self.model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        raw_id2label = self.model.config.id2label
        for idx, lbl in raw_id2label.items():
            name = str(lbl).lower()
            if "entail" in name:
                self.id2label[int(idx)] = "entailment"
            elif "contra" in name:
                self.id2label[int(idx)] = "contradiction"
            else:
                self.id2label[int(idx)] = "neutral"

        logger.info("NLI label mapping: %s", self.id2label)
        return self

    def verify(
        self,
        evidence: str,
        claim: str,
    ) -> VerificationResult:
        """
        Verify a single claim against evidence.
        """
        if self.model is None:
            try:
                self.load()
            except Exception as e:
                logger.warning("Could not load NLI model (%s). Using lexical heuristic.", e)
                # Heuristic fallback if model offline
                evidence_lower = evidence.lower()
                claim_lower = claim.lower()
                if claim_lower in evidence_lower:
                    return VerificationResult(
                        label="entailment",
                        probabilities={"entailment": 0.9, "contradiction": 0.05, "neutral": 0.05},
                        confidence=0.9,
                        evidence_text=evidence,
                        claim_text=claim,
                    )
                return VerificationResult(
                    label="neutral",
                    probabilities={"entailment": 0.2, "contradiction": 0.3, "neutral": 0.5},
                    confidence=0.5,
                    evidence_text=evidence,
                    claim_text=claim,
                )

        inputs = self.tokenizer(
            evidence,
            claim,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(0)
            probs = F.softmax(logits, dim=-1).cpu().tolist()

        prob_dict = {
            self.id2label.get(i, f"class_{i}"): round(float(p), 4)
            for i, p in enumerate(probs)
        }

        best_idx = int(torch.tensor(probs).argmax().item())
        best_label = self.id2label.get(best_idx, "neutral")
        confidence = prob_dict.get(best_label, 0.0)

        return VerificationResult(
            label=best_label,
            probabilities=prob_dict,
            confidence=confidence,
            evidence_text=evidence,
            claim_text=claim,
        )
