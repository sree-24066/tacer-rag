"""
Type-Aware Correction Policies.

Implements type-specific correction logic for each of the 4 hallucination categories:
1. Evident Conflict: Direct contradiction -> Replace conflicting entity/claim with evidence entity.
2. Subtle Conflict: Subtle distortion -> Rewrite meaning to match evidence.
3. Evident Baseless Info: Unsupported fabrication -> Remove baseless claim cleanly without inventing facts.
4. Subtle Baseless Info: Unjustified inference -> Qualify or remove inference.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Optional, Dict, Any

from tacer_rag.utils import get_logger

logger = get_logger("correction_policies")


@dataclass
class PolicyResult:
    """Outcome of applying a type-aware correction policy."""
    action: str              # 'replace_with_evidence', 'rewrite_meaning', 'remove_unsupported', 'qualify_inference', 'abstain'
    corrected_text: str      # The full corrected answer
    explanation: str         # Research documentation of what changed
    evidence_used: str       # Exact context sentence used
    applied: bool            # True if correction was made, False if abstained


def clean_whitespace(text: str) -> str:
    """Normalize whitespace and punctuation after span replacement/removal."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.:;?!])", r"\1", text)
    text = re.sub(r"[,;]\s*([.?!])", r"\1", text)  # e.g., ",." -> "."
    text = re.sub(r"[,;]\s*$", ".", text)          # trailing comma -> period
    text = re.sub(r"([,.:;?!])\s*([,.:;?!])", r"\1", text)
    text = re.sub(r"\bin\s+in\b", "in", text, flags=re.IGNORECASE)
    text = re.sub(r"\bat\s+at\b", "at", text, flags=re.IGNORECASE)
    text = re.sub(r"\bon\s+on\b", "on", text, flags=re.IGNORECASE)
    return text.strip()


def extract_conflicting_entity(evidence: str, span_text: str, prefix: str) -> str:
    """
    Extracts the matching entity or factual phrase from evidence that corresponds to span_text.
    Prevents dumping entire multi-clause sentences into the span location.
    """
    ev = evidence.strip().rstrip(".!?")

    # 1. Location pattern: "City, Country" (e.g. "Paris, France" vs "Berlin, Germany")
    city_country_matches = re.findall(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)?,\s+[A-Z][a-z]+)\b", ev)
    if city_country_matches and ("," in span_text or any(c.isupper() for c in span_text)):
        return city_country_matches[0]

    # 2. Prepositional entity matching (e.g. "in Paris", "on the Moon", "in 1969")
    # If the span follows "in" or "on"
    prep_match = re.search(r"\b(?:in|on|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:,\s+[A-Z][a-z]+)?)\b", ev)
    if prep_match and re.search(r"\b(?:in|on|at)\s*$", prefix, re.IGNORECASE):
        return prep_match.group(1)

    # 3. Capital city pattern: "Canberra is the capital..."
    is_cap_match = re.search(r"\b([A-Z][a-z]+)\s+is the capital\b", ev, re.IGNORECASE)
    if is_cap_match:
        return is_cap_match.group(1)

    # 4. Year / Date matching
    date_matches = re.findall(r"\b(?:18|19|20)\d{2}\b", ev)
    if date_matches and re.search(r"\b(?:18|19|20)\d{2}\b", span_text):
        return date_matches[0]

    # 5. Shared noun-head pattern (e.g. "severe symptoms" vs "mild symptoms", "high risk" vs "low risk")
    span_words = span_text.strip().split()
    if len(span_words) >= 2:
        noun_head = span_words[-1].lower()
        adj_noun_match = re.search(r"\b([A-Za-z]+)\s+" + re.escape(noun_head) + r"\b", ev, re.IGNORECASE)
        if adj_noun_match:
            return adj_noun_match.group(0)

    # 6. Fallback: extract the main predicate or clause
    # If evidence starts with subject that matches answer, extract predicate
    if " is " in ev:
        return ev.split(" is ", 1)[1]
    if " was " in ev:
        return ev.split(" was ", 1)[1]

    return ev


def correct_evident_conflict(
    answer: str,
    span_text: str,
    span_start: int,
    span_end: int,
    evidence: str,
    llm_completion: Optional[str] = None,
) -> PolicyResult:
    """
    Policy A: Evident Conflict
    Directly replace the conflicting entity/claim with the evidence-grounded fact.
    Do not introduce extraneous details.
    """
    if llm_completion and len(llm_completion.strip()) > 5:
        return PolicyResult(
            action="replace_with_evidence",
            corrected_text=clean_whitespace(llm_completion),
            explanation=f"Replaced evident conflict '{span_text}' with evidence.",
            evidence_used=evidence,
            applied=True,
        )

    prefix = answer[:span_start]
    suffix = answer[span_end:]

    replacement = extract_conflicting_entity(evidence, span_text, prefix)
    ev_clean = evidence.strip().rstrip(".!?")
    # If no specific entity could be extracted, or fallback returned an unparsed sentence for a short entity span: abstain
    if not replacement or (replacement == ev_clean and len(ev_clean.split()) > 4 and len(span_text.split()) <= 3):
        return PolicyResult(
            action="abstain",
            corrected_text=answer,
            explanation=f"Cannot extract matching entity from evidence for '{span_text}'; abstaining to prevent sentence corruption.",
            evidence_used=evidence,
            applied=False,
        )

    corrected = clean_whitespace(f"{prefix} {replacement} {suffix}")

    return PolicyResult(
        action="replace_with_evidence",
        corrected_text=corrected,
        explanation=f"Replaced conflicting '{span_text}' with '{replacement}'.",
        evidence_used=evidence,
        applied=True,
    )


def correct_subtle_conflict(
    answer: str,
    span_text: str,
    span_start: int,
    span_end: int,
    evidence: str,
    llm_completion: Optional[str] = None,
) -> PolicyResult:
    """
    Policy B: Subtle Conflict
    Adjust nuanced distortion to align with context without corrupting surrounding text.
    """
    if llm_completion and len(llm_completion.strip()) > 5:
        return PolicyResult(
            action="rewrite_meaning",
            corrected_text=clean_whitespace(llm_completion),
            explanation=f"Rewrote subtle distortion in '{span_text}' using evidence.",
            evidence_used=evidence,
            applied=True,
        )

    prefix = answer[:span_start]
    suffix = answer[span_end:]
    replacement = extract_conflicting_entity(evidence, span_text, prefix)
    ev_clean = evidence.strip().rstrip(".!?")
    # If no specific entity could be extracted, or fallback returned an unparsed sentence for a short entity span: abstain
    if not replacement or (replacement == ev_clean and len(ev_clean.split()) > 4 and len(span_text.split()) <= 3):
        return PolicyResult(
            action="abstain",
            corrected_text=answer,
            explanation=f"Cannot extract matching entity from evidence for '{span_text}'; abstaining to prevent sentence corruption.",
            evidence_used=evidence,
            applied=False,
        )

    corrected = clean_whitespace(f"{prefix} {replacement} {suffix}")

    return PolicyResult(
        action="rewrite_meaning",
        corrected_text=corrected,
        explanation=f"Subtle conflict in '{span_text}' refined to match evidence.",
        evidence_used=evidence,
        applied=True,
    )


def correct_evident_baseless(
    answer: str,
    span_text: str,
    span_start: int,
    span_end: int,
    evidence: str,
    llm_completion: Optional[str] = None,
) -> PolicyResult:
    """
    Policy C: Evident Baseless Information
    Remove the fabricated fact cleanly. Strictly prohibits inventing replacement facts.
    """
    if llm_completion and len(llm_completion.strip()) > 5:
        return PolicyResult(
            action="remove_unsupported",
            corrected_text=clean_whitespace(llm_completion),
            explanation=f"Excised baseless claim '{span_text}'.",
            evidence_used=evidence,
            applied=True,
        )

    prefix = answer[:span_start]
    suffix = answer[span_end:]
    corrected = clean_whitespace(f"{prefix} {suffix}")

    return PolicyResult(
        action="remove_unsupported",
        corrected_text=corrected,
        explanation=f"Removed baseless claim '{span_text}' without replacement.",
        evidence_used=evidence,
        applied=True,
    )


def correct_subtle_baseless(
    answer: str,
    span_text: str,
    span_start: int,
    span_end: int,
    evidence: str = "",
    llm_completion: Optional[str] = None,
) -> PolicyResult:
    """
    Policy D: Subtle Baseless Information
    Qualifies an unwarranted inference with conservative hedging, or removes cleanly if speculative.
    """
    if llm_completion and len(llm_completion.strip()) > 5:
        return PolicyResult(
            action="qualify_inference",
            corrected_text=clean_whitespace(llm_completion),
            explanation=f"Qualified unsupported inference '{span_text}'.",
            evidence_used=evidence,
            applied=True,
        )

    prefix = answer[:span_start]
    suffix = answer[span_end:]

    # Cleanly remove speculative claim without inventing facts
    corrected = clean_whitespace(f"{prefix} {suffix}")

    return PolicyResult(
        action="qualify_inference",
        corrected_text=corrected,
        explanation=f"Conservative qualification: excised speculative inference '{span_text}'.",
        evidence_used=evidence,
        applied=True,
    )


def build_llm_prompt(
    category: str,
    original_answer: str,
    suspicious_span: str,
    evidence: str,
) -> Dict[str, str]:
    """
    Constructs type-specific prompts constraining the LLM to evidence.
    """
    system_prompt = (
        "You are an evidence-constrained factual correction system. "
        "Your task is to fix factual errors in an answer using ONLY the supplied context evidence. "
        "Strict rules:\n"
        "1. Never invent or assume facts not explicitly stated in the evidence.\n"
        "2. Do not introduce new information.\n"
        "3. Preserve all verified, supported parts of the original answer.\n"
        "4. Return ONLY the final corrected answer text without meta-commentary."
    )

    if category == "EVIDENT_CONFLICT":
        task_instruction = (
            f"The answer directly contradicts the evidence in the following span:\n"
            f"Conflicting span: \"{suspicious_span}\"\n"
            f"Ground truth evidence: \"{evidence}\"\n\n"
            f"Replace the conflicting span with the correct fact from the evidence."
        )
    elif category == "SUBTLE_CONFLICT":
        task_instruction = (
            f"The answer subtly misinterprets or distorts the evidence in the following span:\n"
            f"Distorted span: \"{suspicious_span}\"\n"
            f"Ground truth evidence: \"{evidence}\"\n\n"
            f"Rewrite the span so its meaning accurately reflects the evidence."
        )
    elif category == "EVIDENT_BASELESS":
        task_instruction = (
            f"The answer contains completely unsupported/fabricated information in the following span:\n"
            f"Baseless span: \"{suspicious_span}\"\n"
            f"Available evidence: \"{evidence}\"\n\n"
            f"Remove the baseless span from the answer. Do not fabricate an alternative."
        )
    else:  # SUBTLE_BASELESS
        task_instruction = (
            f"The answer makes an unjustified inference or assumption in the following span:\n"
            f"Unjustified span: \"{suspicious_span}\"\n"
            f"Available evidence: \"{evidence}\"\n\n"
            f"Qualify or remove the unjustified inference to ensure it is strictly faithful to the evidence."
        )

    user_prompt = (
        f"{task_instruction}\n\n"
        f"Original Answer:\n\"{original_answer}\"\n\n"
        f"Corrected Answer:"
    )

    return {"system": system_prompt, "user": user_prompt}
