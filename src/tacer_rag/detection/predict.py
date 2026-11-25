"""
Span and token prediction utilities for the TACER-RAG hallucination detector.

Features:
1. Subword-to-word boundary alignment (prevents token fragmentation like 'E' + 'el')
2. Punctuation and standalone stopword suppression (prevents flagging '.', ',', 'is', 'the')
3. Consecutive span merging (merges adjacent tokens into coherent entity phrases like 'Berlin, Germany')
4. Evidence-grounded confidence filtering
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any

import torch
import torch.nn.functional as F

from tacer_rag.data.dataset import (
    LABEL_MAP, ID_TO_LABEL, IGNORE_INDEX
)

# Common stopwords that should not form standalone hallucination spans
STANDALONE_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "and", "or",
    "it", "its", "they", "them", "their", "this", "that", "these", "those"
}

PUNCTUATION_CHARS = set(string.punctuation) | {" ", "\t", "\n", "\r"}


@dataclass
class PredictedSpan:
    """A detected hallucination span in the answer."""
    start: int              # Character start offset in answer
    end: int                # Character end offset in answer
    text: str               # Extracted substring
    label_type: str         # Category: EVIDENT_CONFLICT, SUBTLE_CONFLICT, etc.
    confidence: float       # Average confidence score over span tokens
    raw_tokens: List[str] = field(default_factory=list)


def is_isolated_stopword_or_punct(text: str) -> bool:
    """Check if text is purely punctuation or a standalone stopword."""
    stripped = text.strip()
    if not stripped:
        return True
    # Pure punctuation
    if all(c in string.punctuation for c in stripped):
        return True
    # Standalone stopword
    if stripped.lower() in STANDALONE_STOPWORDS:
        return True
    return False


def clean_and_merge_spans(
    raw_spans: List[PredictedSpan],
    answer_text: str,
    max_merge_gap: int = 3,
) -> List[PredictedSpan]:
    """
    Cleans raw token-level spans by:
    1. Expanding/snapping span boundaries to complete word boundaries.
    2. Merging adjacent spans separated only by whitespace or punctuation (e.g. 'Berlin' + ',' + 'Germany').
    3. Filtering out isolated punctuation and standalone stopwords.
    """
    if not raw_spans:
        return []

    # Step 1: Snap raw spans to word boundaries in answer_text
    snapped_spans: List[PredictedSpan] = []
    for s in raw_spans:
        start, end = s.start, s.end
        # Snap start backwards to whitespace/punct boundary
        while start > 0 and answer_text[start - 1].isalnum() and answer_text[start].isalnum():
            start -= 1
        # Snap end forwards to whitespace/punct boundary
        while end < len(answer_text) and answer_text[end - 1].isalnum() and answer_text[end].isalnum():
            end += 1

        span_text = answer_text[start:end].strip()
        if not span_text or is_isolated_stopword_or_punct(span_text):
            continue

        snapped_spans.append(PredictedSpan(
            start=start,
            end=end,
            text=span_text,
            label_type=s.label_type,
            confidence=s.confidence,
            raw_tokens=s.raw_tokens,
        ))

    if not snapped_spans:
        return []

    # Step 2: Sort spans by start offset
    snapped_spans.sort(key=lambda s: s.start)

    # Step 3: Merge adjacent or closely spaced spans
    merged: List[PredictedSpan] = []
    current = snapped_spans[0]

    for next_span in snapped_spans[1:]:
        gap_text = answer_text[current.end : next_span.start]
        # Merge if overlapping or gap is just punctuation/whitespace (e.g. ', ')
        if next_span.start <= current.end + max_merge_gap and (
            not gap_text.strip() or all(c in PUNCTUATION_CHARS for c in gap_text)
        ):
            new_end = max(current.end, next_span.end)
            new_text = answer_text[current.start : new_end].strip()
            # Combine confidences (take maximum or average)
            new_conf = round(max(current.confidence, next_span.confidence), 4)
            # Retain the more severe category (CONFLICT > BASELESS)
            chosen_type = current.label_type
            if "CONFLICT" in next_span.label_type and "CONFLICT" not in current.label_type:
                chosen_type = next_span.label_type
            elif "EVIDENT" in next_span.label_type and "EVIDENT" not in current.label_type:
                chosen_type = next_span.label_type

            current = PredictedSpan(
                start=current.start,
                end=new_end,
                text=new_text,
                label_type=chosen_type,
                confidence=new_conf,
                raw_tokens=current.raw_tokens + next_span.raw_tokens,
            )
        else:
            if not is_isolated_stopword_or_punct(current.text):
                merged.append(current)
            current = next_span

    if current and not is_isolated_stopword_or_punct(current.text):
        merged.append(current)

    return merged


def extract_spans_from_tokens(
    token_preds: List[int],
    token_probs: List[float],
    token_offsets: List[Tuple[int, int]],
    answer_text: str,
    token_strings: Optional[List[str]] = None,
    id_to_label: Dict[int, str] = ID_TO_LABEL,
    min_confidence: float = 0.50,
) -> List[PredictedSpan]:
    """
    Convert a sequence of token-level BIO predictions into character-level spans.
    Applies word boundary alignment, subword merging, and noise filtering.
    """
    if token_strings is None:
        token_strings = ["" for _ in token_preds]

    raw_spans: List[PredictedSpan] = []
    current_type: Optional[str] = None
    current_start: Optional[int] = None
    current_end: Optional[int] = None
    current_confidences: List[float] = []
    current_tokens: List[str] = []

    for pred_id, prob, (c_start, c_end), tok_str in zip(
        token_preds, token_probs, token_offsets, token_strings
    ):
        tag = id_to_label.get(pred_id, "O")

        # Suppress weak hallucination predictions
        if tag != "O" and prob < min_confidence:
            tag = "O"

        if tag.startswith("B-"):
            # Close existing span
            if current_type is not None and current_start is not None and current_end is not None:
                raw_spans.append(PredictedSpan(
                    start=current_start,
                    end=current_end,
                    text=answer_text[current_start:current_end],
                    label_type=current_type,
                    confidence=round(sum(current_confidences) / max(len(current_confidences), 1), 4),
                    raw_tokens=current_tokens,
                ))
            current_type = tag[2:]
            current_start = c_start
            current_end = c_end
            current_confidences = [prob]
            current_tokens = [tok_str]

        elif tag.startswith("I-"):
            span_type = tag[2:]
            if current_type == span_type and current_start is not None:
                current_end = c_end
                current_confidences.append(prob)
                current_tokens.append(tok_str)
            else:
                if current_type is not None and current_start is not None and current_end is not None:
                    raw_spans.append(PredictedSpan(
                        start=current_start,
                        end=current_end,
                        text=answer_text[current_start:current_end],
                        label_type=current_type,
                        confidence=round(sum(current_confidences) / max(len(current_confidences), 1), 4),
                        raw_tokens=current_tokens,
                    ))
                current_type = span_type
                current_start = c_start
                current_end = c_end
                current_confidences = [prob]
                current_tokens = [tok_str]

        else:  # tag == "O"
            if current_type is not None and current_start is not None and current_end is not None:
                raw_spans.append(PredictedSpan(
                    start=current_start,
                    end=current_end,
                    text=answer_text[current_start:current_end],
                    label_type=current_type,
                    confidence=round(sum(current_confidences) / max(len(current_confidences), 1), 4),
                    raw_tokens=current_tokens,
                ))
                current_type = None
                current_start = None
                current_end = None
                current_confidences = []
                current_tokens = []

    # Close trailing span
    if current_type is not None and current_start is not None and current_end is not None:
        raw_spans.append(PredictedSpan(
            start=current_start,
            end=current_end,
            text=answer_text[current_start:current_end],
            label_type=current_type,
            confidence=round(sum(current_confidences) / max(len(current_confidences), 1), 4),
            raw_tokens=current_tokens,
        ))

    # Apply boundary snapping and phrase merging
    return clean_and_merge_spans(raw_spans, answer_text)


def predict_single(
    model,
    tokenizer,
    context: str,
    answer: str,
    device: torch.device,
    max_length: int = 512,
    temperature: float = 1.0,
    min_confidence: float = 0.50,
) -> List[PredictedSpan]:
    """
    Run detector inference on a single (context, answer) pair.
    Uses sequence_ids and offset_mapping for accurate subword tracking.
    """
    model.eval()

    encoding = tokenizer(
        context,
        answer,
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    token_type_ids = encoding.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        raw_logits = outputs.logits.squeeze(0)
        # Apply temperature scaling
        scaled_logits = raw_logits / max(temperature, 1e-4)
        probs = F.softmax(scaled_logits, dim=-1)
        pred_ids = probs.argmax(dim=-1).cpu().tolist()
        max_probs = probs.max(dim=-1).values.cpu().tolist()

    # Extract sequence_ids to locate answer tokens accurately
    # ModernBERT fast tokenizer returns sequence_ids: 0 for context, 1 for answer, None for special
    seq_ids = encoding.sequence_ids(0)
    raw_offsets = encoding["offset_mapping"][0].tolist()
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())

    ans_preds = []
    ans_probs = []
    valid_offsets = []
    valid_tok_strings = []

    for idx, s_id in enumerate(seq_ids):
        if s_id == 1:
            c_start, c_end = raw_offsets[idx]
            if c_start == c_end:
                continue
            ans_preds.append(pred_ids[idx])
            ans_probs.append(max_probs[idx])
            valid_offsets.append((c_start, c_end))
            valid_tok_strings.append(all_tokens[idx])

    return extract_spans_from_tokens(
        ans_preds, ans_probs, valid_offsets, answer,
        token_strings=valid_tok_strings,
        min_confidence=min_confidence
    )
