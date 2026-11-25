"""
Context Evidence Selector and Factual Grounding.

Retrieves and ranks the most relevant sentence from retrieved context for a suspicious span.
Compares the full claim sentence against candidate context sentences to ensure accurate
premise matching without relying on external web knowledge.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tacer_rag.utils import get_logger

logger = get_logger("evidence_selector")


@dataclass
class EvidenceCandidate:
    """A ranked evidence sentence extracted from the context."""
    text: str
    score: float
    sentence_idx: int
    matched_entity: Optional[str] = None


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into clean, non-empty sentence candidates.
    Handles periods, question marks, exclamation marks, and newlines.
    """
    if not text or not text.strip():
        return []

    cleaned = re.sub(r"\s+", " ", text).strip()
    raw_sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    sentences = [s.strip() for s in raw_sentences if len(s.strip()) > 0]

    if not sentences and text.strip():
        sentences = [text.strip()]

    return sentences


def find_enclosing_sentence(text: str, span_start: int, span_end: int) -> str:
    """Find the full sentence in text that encloses the character offsets."""
    sentences = split_into_sentences(text)
    curr_idx = 0
    for s in sentences:
        pos = text.find(s, curr_idx)
        if pos != -1:
            end_pos = pos + len(s)
            curr_idx = end_pos
            if pos <= span_start and span_end <= end_pos + 1:
                return s
    # Fallback: return text around span
    left = max(0, span_start - 50)
    right = min(len(text), span_end + 50)
    return text[left:right]


class EvidenceSelector:
    """
    Ranks context sentences against a claim using TF-IDF cosine similarity.
    Ensures that verification is strictly grounded in the supplied context.
    """

    def __init__(self, top_k: int = 3, min_score_threshold: float = 0.10):
        self.top_k = top_k
        self.min_score_threshold = min_score_threshold

    def select_evidence(
        self,
        context: str,
        claim: str,
        enclosing_sentence: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> List[EvidenceCandidate]:
        """
        Rank sentences from context by relevance to the suspicious claim.

        Uses the enclosing sentence if provided to ensure the subject/predicate context
        is preserved when ranking context sentences.
        """
        k = top_k or self.top_k
        sentences = split_into_sentences(context)

        if not sentences:
            return []

        # Use enclosing sentence for query if available, otherwise claim
        query_text = enclosing_sentence if enclosing_sentence else claim
        if not query_text or not query_text.strip():
            return [EvidenceCandidate(text=sentences[0], score=0.0, sentence_idx=0)]

        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                ngram_range=(1, 2),
                lowercase=True,
            )
            corpus = sentences + [query_text]
            tfidf_matrix = vectorizer.fit_transform(corpus)

            sentence_vecs = tfidf_matrix[:-1]
            claim_vec = tfidf_matrix[-1]

            sims = cosine_similarity(claim_vec, sentence_vecs).flatten()
            ranked_indices = np.argsort(sims)[::-1]

            candidates = []
            for rank, idx in enumerate(ranked_indices[:k]):
                score = float(sims[idx])
                candidates.append(EvidenceCandidate(
                    text=sentences[idx],
                    score=round(score, 4),
                    sentence_idx=int(idx),
                ))

            return candidates

        except ValueError:
            return [
                EvidenceCandidate(text=sentences[i], score=0.0, sentence_idx=i)
                for i in range(min(k, len(sentences)))
            ]

    def get_best_evidence(
        self,
        context: str,
        claim: str,
        enclosing_sentence: Optional[str] = None,
    ) -> Optional[EvidenceCandidate]:
        """Convenience method returning the single most relevant evidence sentence above threshold."""
        candidates = self.select_evidence(context, claim, enclosing_sentence=enclosing_sentence, top_k=1)
        if candidates and candidates[0].score >= self.min_score_threshold:
            return candidates[0]
        return None
