"""
Tests for evidence selection and NLI verification.
"""

import pytest
from tacer_rag.verification.evidence_selector import (
    EvidenceSelector, split_into_sentences, EvidenceCandidate
)


def test_split_into_sentences():
    text = "The capital of France is Paris. It is known for the Eiffel Tower! Is it cold in winter? Yes."
    sents = split_into_sentences(text)
    assert len(sents) == 4
    assert "Paris" in sents[0]
    assert "Eiffel Tower" in sents[1]


def test_evidence_selector_exact_match():
    context = (
        "Albert Einstein was born in Ulm, Germany in 1879. "
        "He developed the theory of relativity. "
        "He won the Nobel Prize in Physics in 1921."
    )
    selector = EvidenceSelector(top_k=2)
    claim = "Einstein won the Nobel Prize in Physics."

    candidates = selector.select_evidence(context, claim, top_k=1)
    assert len(candidates) == 1
    assert "Nobel Prize in Physics" in candidates[0].text
    assert candidates[0].score > 0.3


def test_evidence_selector_contradiction_candidate():
    context = (
        "The Apollo 11 mission landed on the Moon on July 20, 1969. "
        "Neil Armstrong and Buzz Aldrin were the astronauts who walked on the lunar surface. "
        "Michael Collins remained in lunar orbit."
    )
    selector = EvidenceSelector()
    claim = "Neil Armstrong landed on Mars in 1969."

    best = selector.get_best_evidence(context, claim)
    assert best is not None
    assert "Neil Armstrong" in best.text or "Apollo 11" in best.text
