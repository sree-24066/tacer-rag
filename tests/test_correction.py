"""
Unit tests for Type-Aware Correction Policies, Controller, and Post-Verification.
"""

import pytest
from tacer_rag.detection.predict import PredictedSpan
from tacer_rag.correction.policies import (
    correct_evident_conflict, correct_subtle_conflict,
    correct_evident_baseless, correct_subtle_baseless
)
from tacer_rag.correction.controller import CorrectionController
from tacer_rag.verification.evidence_selector import EvidenceSelector


def test_correct_evident_conflict():
    answer = "The capital of Australia is Sydney according to many people."
    # span: "Sydney" (start: 28, end: 34)
    evidence = "Canberra is the capital city of Australia."
    result = correct_evident_conflict(answer, "Sydney", 28, 34, evidence)

    assert result.applied is True
    assert "Canberra" in result.corrected_text
    assert "Sydney" not in result.corrected_text
    assert result.corrected_text == "The capital of Australia is Canberra according to many people."


def test_correct_eiffel_tower_evident_conflict():
    answer = "The Eiffel Tower is located in Berlin, Germany. It was constructed between 1887 and 1889 and is 330 metres tall."
    span = "Berlin, Germany"
    start = answer.find(span)
    end = start + len(span)
    evidence = "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."
    result = correct_evident_conflict(answer, span, start, end, evidence)

    assert result.applied is True
    assert "Paris, France" in result.corrected_text
    assert "Berlin, Germany" not in result.corrected_text
    assert "1887 and 1889" in result.corrected_text
    assert "330 metres tall" in result.corrected_text


def test_correct_evident_baseless():
    answer = "Python was released in 1991 and was named after Monty Python, which won an Oscar in 1980."
    # span: "which won an Oscar in 1980" (start: 61, end: 88)
    evidence = "Python is a programming language first released in 1991 by Guido van Rossum."
    result = correct_evident_baseless(answer, "which won an Oscar in 1980", 61, 88, evidence)

    assert result.applied is True
    assert "Oscar" not in result.corrected_text
    assert "Python was released in 1991 and was named after Monty Python." in result.corrected_text


def test_correction_controller_reverse_order():
    """Verify that multiple spans are edited in reverse order without offset corruption."""
    answer = "Fact one: red. Fact two: blue. Fact three: green."
    # Spans: "red" (10-13) and "green" (43-48)
    spans = [
        PredictedSpan(start=10, end=13, text="red", label_type="EVIDENT_CONFLICT", confidence=0.9),
        PredictedSpan(start=43, end=48, text="green", label_type="EVIDENT_BASELESS", confidence=0.85),
    ]
    context = "Fact one is crimson. Fact two is blue."

    controller = CorrectionController(
        high_confidence_threshold=0.7,
        low_confidence_threshold=0.4,
        abstain_below=0.3,
    )
    decision = controller.correct_answer(answer, spans, context)

    assert decision.abstained is False
    assert len(decision.actions_taken) == 2
    # The baseless span "green" should be excised
    assert "green" not in decision.corrected_answer


def test_correction_controller_abstain_low_confidence():
    """Verify that low-confidence predictions are abstained."""
    answer = "The flight was delayed by 3 hours."
    spans = [
        PredictedSpan(start=26, end=33, text="3 hours", label_type="SUBTLE_CONFLICT", confidence=0.15),
    ]
    context = "The flight departure had a minor schedule update."

    controller = CorrectionController(abstain_below=0.30)
    decision = controller.correct_answer(answer, spans, context)

    assert decision.abstained is True
    assert decision.actions_taken[0].action == "abstain"
    # Text should remain unchanged on abstention
    assert decision.corrected_answer == answer


def test_no_hallucination():
    """Verify that an answer with no hallucinated spans remains untouched."""
    answer = "The Earth orbits the Sun in approximately 365.25 days."
    context = "The Earth is the third planet from the Sun and orbits it every 365.25 days."
    controller = CorrectionController()
    decision = controller.correct_answer(answer, [], context)

    assert decision.abstained is False
    assert decision.corrected_answer == answer
    assert len(decision.actions_taken) == 0


def test_insufficient_evidence_conflict_abstain():
    """Verify that if context has no evidence for a conflicting claim, the system abstains."""
    answer = "The rover landed in Jezero crater on Mars."
    # span: "Jezero crater" (start: 20, end: 33)
    spans = [
        PredictedSpan(start=20, end=33, text="Jezero crater", label_type="EVIDENT_CONFLICT", confidence=0.88),
    ]
    # Context has completely unrelated information with zero mention of landing sites
    context = "Mars has two small moons named Phobos and Deimos."
    controller = CorrectionController()
    # Selector top_k=2 with unrelated context produces candidate with low score or no text
    decision = controller.correct_answer(answer, spans, context)

    # When context has no information about the conflicting entity, it should not invent a site
    assert decision.abstained is True or "Jezero crater" in decision.corrected_answer


def test_subtle_baseless_qualification():
    """Verify that subtle baseless claims are qualified without hallucinating facts."""
    answer = "The experiment concluded successfully, probably proving Einstein was right."
    span = "probably proving Einstein was right"
    start = answer.find(span)
    end = start + len(span)
    evidence = "The experiment measured gravitational wave signals successfully."

    result = correct_subtle_baseless(answer, span, start, end, evidence)
    assert result.applied is True
    assert "probably proving Einstein was right" not in result.corrected_text
    assert "The experiment concluded successfully." in result.corrected_text


def test_subtle_conflict_refinement():
    """Verify that subtle conflict refines distortion to match evidence."""
    answer = "The patient showed severe symptoms within 2 days."
    span = "severe symptoms"
    start = answer.find(span)
    end = start + len(span)
    evidence = "The patient developed mild symptoms after initial exposure."

    result = correct_subtle_conflict(answer, span, start, end, evidence)
    assert result.applied is True
    assert result.corrected_text != answer


def test_post_verifier_regression_rollback():
    """Verify that when a correction introduces regressions, post-verifier reverts."""
    from unittest.mock import MagicMock
    from tacer_rag.correction.post_verifier import PostCorrectionVerifier

    controller = MagicMock()
    # Mock round 1 decision: mutates answer
    controller.correct_answer.return_value = MagicMock(
        corrected_answer="The Eiffel Tower is in Rome, Italy.",
        actions_taken=[],
        abstained=False,
    )

    verifier = PostCorrectionVerifier(
        controller=controller,
        model=None,
        tokenizer=None,
        device="cpu",
        revert_on_regression=True,
    )

    # Monkeypatch predict_single inside post_verifier module
    import tacer_rag.correction.post_verifier as pv_mod
    original_predict = pv_mod.predict_single
    try:
        # Simulate that re-detection finds a brand new hallucination ("Rome, Italy")
        pv_mod.predict_single = lambda *args, **kwargs: [
            PredictedSpan(start=22, end=33, text="Rome, Italy", label_type="EVIDENT_CONFLICT", confidence=0.92)
        ]

        initial_spans = [
            PredictedSpan(start=22, end=37, text="Berlin, Germany", label_type="EVIDENT_CONFLICT", confidence=0.90)
        ]
        res = verifier.verify_and_refine(
            context="The Eiffel Tower is in Paris, France.",
            answer="The Eiffel Tower is in Berlin, Germany.",
            initial_spans=initial_spans,
        )

        assert res.has_regression is True
        assert res.reverted_to_original is True
        assert res.final_answer == "The Eiffel Tower is in Berlin, Germany."
        assert res.new_hallucination_rate == 1.0
    finally:
        pv_mod.predict_single = original_predict

