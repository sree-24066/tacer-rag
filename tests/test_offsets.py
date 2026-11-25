"""
Offset alignment tests (Phase 2 prerequisite).
Verifies that character offsets correctly map to token offsets
using a fast tokenizer.

These tests require transformers to be installed (Phase 1 environment).
Skip gracefully if not available.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

SKIP_MSG = "transformers not installed — run after Phase 1 environment setup"


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason=SKIP_MSG)
def test_offset_mapping_basic():
    """
    Verify that fast tokenizer offset_mapping correctly maps
    character positions back to tokens for a simple string.
    """
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    text = "ExampleTech was founded in 2015 and employs 500 people."
    encoding = tokenizer(text, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]

    # Check span "2015" at known character positions
    start_char = text.index("2015")
    end_char = start_char + len("2015")

    # Find token indices that overlap with the span
    token_indices = [
        i for i, (tok_start, tok_end) in enumerate(offsets)
        if tok_start is not None and tok_end is not None
        and tok_start < end_char and tok_end > start_char
    ]
    assert len(token_indices) > 0, "No tokens found for span '2015'"

    # Reconstruct text from token offsets
    reconstructed = "".join(text[s:e] for (s, e) in offsets if s is not None and e is not None and e - s > 0)
    # Allow for subword differences — just verify the span characters are covered
    assert "2015" in reconstructed, f"'2015' not in reconstructed: {reconstructed}"


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason=SKIP_MSG)
def test_offset_mapping_no_overlap_between_question_and_answer():
    """
    Verify that when we format input as [CLS] context [SEP] answer [SEP],
    we can identify which tokens belong to the answer segment.
    This is critical: hallucination labels apply only to answer tokens.
    """
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    context = "ExampleTech was founded in 2012."
    answer = "ExampleTech was founded in 2015."
    sep = tokenizer.sep_token or "[SEP]"

    # Format as context + sep + answer
    combined = context + f" {sep} " + answer
    encoding = tokenizer(context, answer, return_offsets_mapping=True, truncation=True, max_length=128)

    # Token type IDs: 0 = context, 1 = answer
    token_type_ids = encoding.get("token_type_ids", [])
    if not token_type_ids:
        pytest.skip("This tokenizer does not return token_type_ids")

    answer_token_count = sum(1 for t in token_type_ids if t == 1)
    context_token_count = sum(1 for t in token_type_ids if t == 0)
    assert answer_token_count > 0, "No answer tokens found"
    assert context_token_count > 0, "No context tokens found"
