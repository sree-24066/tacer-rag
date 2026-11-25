"""
Phase 0 test: synthetic test cases verifying that the
test harness can run before any model is trained.

These tests do NOT require PyTorch or downloaded data.
They verify:
  - Project imports work
  - Utility functions work
  - Synthetic test case schema is correct
  - Offset calculations are sane

Run:
    pytest tests/test_data.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Synthetic test cases (Section 13 of the project specification)
# ---------------------------------------------------------------------------

SYNTHETIC_EXAMPLES = [
    {
        "id": "synth_001",
        "category": "evident_conflict",
        "question": "When was ExampleTech founded?",
        "context": "ExampleTech was founded in 2012. It is headquartered in Bengaluru.",
        "answer": "ExampleTech was founded in 2015 and employs 500 people.",
        "hallucinated_spans": [
            {"text": "2015", "type": "Evident Conflict"},
            {"text": "employs 500 people", "type": "Evident Baseless Information"},
        ],
        "expected_correction": "ExampleTech was founded in 2012.",
        "notes": (
            "Year 2015 directly contradicts the context (2012). "
            "Employee count is unsupported and should not be invented."
        ),
    },
    {
        "id": "synth_002",
        "category": "subtle_conflict",
        "question": "What is the boiling point of water at sea level?",
        "context": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at standard sea-level pressure.",
        "answer": "Water boils at approximately 98 degrees Celsius at sea level.",
        "hallucinated_spans": [
            {"text": "approximately 98 degrees Celsius", "type": "Subtle Conflict"},
        ],
        "expected_correction": "Water boils at 100 degrees Celsius at sea level.",
        "notes": (
            "The answer subtly understates the boiling point by 2 degrees. "
            "This is a subtle conflict, not an obvious contradiction."
        ),
    },
    {
        "id": "synth_003",
        "category": "evident_baseless",
        "question": "What products does Acme Corp sell?",
        "context": "Acme Corp is a manufacturing company based in Chicago, founded in 1998.",
        "answer": "Acme Corp sells industrial tools, safety equipment, and chemical solvents.",
        "hallucinated_spans": [
            {
                "text": "sells industrial tools, safety equipment, and chemical solvents",
                "type": "Evident Baseless Information",
            },
        ],
        "expected_correction": "Acme Corp is a manufacturing company based in Chicago, founded in 1998. Its specific products are not mentioned in the provided context.",
        "notes": (
            "No products are mentioned in the context. "
            "The correction should not invent products."
        ),
    },
    {
        "id": "synth_004",
        "category": "subtle_baseless",
        "question": "Is the CEO of TechFlow popular with employees?",
        "context": "TechFlow's CEO Sarah Johnson has led the company since 2019 and oversaw a major product launch in 2022.",
        "answer": "Sarah Johnson, the CEO of TechFlow, is widely respected and loved by all employees.",
        "hallucinated_spans": [
            {
                "text": "is widely respected and loved by all employees",
                "type": "Subtle Baseless Information",
            },
        ],
        "expected_correction": "Sarah Johnson has been TechFlow's CEO since 2019 and oversaw a major product launch in 2022.",
        "notes": (
            "The claim about employee sentiment is an unsupported inference. "
            "The context only describes her tenure and a product launch."
        ),
    },
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_project_imports():
    """Verify that the tacer_rag package can be imported."""
    import tacer_rag  # noqa: F401
    assert tacer_rag.__version__ == "0.1.0"


def test_utils_set_seed():
    """Verify set_seed runs without error."""
    from tacer_rag.utils import set_seed
    set_seed(42)  # Should not raise


def test_utils_project_root():
    """Verify project root resolves to a real directory."""
    from tacer_rag.utils import project_root
    root = project_root()
    assert root.exists(), f"Project root not found: {root}"
    assert (root / "README.md").exists(), "README.md missing from project root"


def test_utils_ensure_dir(tmp_path):
    """Verify ensure_dir creates a directory."""
    from tacer_rag.utils import ensure_dir
    new_dir = tmp_path / "test_subdir" / "nested"
    result = ensure_dir(new_dir)
    assert result.exists()
    assert result.is_dir()


def test_utils_save_load_json(tmp_path):
    """Verify JSON round-trip."""
    from tacer_rag.utils import save_json, load_json
    data = {"key": "value", "number": 42, "list": [1, 2, 3]}
    path = tmp_path / "test.json"
    save_json(data, path)
    loaded = load_json(path)
    assert loaded == data


def test_utils_save_load_jsonl(tmp_path):
    """Verify JSONL round-trip."""
    from tacer_rag.utils import save_jsonl, load_jsonl
    records = [{"id": i, "text": f"record {i}"} for i in range(5)]
    path = tmp_path / "test.jsonl"
    save_jsonl(records, path)
    loaded = load_jsonl(path)
    assert loaded == records


def test_synthetic_examples_schema():
    """Verify synthetic test examples have required fields."""
    required_fields = ["id", "question", "context", "answer",
                       "hallucinated_spans", "expected_correction"]
    for ex in SYNTHETIC_EXAMPLES:
        for field in required_fields:
            assert field in ex, f"Example {ex.get('id')} missing field: {field}"
        for span in ex["hallucinated_spans"]:
            assert "text" in span
            assert "type" in span


def test_synthetic_span_offsets():
    """Verify that each hallucinated span text is actually found in the answer."""
    for ex in SYNTHETIC_EXAMPLES:
        for span in ex["hallucinated_spans"]:
            assert span["text"] in ex["answer"], (
                f"Span '{span['text']}' not found in answer of example {ex['id']}\n"
                f"Answer: {ex['answer']}"
            )


def test_synthetic_categories_cover_all_types():
    """Verify that synthetic examples cover all four hallucination types."""
    all_types = set()
    for ex in SYNTHETIC_EXAMPLES:
        for span in ex["hallucinated_spans"]:
            all_types.add(span["type"])
    expected = {
        "Evident Conflict",
        "Subtle Conflict",
        "Evident Baseless Information",
        "Subtle Baseless Information",
    }
    assert all_types == expected, f"Missing types: {expected - all_types}"


def test_no_invention_in_baseless_corrections():
    """
    Verify that expected corrections for baseless examples do not
    invent facts not present in the context.
    This is a schema check, not a model evaluation.
    """
    for ex in SYNTHETIC_EXAMPLES:
        if "baseless" in ex["category"]:
            # Basic check: correction should not contain hallucinated span text
            for span in ex["hallucinated_spans"]:
                assert span["text"] not in ex["expected_correction"] or "not mentioned" in ex["expected_correction"].lower(), (
                    f"Expected correction for {ex['id']} still contains hallucinated span text: {span['text']}"
                )
