"""
RAGTruth Dataset Inspection Statistics
Generated: 2026-11-14
Source: external/RAGTruth/dataset/response.jsonl + source_info.jsonl
"""

INSPECTION_STATS = {
    # --- Volume ---
    "n_responses": 17790,
    "n_sources": 2965,
    "responses_per_source": {"min": 6, "max": 6, "mean": 6.0},

    # --- Split ---
    "split_distribution": {"train": 15090, "test": 2700},
    "source_ids_spanning_multiple_splits": 0,  # CLEAN — no leakage in official split

    # --- Models ---
    "model_distribution": {
        "gpt-4-0613": 2965,
        "gpt-3.5-turbo-0613": 2965,
        "mistral-7B-instruct": 2965,
        "llama-2-7b-chat": 2965,
        "llama-2-13b-chat": 2965,
        "llama-2-70b-chat": 2965,
    },

    # --- Labels ---
    # IMPORTANT: actual label_type strings differ from project spec
    # Spec said:  "Evident Baseless Information" and "Subtle Baseless Information"
    # Actual strings: "Evident Baseless Info"      and "Subtle Baseless Info"
    "label_type_distribution": {
        "Evident Conflict": 5324,
        "Subtle Conflict": 201,
        "Evident Baseless Info": 6237,   # NOTE: "Info" not "Information"
        "Subtle Baseless Info": 2527,    # NOTE: "Info" not "Information"
    },
    "total_hallucinated_spans": 14289,
    "responses_with_no_labels_hallucination_free": 10126,

    # --- Offsets ---
    "offset_validation": {
        "ok": 14289,
        "mismatch": 0,
        "out_of_bounds": 0,
    },

    # --- Schema ---
    "response_record_fields": [
        "id", "source_id", "model", "temperature",
        "labels", "split", "quality", "response"
    ],
    "source_info_record_fields": [
        "source_id", "task_type", "source", "source_info", "prompt"
    ],
    # NOTE: source_info uses "source_id" as its key, NOT "id"
    # The join key between response and source_info is: response.source_id == source_info.source_id

    # --- Task types ---
    "task_type_distribution": {
        "Summary": 943,
        "Data2txt": 1033,
        "QA": 989,
    },

    # --- implicit_true semantics ---
    # implicit_true=True: the annotated span is something the LLM "implied"
    # that is technically true in the world but NOT stated in the source context.
    # This is MOST common for Subtle Baseless Info (1861/2527 = 74% of SBI spans).
    # Decision: INCLUDE implicit_true=True spans as hallucinations (they are unsupported
    # by the provided context, which is what matters for RAG faithfulness).
    "implicit_true_distribution": {
        "False": 12361,
        "True": 1928,
    },
    "label_type_x_implicit_true": {
        "Evident Baseless Info | True": 60,
        "Evident Conflict | True": 6,
        "Subtle Baseless Info | True": 1861,   # 74% of SBI!
        "Subtle Conflict | True": 1,
    },

    # --- due_to_null semantics ---
    # due_to_null=True: the hallucination occurred because a JSON field in the source
    # data was null/missing, and the LLM fabricated a value anyway.
    # This is specific to Data2txt task type.
    # Decision: INCLUDE due_to_null=True spans as hallucinations (fabricated from null).
    "due_to_null_distribution": {
        "False": 12647,
        "True": 1642,
    },
    "label_type_x_due_to_null": {
        "Evident Baseless Info | True": 1614,  # Mostly Evident Baseless (from null JSON fields)
        "Evident Conflict | True": 20,
        "Subtle Baseless Info | True": 8,
        "Subtle Conflict | True": 0,
    },

    # --- Data quality field ---
    # RAGTruth baseline filters by quality=='good'
    # We should check how many records have quality != 'good'
    "quality_field_note": "RAGTruth baseline script filters quality=='good'. Check distribution.",
}

# --- Corrected label mapping (use actual strings from dataset) ---
LABEL_MAP = {
    "O": 0,
    "B-EVIDENT_CONFLICT": 1,
    "I-EVIDENT_CONFLICT": 2,
    "B-SUBTLE_CONFLICT": 3,
    "I-SUBTLE_CONFLICT": 4,
    "B-EVIDENT_BASELESS": 5,   # maps from "Evident Baseless Info"
    "I-EVIDENT_BASELESS": 6,
    "B-SUBTLE_BASELESS": 7,    # maps from "Subtle Baseless Info"
    "I-SUBTLE_BASELESS": 8,
}

# Mapping from RAGTruth label_type strings to BIO prefixes
RAGTRUTH_LABEL_TO_BIO_PREFIX = {
    "Evident Conflict": "EVIDENT_CONFLICT",
    "Subtle Conflict": "SUBTLE_CONFLICT",
    "Evident Baseless Info": "EVIDENT_BASELESS",   # actual string — NOT "Information"
    "Subtle Baseless Info": "SUBTLE_BASELESS",     # actual string — NOT "Information"
}

ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
N_LABELS = len(LABEL_MAP)
