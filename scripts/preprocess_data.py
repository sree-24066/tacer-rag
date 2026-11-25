"""
RAGTruth Preprocessing Script — Phase 2

Reads raw RAGTruth JSONL files, validates them, creates train/val/test splits
(grouped by source_id to prevent leakage), and saves processed files.

Usage:
    python scripts/preprocess_data.py

Outputs (in data/processed/):
    train.jsonl       — training split
    val.jsonl         — validation split (carved from train, by source_id)
    test.jsonl        — official test split
    label_map.json    — label string → BIO tag ID mapping
    split_summary.json — counts, distributions, source_id lists
    excluded.jsonl     — records excluded and why

Design decisions recorded here (from Phase 1 inspection, 2026-11-14):
    - Join key is source_id (NOT "id") between response and source_info
    - Actual label strings use "Info" not "Information"
    - quality filter: keep only quality=="good" (removes 173 records)
    - implicit_true=True: INCLUDED as hallucinations (unsupported by context)
    - due_to_null=True: INCLUDED as hallucinations (fabricated null field value)
    - Val split: 50 source_ids per task type (150 total) → ~900 responses
    - Official split has ZERO source_id leakage between train/test
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from tacer_rag.utils import (
    get_logger, load_jsonl, save_jsonl, save_json,
    project_root, ensure_dir
)

logger = get_logger("preprocess")

# ── Label mapping (verified against actual dataset strings 2026-11-14) ──────
RAGTRUTH_LABEL_TO_BIO = {
    "Evident Conflict":       "EVIDENT_CONFLICT",
    "Subtle Conflict":        "SUBTLE_CONFLICT",
    "Evident Baseless Info":  "EVIDENT_BASELESS",
    "Subtle Baseless Info":   "SUBTLE_BASELESS",
}

BIO_TAG_MAP = {
    "O": 0,
    "B-EVIDENT_CONFLICT": 1, "I-EVIDENT_CONFLICT": 2,
    "B-SUBTLE_CONFLICT":  3, "I-SUBTLE_CONFLICT":  4,
    "B-EVIDENT_BASELESS": 5, "I-EVIDENT_BASELESS": 6,
    "B-SUBTLE_BASELESS":  7, "I-SUBTLE_BASELESS":  8,
}

VAL_SOURCE_IDS_PER_TASK = 50   # → 300 responses per task type for val
VAL_SEED = 42
QUALITY_FILTER = "good"


def load_and_join(response_file: Path, source_file: Path) -> tuple[list, list]:
    """
    Load and join response + source_info on source_id.
    Returns (joined_records, excluded_records).
    """
    logger.info("Loading response file: %s", response_file)
    responses = load_jsonl(response_file)
    logger.info("Loaded %d response records", len(responses))

    logger.info("Loading source_info file: %s", source_file)
    sources = load_jsonl(source_file)
    logger.info("Loaded %d source records", len(sources))

    # Build source lookup (key = source_id)
    source_lookup: dict = {}
    for src in sources:
        sid = src.get("source_id")
        if sid is None:
            logger.warning("source_info record missing source_id: %s", src)
            continue
        source_lookup[sid] = src

    joined = []
    excluded = []

    for rec in responses:
        reason = None

        # Quality filter
        if rec.get("quality") != QUALITY_FILTER:
            reason = f"quality={rec.get('quality')}"

        # Missing source
        elif rec.get("source_id") not in source_lookup:
            reason = f"source_id={rec.get('source_id')} not in source_info"

        if reason:
            excluded.append({"record": rec, "reason": reason})
            continue

        # Merge source info into record
        src = source_lookup[rec["source_id"]]
        merged = {**rec, "task_type": src["task_type"],
                  "source": src["source"],
                  "source_info": src["source_info"],
                  "prompt": src["prompt"]}
        joined.append(merged)

    logger.info("Joined records: %d | Excluded: %d", len(joined), len(excluded))
    return joined, excluded


def validate_offsets(records: list) -> dict:
    """Validate that all annotation offsets match the span text field."""
    ok = mismatch = oob = 0
    for rec in records:
        text = rec.get("response", "")
        for lbl in rec.get("labels", []):
            s, e = lbl.get("start", 0), lbl.get("end", 0)
            span_text = lbl.get("text", "")
            if e > len(text):
                oob += 1
            elif text[s:e] == span_text:
                ok += 1
            else:
                mismatch += 1
    return {"ok": ok, "mismatch": mismatch, "out_of_bounds": oob}


def make_val_split(train_records: list, seed: int = VAL_SEED,
                   n_per_task: int = VAL_SOURCE_IDS_PER_TASK
                   ) -> tuple[list, list]:
    """
    Carve a validation split out of train records, grouped by source_id.
    Strategy mirrors the RAGTruth baseline: 50 source_ids per task type.
    Returns (train_records_remaining, val_records).
    """
    rng = random.Random(seed)

    # Group source_ids by task_type
    task_sources: dict[str, list] = defaultdict(list)
    seen_sources = set()
    for rec in train_records:
        sid = rec["source_id"]
        task = rec.get("task_type", "Unknown")
        if sid not in seen_sources:
            task_sources[task].append(sid)
            seen_sources.add(sid)

    val_source_ids: set = set()
    for task, source_ids in task_sources.items():
        n = min(n_per_task, len(source_ids))
        sampled = rng.sample(source_ids, n)
        val_source_ids.update(sampled)
        logger.info("Val split: task=%s sampled %d / %d source_ids", task, n, len(source_ids))

    train_out = [r for r in train_records if r["source_id"] not in val_source_ids]
    val_out   = [r for r in train_records if r["source_id"] in val_source_ids]

    logger.info("After val split: train=%d val=%d", len(train_out), len(val_out))
    return train_out, val_out


def label_distribution(records: list) -> dict:
    """Count label_type distribution across records."""
    counts: Counter = Counter()
    for rec in records:
        for lbl in rec.get("labels", []):
            counts[lbl.get("label_type", "UNKNOWN")] += 1
    return dict(counts)


def preprocess(root: Path | None = None) -> None:
    if root is None:
        root = project_root()

    response_file = root / "external" / "RAGTruth" / "dataset" / "response.jsonl"
    source_file   = root / "external" / "RAGTruth" / "dataset" / "source_info.jsonl"
    out_dir = ensure_dir(root / "data" / "processed")

    # 1. Load and join
    all_records, excluded = load_and_join(response_file, source_file)

    # 2. Validate offsets
    offset_stats = validate_offsets(all_records)
    logger.info("Offset validation: %s", offset_stats)
    if offset_stats["mismatch"] > 0 or offset_stats["out_of_bounds"] > 0:
        logger.warning("Offset issues detected — investigate before training!")

    # 3. Split by official split field
    train_all = [r for r in all_records if r.get("split") == "train"]
    test_all  = [r for r in all_records if r.get("split") == "test"]
    logger.info("Official splits: train=%d test=%d", len(train_all), len(test_all))

    # 4. Carve validation split from train (by source_id)
    train_final, val_final = make_val_split(train_all)

    # 5. Leakage check
    train_sids = {r["source_id"] for r in train_final}
    val_sids   = {r["source_id"] for r in val_final}
    test_sids  = {r["source_id"] for r in test_all}
    tv_overlap = train_sids & val_sids
    vt_overlap = val_sids & test_sids
    tt_overlap = train_sids & test_sids
    if tv_overlap or vt_overlap or tt_overlap:
        raise RuntimeError(
            f"DATA LEAKAGE DETECTED! "
            f"train∩val={len(tv_overlap)} val∩test={len(vt_overlap)} train∩test={len(tt_overlap)}"
        )
    logger.info("Leakage check PASSED — no source_id shared across splits.")

    # 6. Save splits
    save_jsonl(train_final, out_dir / "train.jsonl")
    save_jsonl(val_final,   out_dir / "val.jsonl")
    save_jsonl(test_all,    out_dir / "test.jsonl")
    save_jsonl(excluded,    out_dir / "excluded.jsonl")
    logger.info("Saved train/val/test/excluded to: %s", out_dir)

    # 7. Save label map
    save_json(BIO_TAG_MAP, out_dir / "label_map.json")
    save_json(RAGTRUTH_LABEL_TO_BIO, out_dir / "ragtruth_label_to_bio.json")

    # 8. Summary statistics
    summary = {
        "generated_at": "2026-11-14",
        "n_total_after_quality_filter": len(all_records),
        "n_excluded": len(excluded),
        "exclusion_reasons": dict(Counter(e["reason"].split("=")[0] for e in excluded)),
        "splits": {
            "train": {"n_responses": len(train_final),
                      "n_source_ids": len(train_sids),
                      "label_distribution": label_distribution(train_final),
                      "task_distribution": dict(Counter(r.get("task_type") for r in train_final))},
            "val":   {"n_responses": len(val_final),
                      "n_source_ids": len(val_sids),
                      "label_distribution": label_distribution(val_final),
                      "task_distribution": dict(Counter(r.get("task_type") for r in val_final))},
            "test":  {"n_responses": len(test_all),
                      "n_source_ids": len(test_sids),
                      "label_distribution": label_distribution(test_all),
                      "task_distribution": dict(Counter(r.get("task_type") for r in test_all))},
        },
        "offset_validation": offset_stats,
        "leakage_check": {
            "train_val_overlap": len(tv_overlap),
            "val_test_overlap": len(vt_overlap),
            "train_test_overlap": len(tt_overlap),
            "passed": True,
        },
        "label_map": BIO_TAG_MAP,
        "ragtruth_label_to_bio": RAGTRUTH_LABEL_TO_BIO,
        "design_decisions": {
            "implicit_true": "INCLUDED as hallucinations — unsupported by context regardless of world truth",
            "due_to_null": "INCLUDED as hallucinations — LLM fabricated value for null JSON field",
            "quality_filter": f"Only quality=='{QUALITY_FILTER}' records retained",
            "val_strategy": f"{VAL_SOURCE_IDS_PER_TASK} source_ids per task type, seed={VAL_SEED}",
            "label_strings": "Actual strings use 'Info' not 'Information' (verified 2026-11-14)",
        },
    }
    save_json(summary, out_dir / "split_summary.json")
    logger.info("Summary saved to: %s", out_dir / "split_summary.json")

    # 9. Print report
    print("\n" + "="*65)
    print("PREPROCESSING COMPLETE")
    print("="*65)
    print(f"  Total (after quality filter): {len(all_records)}")
    print(f"  Excluded:                     {len(excluded)}")
    print(f"  Train:  {len(train_final)} responses / {len(train_sids)} source_ids")
    print(f"  Val:    {len(val_final)} responses   / {len(val_sids)} source_ids")
    print(f"  Test:   {len(test_all)} responses   / {len(test_sids)} source_ids")
    print(f"  Offset validation:            {offset_stats}")
    print(f"  Leakage:                      NONE ✅")
    print(f"\n  Label distribution (train):")
    for lt, cnt in label_distribution(train_final).items():
        print(f"    {lt:<30} {cnt}")
    print("="*65)


if __name__ == "__main__":
    preprocess()
