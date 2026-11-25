"""
RAGTruth dataset inspection utility.

Run this after cloning the RAGTruth repository to understand
the actual schema before implementing any preprocessing.

Usage:
    python scripts/inspect_data.py
or:
    python -m tacer_rag.data.inspect
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

# Allow running directly before package installation
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tacer_rag.utils import get_logger, load_jsonl, project_root, save_json, ensure_dir

logger = get_logger("inspect")


def inspect_ragtruth(
    response_file: Path,
    source_file: Path,
    output_dir: Path,
) -> dict:
    """
    Load, validate, and report statistics for the RAGTruth dataset.

    Returns a statistics dict that is also saved to output_dir/stats.json.
    """
    logger.info("Loading response file: %s", response_file)
    responses = load_jsonl(response_file)
    logger.info("Loaded %d response records", len(responses))

    logger.info("Loading source info file: %s", source_file)
    sources = load_jsonl(source_file)
    logger.info("Loaded %d source records", len(sources))

    # ------------------------------------------------------------------
    # 1. Field inspection: document all keys present in actual data
    # ------------------------------------------------------------------
    all_response_keys: Counter = Counter()
    for rec in responses:
        all_response_keys.update(rec.keys())

    all_source_keys: Counter = Counter()
    for rec in sources:
        all_source_keys.update(rec.keys())

    logger.info("Response record keys (count of records containing each):")
    for key, cnt in sorted(all_response_keys.items(), key=lambda x: -x[1]):
        logger.info("  %-30s  %d / %d  (%.1f%%)", key, cnt, len(responses), 100*cnt/len(responses))

    logger.info("Source record keys (count of records containing each):")
    for key, cnt in sorted(all_source_keys.items(), key=lambda x: -x[1]):
        logger.info("  %-30s  %d / %d  (%.1f%%)", key, cnt, len(sources), 100*cnt/len(sources))

    # ------------------------------------------------------------------
    # 2. Split distribution
    # ------------------------------------------------------------------
    split_counts: Counter = Counter()
    for rec in responses:
        split = rec.get("split", "MISSING")
        split_counts[str(split)] += 1

    logger.info("Split distribution: %s", dict(split_counts))

    # ------------------------------------------------------------------
    # 3. Label type distribution
    # ------------------------------------------------------------------
    label_type_counts: Counter = Counter()
    implicit_true_counts: Counter = Counter()
    due_to_null_counts: Counter = Counter()
    missing_labels = 0

    for rec in responses:
        labels = rec.get("labels", [])
        if not labels:
            missing_labels += 1
        for lbl in labels:
            lbl_type = lbl.get("label_type", "MISSING")
            label_type_counts[str(lbl_type)] += 1
            impl = lbl.get("implicit_true", None)
            if impl is not None:
                implicit_true_counts[str(impl)] += 1
            dtn = lbl.get("due_to_null", None)
            if dtn is not None:
                due_to_null_counts[str(dtn)] += 1

    logger.info("Label type distribution (across all spans): %s", dict(label_type_counts))
    logger.info("implicit_true values: %s", dict(implicit_true_counts))
    logger.info("due_to_null values: %s", dict(due_to_null_counts))
    logger.info("Responses with no labels (hallucination-free): %d", missing_labels)

    # ------------------------------------------------------------------
    # 4. Source ID relationships
    # ------------------------------------------------------------------
    source_ids_in_responses: set = {rec.get("source_id") for rec in responses}
    source_ids_in_sources: set = {rec.get("id") for rec in sources}

    in_responses_not_sources = source_ids_in_responses - source_ids_in_sources
    in_sources_not_responses = source_ids_in_sources - source_ids_in_responses

    logger.info("Unique source_ids in responses: %d", len(source_ids_in_responses))
    logger.info("Unique source IDs in source_info: %d", len(source_ids_in_sources))
    if in_responses_not_sources:
        logger.warning("source_ids in responses with no matching source_info: %d", len(in_responses_not_sources))
    if in_sources_not_responses:
        logger.info("source_ids in source_info with no responses: %d", len(in_sources_not_responses))

    # Responses per source_id
    responses_per_source: Counter = Counter()
    for rec in responses:
        sid = rec.get("source_id", "MISSING")
        responses_per_source[sid] += 1

    rps_values = list(responses_per_source.values())
    logger.info(
        "Responses per source_id: min=%d max=%d mean=%.1f",
        min(rps_values), max(rps_values), sum(rps_values)/len(rps_values),
    )

    # ------------------------------------------------------------------
    # 5. Duplicate response IDs
    # ------------------------------------------------------------------
    response_ids = [rec.get("id") for rec in responses]
    duplicate_ids = [rid for rid, cnt in Counter(response_ids).items() if cnt > 1]
    if duplicate_ids:
        logger.warning("Duplicate response IDs found: %d", len(duplicate_ids))
    else:
        logger.info("No duplicate response IDs found.")

    # ------------------------------------------------------------------
    # 6. Source_id split integrity
    # ------------------------------------------------------------------
    source_splits: dict[str, set] = defaultdict(set)
    for rec in responses:
        sid = rec.get("source_id", "MISSING")
        spl = str(rec.get("split", "MISSING"))
        source_splits[sid].add(spl)

    multi_split_sources = {sid: splits for sid, splits in source_splits.items() if len(splits) > 1}
    if multi_split_sources:
        logger.warning(
            "source_ids appearing in multiple official splits: %d  (data leakage risk!)",
            len(multi_split_sources),
        )
        for sid, splits in list(multi_split_sources.items())[:5]:
            logger.warning("  source_id=%s  splits=%s", sid, splits)
    else:
        logger.info("Official split is clean: each source_id appears in exactly one split.")

    # ------------------------------------------------------------------
    # 7. Character offset validation
    # ------------------------------------------------------------------
    offset_issues = 0
    for rec in responses:
        text = rec.get("response", "")
        for lbl in rec.get("labels", []):
            start = lbl.get("start")
            end = lbl.get("end")
            span_text = lbl.get("text", "")
            if start is None or end is None:
                offset_issues += 1
                continue
            if end > len(text):
                offset_issues += 1
                continue
            extracted = text[start:end]
            if span_text and extracted != span_text:
                offset_issues += 1

    if offset_issues:
        logger.warning("Annotation offset issues found: %d", offset_issues)
    else:
        logger.info("All annotation offsets validated successfully.")

    # ------------------------------------------------------------------
    # 8. Compile and save statistics
    # ------------------------------------------------------------------
    stats = {
        "n_responses": len(responses),
        "n_sources": len(sources),
        "split_distribution": dict(split_counts),
        "label_type_distribution": dict(label_type_counts),
        "responses_with_no_labels": missing_labels,
        "implicit_true_values": dict(implicit_true_counts),
        "due_to_null_values": dict(due_to_null_counts),
        "unique_source_ids_in_responses": len(source_ids_in_responses),
        "unique_source_ids_in_source_info": len(source_ids_in_sources),
        "source_ids_missing_from_source_info": len(in_responses_not_sources),
        "duplicate_response_ids": len(duplicate_ids),
        "multi_split_sources": len(multi_split_sources),
        "offset_issues": offset_issues,
        "responses_per_source": {
            "min": min(rps_values),
            "max": max(rps_values),
            "mean": round(sum(rps_values)/len(rps_values), 2),
        },
    }

    ensure_dir(output_dir)
    stats_path = output_dir / "inspection_stats.json"
    save_json(stats, stats_path)
    logger.info("Inspection statistics saved to: %s", stats_path)

    return stats


if __name__ == "__main__":
    root = project_root()
    response_file = root / "external" / "RAGTruth" / "data" / "response.jsonl"
    source_file = root / "external" / "RAGTruth" / "data" / "source_info.jsonl"
    output_dir = root / "data" / "processed"

    if not response_file.exists():
        logger.error(
            "RAGTruth response file not found at: %s\n"
            "Run Phase 1 (clone RAGTruth) first:\n"
            "  cd external && git clone https://github.com/ParticleMedia/RAGTruth.git",
            response_file,
        )
        sys.exit(1)

    inspect_ragtruth(response_file, source_file, output_dir)
