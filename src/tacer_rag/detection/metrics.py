"""
Detection metrics: span-level F1, token-level F1, per-category F1,
confusion matrix, macro-F1.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score
)

from tacer_rag.data.dataset import LABEL_MAP, ID_TO_LABEL, IGNORE_INDEX


def compute_token_metrics(
    all_preds: list[int],
    all_labels: list[int],
    label_map: dict = LABEL_MAP,
    id_to_label: dict = ID_TO_LABEL,
) -> dict:
    """
    Compute token-level classification metrics, excluding IGNORE_INDEX tokens.
    Returns precision, recall, F1 per class and macro averages.
    """
    # Filter out ignored tokens
    preds_filtered = []
    labels_filtered = []
    for p, l in zip(all_preds, all_labels):
        if l != IGNORE_INDEX:
            preds_filtered.append(p)
            labels_filtered.append(l)

    if not labels_filtered:
        return {"error": "no valid labels found"}

    label_ids = sorted(label_map.values())
    label_names = [id_to_label[i] for i in label_ids]

    report = classification_report(
        labels_filtered,
        preds_filtered,
        labels=label_ids,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )

    macro_f1 = f1_score(
        labels_filtered, preds_filtered,
        labels=label_ids, average="macro", zero_division=0
    )

    # Binary: hallucinated (any non-O label) vs. supported (O)
    binary_preds  = [0 if p == label_map["O"] else 1 for p in preds_filtered]
    binary_labels = [0 if l == label_map["O"] else 1 for l in labels_filtered]
    binary_f1 = f1_score(binary_labels, binary_preds, average="binary", zero_division=0)

    cm = confusion_matrix(labels_filtered, preds_filtered, labels=label_ids)

    # Per-category F1 (merge B and I for the same type)
    category_preds  = [_bio_to_category(p, id_to_label) for p in preds_filtered]
    category_labels = [_bio_to_category(l, id_to_label) for l in labels_filtered]
    categories = ["O", "EVIDENT_CONFLICT", "SUBTLE_CONFLICT",
                  "EVIDENT_BASELESS", "SUBTLE_BASELESS"]
    cat_report = classification_report(
        category_labels, category_preds,
        labels=categories,
        target_names=categories,
        output_dict=True,
        zero_division=0,
    )

    return {
        "token_macro_f1": round(macro_f1, 4),
        "binary_hallucination_f1": round(binary_f1, 4),
        "per_token_class_report": report,
        "per_category_report": cat_report,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": label_names,
    }


def _bio_to_category(label_id: int, id_to_label: dict) -> str:
    """Convert BIO label ID to category name (strips B-/I- prefix)."""
    name = id_to_label.get(label_id, "O")
    if name.startswith("B-") or name.startswith("I-"):
        return name[2:]
    return name


def span_level_f1(
    pred_spans: list[tuple[int, int, str]],
    gold_spans: list[tuple[int, int, str]],
) -> dict:
    """
    Compute span-level precision, recall, F1.
    A span is a (start_char, end_char, label_type) tuple.
    Exact match required for a true positive.
    """
    pred_set = set(pred_spans)
    gold_set = set(gold_spans)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "span_precision": round(precision, 4),
        "span_recall":    round(recall, 4),
        "span_f1":        round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }
