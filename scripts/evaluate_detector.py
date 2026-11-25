"""
Evaluation script for the TACER-RAG hallucination detector.

Loads a trained detector checkpoint, runs evaluation on the validation or test set,
computes token-level and span-level metrics, and saves predictions and metrics.

Usage:
    python scripts/evaluate_detector.py --run_name debug_ModernBERT-base_ep5_lr2e-05 --split val
    python scripts/evaluate_detector.py --checkpoint experiments/checkpoints/ModernBERT-base_ep5_lr2e-05/best_model.pt --split test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from tacer_rag.utils import get_logger, project_root, save_json, ensure_dir
from tacer_rag.data.dataset import (
    load_examples, RagTruthTokenDataset,
    LABEL_MAP, ID_TO_LABEL, N_LABELS, IGNORE_INDEX
)
from tacer_rag.detection.model import HallucinationDetector, DetectorConfig
from tacer_rag.detection.metrics import compute_token_metrics, span_level_f1
from tacer_rag.detection.predict import extract_spans_from_tokens, PredictedSpan

logger = get_logger("evaluate_detector")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                   help="Path to checkpoint .pt file")
    p.add_argument("--run_name", default=None,
                   help="Run name under experiments/checkpoints/")
    p.add_argument("--split", default="test", choices=["val", "test", "train"],
                   help="Dataset split to evaluate")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--device", default=None)
    return p.parse_args()


def compute_overlap_span_f1(pred_spans: list, gold_spans: list) -> dict:
    """
    Compute span-level F1 with relaxed overlap criterion.
    A predicted span matches a gold span if they overlap and share label_type.
    """
    matched_gold = set()
    tp = 0
    fp = 0

    for p_start, p_end, p_type in pred_spans:
        found_match = False
        for g_idx, (g_start, g_end, g_type) in enumerate(gold_spans):
            if p_type == g_type and max(p_start, g_start) < min(p_end, g_end):
                if g_idx not in matched_gold:
                    matched_gold.add(g_idx)
                    found_match = True
                    break
        if found_match:
            tp += 1
        else:
            fp += 1

    fn = len(gold_spans) - len(matched_gold)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    return {
        "overlap_precision": round(prec, 4),
        "overlap_recall": round(rec, 4),
        "overlap_f1": round(f1, 4),
        "overlap_tp": tp, "overlap_fp": fp, "overlap_fn": fn
    }


def main() -> None:
    args = parse_args()
    root = project_root()

    # Determine checkpoint path
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        run_name = ckpt_path.parent.name
    elif args.run_name:
        run_name = args.run_name
        ckpt_path = root / "experiments" / "checkpoints" / run_name / "best_model.pt"
    else:
        logger.error("Must specify either --checkpoint or --run_name")
        sys.exit(1)

    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    logger.info("Loading checkpoint from: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    saved_args = ckpt.get("args", {})
    encoder_name = saved_args.get("encoder", "answerdotai/ModernBERT-base")
    max_length = args.max_length or saved_args.get("max_length", 512)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Device: %s | Encoder: %s", device, encoder_name)

    # Load Model
    config = DetectorConfig(
        encoder_name=encoder_name,
        num_labels=N_LABELS,
        max_length=max_length,
    )
    model = HallucinationDetector(config)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)
    model.eval()

    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    # Load Split Examples
    split_file = root / "data" / "processed" / f"{args.split}.jsonl"
    if not split_file.exists():
        logger.error("Data split file not found: %s", split_file)
        sys.exit(1)

    logger.info("Loading %s data from: %s", args.split, split_file)
    examples = load_examples(split_file)
    logger.info("Loaded %d examples", len(examples))

    # Tokenize and create DataLoader
    trunc_stats = {"dropped_spans": 0, "total_spans": 0}
    dataset = RagTruthTokenDataset(examples, tokenizer, max_length, trunc_stats)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Evaluation loop
    all_preds = []
    all_labels = []
    all_probs = []
    total_loss = 0.0
    n_batches = 0

    logger.info("Running inference on %s set (%d batches)...", args.split, len(loader))
    t0 = time.time()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels,
            )
            total_loss += outputs.loss.item()
            n_batches += 1

            probs = torch.softmax(outputs.logits, dim=-1)
            preds = probs.argmax(dim=-1)
            max_p = probs.max(dim=-1).values

            all_preds.extend(preds.view(-1).cpu().tolist())
            all_labels.extend(labels.view(-1).cpu().tolist())
            all_probs.extend(max_p.view(-1).cpu().tolist())

    elapsed = time.time() - t0
    logger.info("Inference complete in %.1fs", elapsed)

    # Token-level metrics
    token_metrics = compute_token_metrics(all_preds, all_labels)
    token_metrics["eval_loss"] = round(total_loss / max(n_batches, 1), 4)

    # Span-level reconstruction & metrics
    logger.info("Reconstructing predicted spans from token outputs...")
    pred_records = []
    all_gold_spans = []
    all_pred_spans = []

    # Map dataset item predictions back to character spans per example
    pred_ptr = 0
    for ex in examples:
        # Standalone answer encoding for character offsets
        answer_enc = tokenizer(
            ex.answer,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        answer_offsets = answer_enc["offset_mapping"]

        # Number of tokens evaluated for this example is seq_len
        item_labels = all_labels[pred_ptr : pred_ptr + max_length]
        item_preds  = all_preds[pred_ptr : pred_ptr + max_length]
        item_probs  = all_probs[pred_ptr : pred_ptr + max_length]
        pred_ptr += max_length

        # Find answer tokens (non-ignore)
        ans_preds = []
        ans_probs = []
        valid_offsets = []

        off_idx = 0
        for l, p, prob in zip(item_labels, item_preds, item_probs):
            if l != IGNORE_INDEX:
                if off_idx < len(answer_offsets):
                    c_start, c_end = answer_offsets[off_idx]
                    if c_end > 0:
                        ans_preds.append(p)
                        ans_probs.append(prob)
                        valid_offsets.append((c_start, c_end))
                    off_idx += 1

        predicted_spans = extract_spans_from_tokens(
            ans_preds, ans_probs, valid_offsets, ex.answer
        )

        gold_spans_list = [
            (s, e, t) for s, e, t in ex.hallucinated_spans
        ]
        pred_spans_list = [
            (ps.start, ps.end, ps.label_type) for ps in predicted_spans
        ]

        all_gold_spans.extend(gold_spans_list)
        all_pred_spans.extend(pred_spans_list)

        pred_records.append({
            "record_id": ex.record_id,
            "source_id": ex.source_id,
            "model": ex.model,
            "task_type": ex.task_type,
            "answer": ex.answer,
            "predicted_spans": [
                {
                    "start": ps.start,
                    "end": ps.end,
                    "text": ps.text,
                    "label_type": ps.label_type,
                    "confidence": ps.confidence,
                }
                for ps in predicted_spans
            ],
            "gold_spans": [
                {"start": s, "end": e, "label_type": t}
                for s, e, t in ex.hallucinated_spans
            ],
        })

    # Span metrics: Exact & Overlap
    exact_span_metrics = span_level_f1(all_pred_spans, all_gold_spans)
    overlap_span_metrics = compute_overlap_span_f1(all_pred_spans, all_gold_spans)

    combined_metrics = {
        "run_name": run_name,
        "split": args.split,
        "encoder": encoder_name,
        "checkpoint": str(ckpt_path),
        "total_examples": len(examples),
        "token_metrics": token_metrics,
        "exact_span_metrics": exact_span_metrics,
        "overlap_span_metrics": overlap_span_metrics,
        "truncation_stats": trunc_stats,
    }

    # Save outputs
    pred_dir = ensure_dir(root / "experiments" / "predictions")
    metrics_dir = ensure_dir(root / "experiments" / "metrics")

    pred_out_path = pred_dir / f"{run_name}_{args.split}_preds.jsonl"
    with open(pred_out_path, "w", encoding="utf-8") as f:
        for r in pred_records:
            f.write(json.dumps(r) + "\n")
    logger.info("Saved predictions to: %s", pred_out_path)

    metrics_out_path = metrics_dir / f"{run_name}_{args.split}_metrics.json"
    save_json(combined_metrics, metrics_out_path)
    logger.info("Saved metrics to: %s", metrics_out_path)

    # Print summary report
    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY: {run_name} [{args.split.upper()} SET]")
    print("=" * 60)
    print(f"  Encoder:               {encoder_name}")
    print(f"  Total examples:        {len(examples)}")
    print(f"  Eval Loss:             {token_metrics['eval_loss']:.4f}")
    print(f"  Token Macro-F1:        {token_metrics['token_macro_f1']:.4f}")
    print(f"  Binary Hallucination F1:{token_metrics['binary_hallucination_f1']:.4f}")
    print("-" * 60)
    print("  Per-Category F1 (token-level):")
    cat_report = token_metrics.get("per_category_report", {})
    for cat in ["EVIDENT_CONFLICT", "SUBTLE_CONFLICT", "EVIDENT_BASELESS", "SUBTLE_BASELESS"]:
        score = cat_report.get(cat, {}).get("f1-score", 0.0)
        support = cat_report.get(cat, {}).get("support", 0)
        print(f"    {cat:<22} F1: {score:.4f}  (tokens: {support})")
    print("-" * 60)
    print(f"  Exact Span F1:         {exact_span_metrics['span_f1']:.4f} (P: {exact_span_metrics['span_precision']:.4f}, R: {exact_span_metrics['span_recall']:.4f})")
    print(f"  Overlap Span F1:       {overlap_span_metrics['overlap_f1']:.4f} (P: {overlap_span_metrics['overlap_precision']:.4f}, R: {overlap_span_metrics['overlap_recall']:.4f})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
