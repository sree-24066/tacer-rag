"""
Full training script for the TACER-RAG hallucination detector.

Trains a ModernBERT (or any BERT-family) encoder on RAGTruth for
token-level hallucination detection with 9 BIO classes.

Usage:
    # Full ModernBERT-base training
    python scripts/train_detector.py

    # Quick debug run on tiny subset
    python scripts/train_detector.py --debug

    # Custom encoder
    python scripts/train_detector.py --encoder bert-base-uncased

Outputs saved to experiments/checkpoints/<run_name>/
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

# Suppress harmless symlink warning on Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from tacer_rag.utils import get_logger, project_root, set_seed, save_json, ensure_dir
from tacer_rag.data.dataset import (
    load_examples, RagTruthTokenDataset,
    LABEL_MAP, ID_TO_LABEL, N_LABELS
)
from tacer_rag.detection.model import HallucinationDetector, DetectorConfig
from tacer_rag.detection.metrics import compute_token_metrics

logger = get_logger("train_detector")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default="answerdotai/ModernBERT-base",
                   help="HuggingFace encoder model ID")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4,
                   help="Gradient accumulation steps (effective batch = batch_size × grad_accum)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", default=True,
                   help="Use mixed precision fp16 training")
    p.add_argument("--debug", action="store_true",
                   help="Debug mode: tiny subset, 1 epoch")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit number of training samples (e.g. 2000)")
    p.add_argument("--run_name", default=None,
                   help="Experiment name (auto-generated if not set)")
    return p.parse_args()


def evaluate(model, loader, device) -> dict:
    """Run evaluation and return token metrics dict."""
    import torch
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
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
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.view(-1).cpu().tolist())
            all_labels.extend(labels.view(-1).cpu().tolist())
    metrics = compute_token_metrics(all_preds, all_labels)
    metrics["eval_loss"] = round(total_loss / max(n_batches, 1), 4)
    return metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    root = project_root()

    # ── Run name ─────────────────────────────────────────────────────────────
    encoder_short = args.encoder.split("/")[-1]
    run_name = args.run_name or f"{encoder_short}_ep{args.epochs}_lr{args.lr:.0e}"
    if args.debug:
        run_name = f"debug_{run_name}"

    out_dir = ensure_dir(root / "experiments" / "checkpoints" / run_name)
    log_dir = ensure_dir(root / "experiments" / "logs")
    pred_dir = ensure_dir(root / "experiments" / "predictions")
    metrics_dir = ensure_dir(root / "experiments" / "metrics")

    logger.info("=" * 60)
    logger.info("TACER-RAG Hallucination Detector Training")
    logger.info("Run: %s", run_name)
    logger.info("Encoder: %s", args.encoder)
    logger.info("=" * 60)

    # ── Imports ───────────────────────────────────────────────────────────────
    import torch
    from torch.utils.data import DataLoader
    from torch.amp import GradScaler, autocast
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = args.fp16 and device.type == "cuda"
    logger.info("Device: %s | fp16: %s", device, use_fp16)

    if device.type == "cuda":
        logger.info("GPU: %s | VRAM: %.1f GB",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    # ── Load data ─────────────────────────────────────────────────────────────
    train_file = root / "data" / "processed" / "train.jsonl"
    val_file   = root / "data" / "processed" / "val.jsonl"

    logger.info("Loading training data...")
    train_examples = load_examples(train_file)
    val_examples   = load_examples(val_file)

    if args.debug:
        logger.info("DEBUG MODE: using 100 train / 50 val examples")
        train_examples = train_examples[:100]
        val_examples   = val_examples[:50]
        args.epochs = 1
    elif args.max_samples:
        logger.info("Sampling %d training examples", args.max_samples)
        train_examples = train_examples[:args.max_samples]

    logger.info("Train: %d examples | Val: %d examples",
                len(train_examples), len(val_examples))

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s", args.encoder)
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)

    trunc_stats = {"dropped_spans": 0, "total_spans": 0}
    logger.info("Tokenizing training set...")
    train_dataset = RagTruthTokenDataset(
        train_examples, tokenizer, args.max_length, trunc_stats
    )
    logger.info("Tokenizing validation set...")
    val_dataset = RagTruthTokenDataset(
        val_examples, tokenizer, args.max_length, trunc_stats
    )
    logger.info("Truncation stats: %s", trunc_stats)
    logger.info("Train dataset: %d | Val dataset: %d",
                len(train_dataset), len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size * 2,
                              shuffle=False, num_workers=0, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Building model: %s", args.encoder)
    config = DetectorConfig(
        encoder_name=args.encoder,
        num_labels=N_LABELS,
        max_length=args.max_length,
    )
    model = HallucinationDetector(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable parameters: %.1f M", n_params / 1e6)

    # ── Optimizer & Scheduler ────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    scaler = GradScaler("cuda") if use_fp16 else None
    logger.info("Total steps: %d | Warmup: %d", total_steps, warmup_steps)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_f1 = -1.0
    best_epoch  = -1
    history = []

    save_json(vars(args), out_dir / "training_args.json")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            if use_fp16:
                with autocast("cuda"):
                    outputs = model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids,
                                    labels=labels)
                loss = outputs.loss / args.grad_accum
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                labels=labels)
                loss = outputs.loss / args.grad_accum
                loss.backward()

            epoch_loss += outputs.loss.item()

            if (step + 1) % args.grad_accum == 0:
                if use_fp16:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % 50 == 0:
                logger.info("Epoch %d | Step %d/%d | Loss %.4f",
                            epoch, step + 1, len(train_loader),
                            epoch_loss / (step + 1))

        avg_train_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - t0
        logger.info("Epoch %d done in %.1fs | Avg train loss: %.4f",
                    epoch, elapsed, avg_train_loss)

        # ── Validation ───────────────────────────────────────────────────────
        logger.info("Running validation...")
        val_metrics = evaluate(model, val_loader, device)
        val_f1 = val_metrics["token_macro_f1"]
        val_loss = val_metrics["eval_loss"]

        logger.info("Epoch %d | Val loss: %.4f | Val macro-F1: %.4f | Binary-F1: %.4f",
                    epoch, val_loss, val_f1, val_metrics["binary_hallucination_f1"])

        # Per-category F1
        cat_report = val_metrics.get("per_category_report", {})
        for cat in ["EVIDENT_CONFLICT", "SUBTLE_CONFLICT",
                    "EVIDENT_BASELESS", "SUBTLE_BASELESS"]:
            cat_f1 = cat_report.get(cat, {}).get("f1-score", 0.0)
            logger.info("  %-20s F1: %.4f", cat, cat_f1)

        epoch_record = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
            "val_binary_f1": val_metrics["binary_hallucination_f1"],
            "elapsed_s": round(elapsed, 1),
        }
        history.append(epoch_record)

        # Save checkpoint if best
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            ckpt_path = out_dir / "best_model.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_macro_f1": val_f1,
                "args": vars(args),
            }, ckpt_path)
            logger.info("✅ New best model saved (epoch %d, macro-F1=%.4f)",
                        epoch, val_f1)
            # Save full metrics
            save_json(val_metrics, metrics_dir / f"{run_name}_best_val_metrics.json")

        # Always save latest checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_macro_f1": val_f1,
        }, out_dir / "latest_model.pt")

    # ── Training complete ────────────────────────────────────────────────────
    save_json(history, metrics_dir / f"{run_name}_training_history.json")

    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("Best epoch: %d | Best val macro-F1: %.4f", best_epoch, best_val_f1)
    logger.info("Checkpoint: %s", out_dir / "best_model.pt")
    logger.info("=" * 60)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Encoder:         {args.encoder}")
    print(f"  Best epoch:      {best_epoch} / {args.epochs}")
    print(f"  Best val F1:     {best_val_f1:.4f}")
    print(f"  Checkpoint:      {out_dir / 'best_model.pt'}")
    print(f"{'='*60}")
    print(f"\nNext: python scripts/evaluate_detector.py --run_name {run_name}")


if __name__ == "__main__":
    main()
