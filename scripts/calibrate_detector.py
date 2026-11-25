"""
Calibration script for the TACER-RAG hallucination detector.

Calibrates model confidence on the validation set using Temperature Scaling (Guo et al., 2017),
computes ECE and Brier scores, plots reliability diagrams, and selects confidence thresholds.

Usage:
    python scripts/calibrate_detector.py --run_name debug_ModernBERT-base_ep5_lr2e-05
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

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from tacer_rag.utils import get_logger, project_root, save_json, ensure_dir
from tacer_rag.data.dataset import (
    load_examples, RagTruthTokenDataset,
    LABEL_MAP, ID_TO_LABEL, N_LABELS, IGNORE_INDEX
)
from tacer_rag.detection.model import HallucinationDetector, DetectorConfig
from tacer_rag.calibration.calibrate import (
    TemperatureScaler, compute_ece, compute_brier_score,
    plot_reliability_diagram, select_confidence_thresholds
)

logger = get_logger("calibrate_detector")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--device", default=None)
    return p.parse_args()


def get_logits_and_labels(model, loader, device):
    """Run model on loader and collect all token logits and labels."""
    model.eval()
    all_logits = []
    all_labels = []

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
            )
            all_logits.append(outputs.logits.view(-1, N_LABELS).cpu())
            all_labels.append(labels.view(-1).cpu())

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0)


def main() -> None:
    args = parse_args()
    root = project_root()

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

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    logger.info("Loading checkpoint: %s (device: %s)", ckpt_path, device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    saved_args = ckpt.get("args", {})
    encoder_name = saved_args.get("encoder", "answerdotai/ModernBERT-base")

    config = DetectorConfig(
        encoder_name=encoder_name,
        num_labels=N_LABELS,
        max_length=args.max_length,
    )
    model = HallucinationDetector(config)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(encoder_name)

    # 1. Collect validation set logits and labels
    val_file = root / "data" / "processed" / "val.jsonl"
    logger.info("Loading validation data: %s", val_file)
    val_examples = load_examples(val_file)
    val_dataset = RagTruthTokenDataset(val_examples, tokenizer, args.max_length)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info("Extracting validation logits (%d examples)...", len(val_examples))
    val_logits, val_labels = get_logits_and_labels(model, val_loader, device)

    # Filter out ignore_index tokens (-100)
    valid_mask = val_labels != IGNORE_INDEX
    val_logits_filt = val_logits[valid_mask]
    val_labels_filt = val_labels[valid_mask]

    logger.info("Valid token count for calibration: %d", len(val_labels_filt))

    # 2. Compute uncalibrated metrics
    uncal_probs = torch.softmax(val_logits_filt, dim=-1).numpy()
    val_labels_np = val_labels_filt.numpy()

    uncal_ece_dict = compute_ece(uncal_probs, val_labels_np, n_bins=15)
    uncal_brier = compute_brier_score(uncal_probs, val_labels_np, num_classes=N_LABELS)

    logger.info("Uncalibrated Validation ECE: %.4f | Brier: %.4f",
                uncal_ece_dict["ece"], uncal_brier)

    # 3. Fit Temperature Scaler
    scaler = TemperatureScaler()
    optimal_temp = scaler.fit(val_logits_filt, val_labels_filt)

    # 4. Compute calibrated metrics
    cal_probs = scaler.scale_probabilities(val_logits_filt).cpu().numpy()
    cal_ece_dict = compute_ece(cal_probs, val_labels_np, n_bins=15)
    cal_brier = compute_brier_score(cal_probs, val_labels_np, num_classes=N_LABELS)

    logger.info("Calibrated Validation ECE:   %.4f | Brier: %.4f | Temp: %.4f",
                cal_ece_dict["ece"], cal_brier, optimal_temp)

    # 5. Plot reliability diagram
    figures_dir = ensure_dir(root / "experiments" / "figures")
    diagram_path = figures_dir / f"{run_name}_reliability_diagram.png"
    plot_reliability_diagram(uncal_ece_dict, cal_ece_dict, diagram_path,
                             title_suffix=f"({run_name})")

    # 6. Select validation confidence operating thresholds
    hal_ids = [v for k, v in LABEL_MAP.items() if k != "O"]
    thresholds = select_confidence_thresholds(cal_probs, val_labels_np, hal_ids)
    logger.info("Validation-selected thresholds: %s", thresholds)

    # 7. Save calibration parameters
    calibration_output = {
        "run_name": run_name,
        "encoder": encoder_name,
        "temperature": round(optimal_temp, 4),
        "validation_metrics": {
            "uncalibrated_ece": uncal_ece_dict["ece"],
            "calibrated_ece": cal_ece_dict["ece"],
            "ece_reduction_pct": round((uncal_ece_dict["ece"] - cal_ece_dict["ece"]) / max(uncal_ece_dict["ece"], 1e-6) * 100, 2),
            "uncalibrated_brier": uncal_brier,
            "calibrated_brier": cal_brier,
        },
        "selected_thresholds": thresholds,
        "reliability_diagram_path": str(diagram_path),
    }

    cal_params_path = ckpt_path.parent / "calibration_params.json"
    save_json(calibration_output, cal_params_path)
    logger.info("Calibration parameters saved to: %s", cal_params_path)

    # Print summary
    print("\n" + "=" * 60)
    print("CALIBRATION RESULTS (Validation Set)")
    print("=" * 60)
    print(f"  Optimal Temperature (T):    {optimal_temp:.4f}")
    print(f"  Uncalibrated ECE:           {uncal_ece_dict['ece']:.4f}")
    print(f"  Calibrated ECE:             {cal_ece_dict['ece']:.4f}")
    print(f"  ECE Reduction:              {calibration_output['validation_metrics']['ece_reduction_pct']}%")
    print(f"  Uncalibrated Brier Score:   {uncal_brier:.4f}")
    print(f"  Calibrated Brier Score:     {cal_brier:.4f}")
    print(f"  High-Confidence Threshold:  {thresholds['high_confidence_threshold']}")
    print(f"  Low-Confidence Threshold:   {thresholds['low_confidence_threshold']}")
    print(f"  Abstain Below:              {thresholds['abstain_below_threshold']}")
    print(f"  Reliability Diagram:        {diagram_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
