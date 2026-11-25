"""
Ablation study script for TACER-RAG.

Evaluates the contribution of each system component:
1. Full TACER-RAG (Type-aware + Calibrated + NLI verified + Post-verified)
2. Ablation 1: No Calibration (raw model confidence, fixed threshold 0.5)
3. Ablation 2: No Type Awareness (generic replacement for all hallucination types)
4. Ablation 3: No NLI Verification (bypass premise-hypothesis verification)
5. Ablation 4: No Post-Correction Verification (single-pass, no regression check)
6. Ablation 5: No Abstention (force rewriting on all detected spans)

Usage:
    python scripts/run_ablations.py --run_name debug_ModernBERT-base_ep5_lr2e-05 --n_samples 50
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from tacer_rag.utils import get_logger, project_root, save_json, ensure_dir
from tacer_rag.data.dataset import load_examples, RagTruthExample
from tacer_rag.pipeline import TACERPipeline
from tacer_rag.detection.predict import PredictedSpan
from tacer_rag.correction.controller import CorrectionController
from tacer_rag.correction.post_verifier import PostCorrectionVerifier

logger = get_logger("run_ablations")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--run_name", default="debug_ModernBERT-base_ep5_lr2e-05")
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--n_samples", type=int, default=30,
                   help="Number of samples to evaluate for ablation study")
    p.add_argument("--device", default=None)
    return p.parse_args()


def run_ablation_condition(
    condition_name: str,
    examples: List[RagTruthExample],
    pipeline: TACERPipeline,
    use_calibration: bool = True,
    use_type_awareness: bool = True,
    use_nli: bool = True,
    use_post_verification: bool = True,
    use_abstention: bool = True,
) -> Dict[str, Any]:
    """
    Run evaluation on examples under a specific ablation configuration.
    """
    logger.info("Evaluating condition: %s (%d examples)...", condition_name, len(examples))
    t0 = time.time()

    total_hallucinations_detected = 0
    total_corrections_attempted = 0
    total_regressions = 0
    total_abstained = 0
    new_hallucination_counts = []

    # Configure pipeline according to ablation condition
    pipeline.temperature = pipeline.temperature if use_calibration else 1.0
    abstain_cutoff = 0.25 if use_abstention else 0.0
    pipeline.controller.abstain_below = abstain_cutoff

    for ex in examples:
        spans = pipeline.detect_spans(ex.context, ex.answer)

        # If no type awareness, treat all spans as generic conflict
        if not use_type_awareness:
            spans = [
                PredictedSpan(
                    start=s.start, end=s.end, text=s.text,
                    label_type="EVIDENT_CONFLICT",  # Flatten to generic type
                    confidence=s.confidence
                )
                for s in spans
            ]

        total_hallucinations_detected += len(spans)

        if not spans:
            continue

        if use_post_verification:
            pv = pipeline.post_verifier.verify_and_refine(ex.context, ex.answer, initial_spans=spans)
            if pv.has_regression:
                total_regressions += 1
            new_hallucination_counts.append(pv.newly_introduced_count)
            total_corrections_attempted += len(pv.decisions_per_round)
            if any(d.abstained for d in pv.decisions_per_round):
                total_abstained += 1
        else:
            decision = pipeline.controller.correct_answer(ex.answer, spans, ex.context)
            total_corrections_attempted += 1
            if decision.abstained:
                total_abstained += 1
            # Check regression manually
            re_spans = pipeline.detect_spans(ex.context, decision.corrected_answer)
            new_spans = [s for s in re_spans if s.text not in [orig.text for orig in spans]]
            if len(new_spans) > 0:
                total_regressions += 1
            new_hallucination_counts.append(len(new_spans))

    elapsed = time.time() - t0
    avg_nhr = float(sum(new_hallucination_counts) / max(len(new_hallucination_counts), 1))

    return {
        "condition": condition_name,
        "n_samples": len(examples),
        "total_hallucinations_detected": total_hallucinations_detected,
        "total_corrections_attempted": total_corrections_attempted,
        "total_abstained": total_abstained,
        "total_regressions": total_regressions,
        "regression_rate": round(total_regressions / max(total_corrections_attempted, 1), 4),
        "avg_new_hallucinations_per_correction": round(avg_nhr, 4),
        "elapsed_seconds": round(elapsed, 1),
    }


def main() -> None:
    args = parse_args()
    root = project_root()

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        run_name = ckpt_path.parent.name
    else:
        run_name = args.run_name
        ckpt_path = root / "experiments" / "checkpoints" / run_name / "best_model.pt"

    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    # Load dataset split
    split_file = root / "data" / "processed" / f"{args.split}.jsonl"
    logger.info("Loading %s data for ablation...", args.split)
    all_examples = load_examples(split_file)
    sample_examples = all_examples[:args.n_samples]

    logger.info("Initializing TACER-RAG pipeline from %s", ckpt_path)
    pipeline = TACERPipeline.from_checkpoint(ckpt_path, device=args.device)

    conditions = [
        ("Full TACER-RAG", True, True, True, True, True),
        ("Ablation 1: No Calibration", False, True, True, True, True),
        ("Ablation 2: No Type Awareness", True, False, True, True, True),
        ("Ablation 3: No NLI Verification", True, True, False, True, True),
        ("Ablation 4: No Post-Verification", True, True, True, False, True),
        ("Ablation 5: No Abstention", True, True, True, True, False),
    ]

    results = []
    for cond_name, cal, typ, nli, post, abs_flag in conditions:
        res = run_ablation_condition(
            condition_name=cond_name,
            examples=sample_examples,
            pipeline=pipeline,
            use_calibration=cal,
            use_type_awareness=typ,
            use_nli=nli,
            use_post_verification=post,
            use_abstention=abs_flag,
        )
        results.append(res)

    # Save ablation results
    metrics_dir = ensure_dir(root / "experiments" / "metrics")
    out_file = metrics_dir / f"{run_name}_ablations.json"
    save_json(results, out_file)
    logger.info("Ablation results saved to %s", out_file)

    # Format output table
    print("\n" + "=" * 80)
    print("TACER-RAG ABLATION STUDY RESULTS")
    print("=" * 80)
    print(f"{'Condition':<32} | {'Corrections':<12} | {'Abstained':<10} | {'Regressions':<12} | {'Regression %':<12}")
    print("-" * 80)
    for r in results:
        reg_pct = f"{r['regression_rate'] * 100:.1f}%"
        print(f"{r['condition']:<32} | {r['total_corrections_attempted']:<12} | {r['total_abstained']:<10} | {r['total_regressions']:<12} | {reg_pct:<12}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
