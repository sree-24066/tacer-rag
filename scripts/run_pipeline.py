"""
CLI runner for the end-to-end TACER-RAG pipeline.

Usage:
    # Run on sample example
    python scripts/run_pipeline.py --run_name debug_ModernBERT-base_ep5_lr2e-05

    # Run on custom text
    python scripts/run_pipeline.py --checkpoint experiments/checkpoints/debug_ModernBERT-base_ep5_lr2e-05/best_model.pt \
        --context "The Apollo 11 mission landed on the Moon in July 1969. Neil Armstrong was the first person to walk on the lunar surface." \
        --answer "Apollo 11 landed on Mars in 1969 and Neil Armstrong walked on the Martian surface."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from tacer_rag.utils import get_logger, project_root
from tacer_rag.pipeline import TACERPipeline

logger = get_logger("run_pipeline")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--run_name", default="debug_ModernBERT-base_ep5_lr2e-05")
    p.add_argument("--context", default=None)
    p.add_argument("--answer", default=None)
    p.add_argument("--question", default="")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()

    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = root / "experiments" / "checkpoints" / args.run_name / "best_model.pt"

    if not ckpt_path.exists():
        logger.error("Checkpoint not found: %s", ckpt_path)
        sys.exit(1)

    # Sample input if not provided
    context = args.context or (
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
        "It is named after the engineer Gustave Eiffel, whose company designed and built the tower from 1887 to 1889. "
        "It has a height of 330 metres."
    )
    answer = args.answer or (
        "The Eiffel Tower is in Berlin, Germany. It was constructed in 1889 and is 330 metres tall."
    )
    question = args.question or "Where is the Eiffel Tower and when was it built?"

    logger.info("Initializing TACER-RAG pipeline from %s", ckpt_path)
    pipeline = TACERPipeline.from_checkpoint(ckpt_path, device=args.device)

    logger.info("Processing query...")
    result = pipeline.process(context=context, answer=answer, question=question)

    print("\n" + "=" * 65)
    print("TACER-RAG EXECUTION REPORT")
    print("=" * 65)
    print(f"Question:        {question}")
    print(f"Context:         {context}")
    print("-" * 65)
    print(f"Original Answer: {answer}")
    print(f"Hallucinated?    {result.has_hallucinations}")
    print(f"Detected Spans:  {len(result.detected_spans)}")

    for i, s in enumerate(result.detected_spans, 1):
        print(f"  [{i}] Span:       '{s.text}' (chars {s.start}-{s.end})")
        print(f"      Type:       {s.label_type}")
        print(f"      Confidence: {s.confidence:.4f} (Calibrated T={result.temperature_applied:.3f})")

    print("-" * 65)
    print("Actions Executed:")
    if result.actions_taken:
        for a in result.actions_taken:
            print(f"  * Action:      {a.action}")
            print(f"    Target:      '{a.span_text}'")
            print(f"    Explanation: {a.explanation}")
    else:
        print("  * None (No hallucinated spans detected or all preserved)")

    if result.post_verification:
        pv = result.post_verification
        print("-" * 65)
        print("Post-Correction Verification:")
        print(f"  Rounds run:         {pv.rounds_executed}")
        print(f"  Removed spans:      {pv.removed_count}")
        print(f"  Remaining spans:    {pv.remaining_count}")
        print(f"  New hallucinations: {pv.newly_introduced_count} (NHR: {pv.new_hallucination_rate:.2f})")
        print(f"  Regression found:   {pv.has_regression} (Reverted: {pv.reverted_to_original})")

    print("=" * 65)
    print(f"Final Answer:    {result.final_answer}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
