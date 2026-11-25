"""
Smoke test for Phase 3: verify the full training pipeline on a tiny
subset before committing to a full training run.

Runs on CPU or GPU. Does NOT require ModernBERT download — uses
bert-base-uncased as a fast stand-in. Full ModernBERT run is in
scripts/train_detector.py.

Usage:
    python scripts/smoke_test_detector.py

Expected outcome: loss decreases over 3 steps, no crashes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(_SRC))

from tacer_rag.utils import get_logger, project_root, set_seed

logger = get_logger("smoke_test")


def main() -> None:
    set_seed(42)
    root = project_root()

    # ── 1. Check processed data exists ───────────────────────────────────────
    train_file = root / "data" / "processed" / "train.jsonl"
    if not train_file.exists():
        logger.error("train.jsonl not found. Run scripts/preprocess_data.py first.")
        sys.exit(1)

    # ── 2. Import ML stack ────────────────────────────────────────────────────
    logger.info("Importing torch and transformers...")
    import torch
    from transformers import AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── 3. Load a tiny subset (20 examples for speed) ────────────────────────
    logger.info("Loading 20 training examples for smoke test...")
    from tacer_rag.data.dataset import load_examples, RagTruthTokenDataset

    all_examples = load_examples(train_file)
    smoke_examples = all_examples[:20]
    logger.info("Loaded %d examples (smoke subset)", len(smoke_examples))

    # ── 4. Tokenize with bert-base-uncased (fast, no large download) ──────────
    # NOTE: For the real run, replace with "answerdotai/ModernBERT-base"
    SMOKE_ENCODER = "bert-base-uncased"
    logger.info("Loading tokenizer: %s", SMOKE_ENCODER)
    tokenizer = AutoTokenizer.from_pretrained(SMOKE_ENCODER)

    truncation_stats = {"dropped_spans": 0, "total_spans": 0}
    dataset = RagTruthTokenDataset(
        smoke_examples,
        tokenizer,
        max_length=256,  # Short for speed
        truncation_stats=truncation_stats,
    )
    logger.info("Dataset size: %d examples", len(dataset))
    logger.info("Truncation stats: %s", truncation_stats)

    # ── 5. Inspect one example ────────────────────────────────────────────────
    item = dataset[0]
    logger.info("Example 0 keys: %s", list(item.keys()))
    logger.info("input_ids shape: %s", item["input_ids"].shape)
    logger.info("labels shape:    %s", item["labels"].shape)

    from tacer_rag.data.dataset import LABEL_MAP, ID_TO_LABEL, IGNORE_INDEX
    unique_labels = set(item["labels"].tolist())
    unique_labels.discard(IGNORE_INDEX)
    readable = {ID_TO_LABEL.get(l, f"ID={l}") for l in unique_labels}
    logger.info("Unique labels in example 0 (excluding IGNORE): %s", readable)

    # ── 6. Build model ────────────────────────────────────────────────────────
    logger.info("Building model with encoder: %s", SMOKE_ENCODER)
    from tacer_rag.detection.model import HallucinationDetector, DetectorConfig

    config = DetectorConfig(encoder_name=SMOKE_ENCODER, num_labels=9, max_length=256)
    model = HallucinationDetector(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %s M", f"{n_params/1e6:.1f}")

    # ── 7. Mini training loop (3 steps) ──────────────────────────────────────
    from torch.utils.data import DataLoader
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    model.train()

    logger.info("Starting 3-step smoke training...")
    losses = []
    for step, batch in enumerate(loader):
        if step >= 3:
            break
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        token_type_ids = batch.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        logger.info("Step %d | loss = %.4f", step, loss.item())

    # ── 8. Mini evaluation (greedy decode, no calibration yet) ───────────────
    model.eval()
    all_preds, all_labels_flat = [], []
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
            )
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.view(-1).cpu().tolist())
            all_labels_flat.extend(labels.view(-1).cpu().tolist())

    from tacer_rag.detection.metrics import compute_token_metrics
    metrics = compute_token_metrics(all_preds, all_labels_flat)
    logger.info("Smoke eval token_macro_f1:            %.4f", metrics["token_macro_f1"])
    logger.info("Smoke eval binary_hallucination_f1:   %.4f", metrics["binary_hallucination_f1"])

    # ── 9. Summary ────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("SMOKE TEST PASSED ✅")
    print("="*60)
    print(f"  Encoder:          {SMOKE_ENCODER}")
    print(f"  Device:           {device}")
    print(f"  Examples:         {len(dataset)}")
    print(f"  Steps trained:    3")
    print(f"  Losses:           {[round(l, 4) for l in losses]}")
    print(f"  Token macro-F1:   {metrics['token_macro_f1']} (random baseline expected)")
    print(f"  Binary hall. F1:  {metrics['binary_hallucination_f1']}")
    print(f"\n  NOTE: F1 scores are near-random — model is untrained (3 steps only).")
    print(f"  This test validates the pipeline works, not model quality.")
    print("="*60)
    print(f"\nNext step: Run full training with ModernBERT-base:")
    print(f"  python scripts/train_detector.py --encoder answerdotai/ModernBERT-base")


if __name__ == "__main__":
    t0 = time.time()
    main()
    logger.info("Smoke test completed in %.1f seconds", time.time() - t0)
