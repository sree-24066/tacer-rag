# Model Checkpoints — Not Included in Repo

The `.pt` model weight files are excluded from this repository because they are very large (~568 MB each).

## Available Checkpoints

| Checkpoint | Description |
|---|---|
| `ModernBERT_trained_1500/` | Main model — fine-tuned on 1500 RAGTruth samples (recommended) |
| `debug_ModernBERT-base_ep5_lr2e-05/` | Debug run — 5 epochs, lr=2e-05 |

Each checkpoint folder contains:
- `best_model.pt` — Best validation checkpoint (**excluded from repo**)
- `latest_model.pt` — Last epoch checkpoint (**excluded from repo**)
- `calibration_params.json` — Temperature scaling params ✅ (included)
- `training_args.json` — Training hyperparameters ✅ (included)

## How to Reproduce

```bash
# Train the detector from scratch
python scripts/train_detector.py

# Calibrate confidence scores
python scripts/calibrate_detector.py
```

## Base Model

**answerdotai/ModernBERT-base** (HuggingFace)
- 🔗 https://huggingface.co/answerdotai/ModernBERT-base
- Downloaded automatically on first run
