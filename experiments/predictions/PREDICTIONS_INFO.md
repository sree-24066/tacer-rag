# Prediction Files — Not Included in Repo

The `*_val_preds.jsonl` files are excluded because they are generated outputs from model evaluation.

## How to Regenerate

```bash
python scripts/evaluate_detector.py
```

This will create prediction JSONL files for each checkpoint in `experiments/predictions/`.
