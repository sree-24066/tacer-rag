# ============================================================
# Data Directory
# ============================================================

This directory holds **processed** versions of the RAGTruth dataset.

## Structure

```
data/
├── raw/           ← Original downloaded files (do not modify)
├── processed/     ← Preprocessed splits ready for model training
└── README.md      ← This file
```

## Data Files (populated by preprocessing scripts)

| File | Description |
|---|---|
| `processed/train.jsonl` | Training split (by source_id) |
| `processed/val.jsonl` | Validation split (by source_id, no source_id overlap with train/test) |
| `processed/test.jsonl` | Test split (official RAGTruth test split) |
| `processed/label_map.json` | Label-to-ID mapping |
| `processed/stats.json` | Preprocessing statistics |
| `processed/split_summary.json` | Split sizes, source_id counts, category distributions |

## Critical Data Rules

1. **Never train on the test set.**
2. **Split by source_id**, not by individual response IDs.
3. **Never modify raw/ files** — always process to processed/.
4. **Record exclusions** — do not silently discard records.
5. **Preserve character offsets** from the original annotations.

## Generating Processed Data

```powershell
cd "d:\Text project\tacer-rag"
.\.venv\Scripts\Activate.ps1
python scripts/preprocess_data.py --config configs/data.yaml
```
