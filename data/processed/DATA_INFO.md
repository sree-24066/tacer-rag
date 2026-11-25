# Processed Data — Not Included in Repo

The processed `.jsonl` files are excluded from this repository because they are large (train: ~85 MB, val: ~5 MB, test: ~16 MB).

## Source Dataset

**RAGTruth** — A Hallucination Corpus for Developing Trustworthy RAG Systems

- 📄 Paper: https://arxiv.org/abs/2401.00396
- 🔗 Dataset: https://github.com/ParticleMedia/RAGTruth
- 📦 HuggingFace: https://huggingface.co/datasets/wandb/RAGTruth

## How to Regenerate

```bash
# 1. Download raw data
python scripts/download_data.py

# 2. Preprocess into BIO-tagged JSONL splits
python scripts/preprocess_data.py
```

This will create:
- `data/processed/train.jsonl`  (~85 MB, ~1500 labelled samples)
- `data/processed/val.jsonl`    (~5 MB)
- `data/processed/test.jsonl`   (~16 MB)
