

# ============================================================
# External Research Repositories
# ============================================================

This directory contains **read-only clones** of external repositories
used as reference implementations, baselines, or data sources.

> **Do NOT modify any files inside these cloned repositories.**
> Run `git status` inside each subdirectory to verify they remain clean.

## Repositories

| Directory | Source | Purpose | License |
|---|---|---|---|
| `RAGTruth/` | https://github.com/ParticleMedia/RAGTruth | Dataset + annotations | See repo |
| `LettuceDetect/` | https://github.com/KRLabsOrg/LettuceDetect | Baseline model | See repo |

## Usage

These repositories are referenced but not imported directly by the
`tacer_rag` package. Data is copied to `data/raw/` and then processed
to `data/processed/` using our own preprocessing scripts.

Baseline model weights from LettuceDetect are loaded via Hugging Face Hub
using their published model IDs, not by importing their source code.

## Cloning

```powershell
cd "d:\Text project\tacer-rag\external"

# Clone RAGTruth
git clone https://github.com/ParticleMedia/RAGTruth.git

# Clone LettuceDetect
git clone https://github.com/KRLabsOrg/LettuceDetect.git
```
