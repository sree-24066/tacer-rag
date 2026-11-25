# 🛡️ TACER-RAG

**Type-Aware & Confidence-Calibrated Hallucination Detection and Evidence-Constrained Correction for RAG**

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Streamlit%20Cloud-FF4B4B?style=for-the-badge)](https://tacer-rag.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-sree--24066%2Ftacer--rag-181717?style=for-the-badge&logo=github)](https://github.com/sree-24066/tacer-rag)
[![Model](https://img.shields.io/badge/🤗%20Model-Sreekanth2005%2Ftacer--rag--model-FFD21E?style=for-the-badge)](https://huggingface.co/Sreekanth2005/tacer-rag-model)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?style=for-the-badge&logo=pytorch)](https://pytorch.org)

---

## 🌐 Live Demo

**Try it now (no setup needed):**  
👉 **[https://tacer-rag.streamlit.app](https://tacer-rag.streamlit.app)**

The live app lets you:
- Paste any RAG context + generated answer
- Detect hallucinated spans highlighted in colour
- See the hallucination **type** (Evident Conflict / Subtle Conflict / Evident Baseless / Subtle Baseless)
- View the corrected answer after evidence-constrained repair
- Try 5 built-in preset examples covering all hallucination categories

> **Note:** First load may take 30–60 seconds as the model downloads from HuggingFace Hub (~570 MB). Subsequent runs are instant.

---

## 🔬 What is TACER-RAG?

TACER-RAG addresses the fundamental challenge of **factual hallucination in Retrieval-Augmented Generation (RAG)**. Unlike binary hallucination detectors or unconstrained LLM regeneration (which often introduces *new* hallucinations), TACER-RAG implements a **modular six-stage pipeline** grounded in NLI verification.

### The 4 Hallucination Types

| Type | Description | Example |
|------|-------------|---------|
| 🔴 **Evident Conflict** | Direct factual contradiction (entity, place, number swapped) | "The Eiffel Tower is in *Berlin*" |
| 🟠 **Subtle Conflict** | Semantic distortion — overstatement or scope change | "exercise *completely eliminates* risk" vs "reduces risk" |
| 🟣 **Evident Baseless** | Specific concrete claim absent from context | "They found *alien artefacts* on the Moon" |
| 🔵 **Subtle Baseless** | Vague inference or ungrounded speculation | "probably work by *psychologically motivating* patients" |

---

## 🏗️ Architecture

```
Input: (Question, Context, RAG Answer)
            │
            ▼
┌─────────────────────────────────┐
│  1. ModernBERT Token Classifier │  ← Fine-tuned on RAGTruth (1500 samples)
│     BIO tagging, 9-class        │    with Temperature Scaling calibration
└────────────────┬────────────────┘
                 │  Candidate spans
                 ▼
┌─────────────────────────────────┐
│  2. Word-Overlap Filter         │  ← Drops verbatim context spans (fast)
│  3. NLI Entailment Gate         │  ← DeBERTa-v3: drops p_entail ≥ 0.40
└────────────────┬────────────────┘
                 │  Verified hallucinated spans
                 ▼
┌─────────────────────────────────┐
│  4. 4-Type Semantic Classifier  │  ← NLI probs + entity/hedge heuristics
└────────────────┬────────────────┘
                 │  Typed spans
                 ▼
┌─────────────────────────────────┐
│  5. Type-Aware Correction       │  ← Policy: Replace/Rewrite/Excise/Qualify
│     Controller                  │
└────────────────┬────────────────┘
                 │
                 ▼
┌─────────────────────────────────┐
│  6. Post-Correction Verifier    │  ← Detects regression (new hallucinations)
└────────────────┬────────────────┘
                 │
                 ▼
         Corrected Answer
```

---

## 📊 Results

| Metric | Value |
|--------|-------|
| **Detection Backbone** | ModernBERT-base (149M params) |
| **Training Data** | RAGTruth — 1,500 annotated QA responses |
| **Uncalibrated ECE** | 0.2111 |
| **Calibrated ECE** | 0.1360 (**−35.58%**) |
| **Temperature T*** | 1.2772 |
| **Train split** | 14,050 responses (2,342 source IDs) |
| **Val split** | 892 responses (150 source IDs) |
| **Test split** | 2,675 responses (450 source IDs) |
| **Unit tests** | 19/19 passing ✅ |

---

## 📁 Repository Structure

```
tacer-rag/
├── app/
│   └── streamlit_app.py          # Interactive web UI (Streamlit)
├── src/tacer_rag/
│   ├── pipeline.py               # Unified TACERPipeline (end-to-end)
│   ├── data/                     # RAGTruth loaders, BIO dataset, schema
│   ├── detection/                # ModernBERT detector, predict.py, metrics
│   ├── calibration/              # Temperature scaling, ECE, reliability diagrams
│   ├── verification/             # Evidence selector + NLI verifier (DeBERTa-v3)
│   └── correction/               # Type-aware controller + post-verifier
├── scripts/
│   ├── train_detector.py         # ModernBERT fine-tuning (fp16 AMP)
│   ├── evaluate_detector.py      # Token & span evaluation
│   ├── calibrate_detector.py     # Temperature scaling & ECE plotting
│   ├── run_pipeline.py           # CLI runner for custom queries
│   └── run_ablations.py          # 6-condition ablation benchmarks
├── configs/                      # YAML: data, model, correction params
├── tests/                        # 19 unit & integration tests
├── experiments/                  # Calibration figures, metrics, predictions
└── research/                     # Methodology, literature review
```

---

## 🚀 Setup & Run Locally

### Prerequisites
- Python 3.11+
- Git
- ~4 GB disk (model weights)
- GPU optional (CPU works, slower)

### 1. Clone the repo

```bash
git clone https://github.com/sree-24066/tacer-rag.git
cd tacer-rag
```

### 2. Create virtual environment & install

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -e ".[dev]"
```

### 3. Run the Streamlit app

```bash
python -m streamlit run app/streamlit_app.py
```

Open **http://localhost:8501** in your browser.

> On first run, the app automatically downloads:
> - `answerdotai/ModernBERT-base` tokenizer (~500 MB, from HuggingFace Hub)
> - `cross-encoder/nli-deberta-v3-base` NLI model (~180 MB)
> - Fine-tuned checkpoint from [`Sreekanth2005/tacer-rag-model`](https://huggingface.co/Sreekanth2005/tacer-rag-model) (~570 MB)

### 4. Run the CLI pipeline

```bash
python scripts/run_pipeline.py \
  --context "The Eiffel Tower is in Paris, France. It was built in 1889." \
  --answer "The Eiffel Tower is in Berlin, Germany. It was built in 1889."
```

### 5. Run tests

```bash
python -m pytest tests/ -v
```

---

## 🧩 Preset Examples (in the app)

| Example | Type | What it tests |
|---------|------|--------------|
| ✅ Example 1 | **No Hallucinations** | Clean verbatim answer — zero false positives |
| 🔴 Example 2 | **Evident Conflict** | "Berlin, Germany" contradicts "Paris, France" |
| 🟠 Example 3 | **Subtle Conflict** | "completely eliminates" vs "reduces risk" |
| 🟣 Example 4 | **Evident Baseless** | Alien artefacts fabricated fact |
| 🔵 Example 5 | **Subtle Baseless** | "psychologically motivating" unjustified inference |

---

## 🤗 Model on HuggingFace

Fine-tuned checkpoint available at:  
👉 **[huggingface.co/Sreekanth2005/tacer-rag-model](https://huggingface.co/Sreekanth2005/tacer-rag-model)**

The app downloads this automatically on first run. Manual download:

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Sreekanth2005/tacer-rag-model", local_dir="experiments/checkpoints/ModernBERT_trained_1500")
```

---

## 📦 Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Deep learning backbone |
| `transformers` | ModernBERT + DeBERTa-v3 |
| `streamlit` | Web UI |
| `huggingface_hub` | Model auto-download |
| `scikit-learn` | TF-IDF evidence selector |
| `numpy`, `scipy` | Calibration maths |

---

## 🏷️ Citation

```bibtex
@misc{tacer-rag-2026,
  title   = {TACER-RAG: Type-Aware and Confidence-Calibrated Hallucination Detection and Evidence-Constrained Correction for RAG},
  author  = {Sreekanth S},
  year    = {2026},
  url     = {https://github.com/sree-24066/tacer-rag}
}
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
