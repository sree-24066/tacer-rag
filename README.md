# TACER-RAG: Type-Aware and Confidence-Calibrated Hallucination Detection and Evidence-Constrained Correction for Retrieval-Augmented Generation

> **Status:** Research Implementation Complete & Verified  
> **Environment:** Python 3.11.9 (64-bit) | PyTorch 2.6.0+cu124 | NVIDIA RTX 3050 (4 GB VRAM)  
> **Tests:** 19/19 Unit & Integration Tests Passing (100%)

---

## 🔬 Research Overview

**TACER-RAG** addresses the fundamental challenge of factual hallucination in Retrieval-Augmented Generation (RAG). Rather than treating hallucination as an amorphous binary problem or relying on unconstrained LLM regeneration (which often introduces secondary hallucinations), TACER-RAG implements a modular, six-stage pipeline:

1. **Token-Level Hallucination Localization:** Uses ModernBERT fine-tuned on RAGTruth with a 9-class BIO tagging formulation.
2. **Four-Category Error Typology:** Distinguishes between *Evident Conflict*, *Subtle Conflict*, *Evident Baseless Info*, and *Subtle Baseless Info*.
3. **Post-Hoc Confidence Calibration:** Learns temperature parameter $T^*$ on validation negative log-likelihood, reducing Expected Calibration Error (ECE) by **35.58%**.
4. **Context Evidence Selection & NLI Grounding:** Retrieves the most relevant evidence sentences from the context using TF-IDF ranking and verifies claims with NLI without external web knowledge.
5. **Decoupled Type-Aware Correction Controller:** Applies policy-guided edits (Replace, Rewrite, Excise, Qualify) in reverse character offset order.
6. **Post-Correction Verification & Regression Guard:** Measures New Hallucination Rate (NHR) and automatically reverts changes if secondary hallucinations are introduced.

---

## 📁 Repository Structure

```
tacer-rag/
├── configs/            # YAML configs: data, model, correction parameters
├── data/               # Processed datasets (train: 14k, val: 892, test: 2.6k)
├── external/           # RAGTruth and LettuceDetect cloned submodules
├── src/tacer_rag/      # Core Python research package
│   ├── data/           # RAGTruth loaders, token dataset, schema validation
│   ├── detection/      # ModernBERT detector, BIO metrics, span extraction
│   ├── calibration/    # Temperature scaling, ECE/MCE, reliability diagrams
│   ├── verification/   # Context evidence selector & NLI verifier
│   ├── correction/     # Type-specific policies, controller, post-verifier
│   └── pipeline.py     # Unified end-to-end TACERPipeline
├── scripts/            # CLI scripts for each research phase
│   ├── train_detector.py      # ModernBERT training with fp16 AMP
│   ├── evaluate_detector.py   # Token & span evaluation on val/test
│   ├── calibrate_detector.py  # Temperature scaling & ECE plotting
│   ├── run_pipeline.py        # End-to-end CLI runner on custom queries
│   └── run_ablations.py       # Benchmarking 6 ablation conditions
├── app/
│   └── streamlit_app.py       # Interactive web UI with span visualization
├── tests/              # 19 passing unit & integration tests
├── experiments/        # Checkpoints, calibration params, figures, metrics
└── research/           # Methodology, literature review, paper outline
```

---

## ⚡ Quickstart

### 1. Run the End-to-End Pipeline

Test the complete pipeline on any query, context, and answer:

```powershell
& "d:\Text project\tacer-rag\.venv\Scripts\python.exe" scripts/run_pipeline.py
```

Or provide custom inputs:
```powershell
& "d:\Text project\tacer-rag\.venv\Scripts\python.exe" scripts/run_pipeline.py `
  --context "The Apollo 11 mission landed on the Moon in July 1969. Neil Armstrong was the first person to walk on the lunar surface." `
  --answer "Apollo 11 landed on Mars in 1969 and Neil Armstrong walked on the Martian surface."
```

### 2. Launch the Streamlit Interactive Demonstration

```powershell
& "d:\Text project\tacer-rag\.venv\Scripts\streamlit.exe" run app/streamlit_app.py
```
Open your browser at `http://localhost:8501` to view color-coded hallucination span highlighting, calibration diagrams, and live before/after answer comparisons.

### 3. Run the Test Suite

```powershell
& "d:\Text project\tacer-rag\.venv\Scripts\python.exe" -m pytest "d:\Text project\tacer-rag\tests" -v
```

---

## 📊 Experimental Results Summary

- **Detection Backbone:** ModernBERT-base (149M parameters, RoPE embeddings, unpadded attention)
- **Token Calibration:**
  - Optimal Temperature: $T = 1.2772$
  - Uncalibrated ECE: $0.2111 \rightarrow$ Calibrated ECE: $0.1360$ (**-35.58%**)
  - Multi-class Brier Score: $0.8870 \rightarrow 0.8481$
  - Reliability Diagram: `experiments/figures/debug_ModernBERT-base_ep5_lr2e-05_reliability_diagram.png`
- **Data Splits:**
  - Train: 14,050 responses (2,342 source IDs)
  - Validation: 892 responses (150 source IDs: 50 QA, 50 Summary, 50 Data2txt)
  - Test: 2,675 responses (450 source IDs)
  - Zero source_id leakage across splits.
