# TACER-RAG: Experiment Log

**Project:** Type-Aware and Confidence-Calibrated Hallucination Detection  
and Evidence-Constrained Correction for Retrieval-Augmented Generation

> This log records all experiments, their configurations, and their outcomes.
> Failed experiments are recorded here too.
> No fabricated results.

---

## Experiment Summary Table

| Exp ID | Phase | Description | Result | Key Metric / Output |
|---|---|---|---|---|
| EXP-001 | 0 | Hardware and Environment Audit | ✅ PASS | RTX 3050 (4GB), PyTorch 2.6.0+cu124, Python 3.11.9 |
| EXP-002 | 1 | RAGTruth dataset inspection & schema audit | ✅ PASS | 17,790 responses, 4 labels verified, 0 split leakage |
| EXP-003 | 2 | Preprocessing, offset alignment & validation split | ✅ PASS | 14,050 train / 892 val / 2,675 test; 0 source_id leakage |
| EXP-004 | 3 | Detector smoke test (`bert-base-uncased`, 20 ex) | ✅ PASS | Loss 2.588 -> 1.272, 69.9s on CUDA |
| EXP-005 | 3-4 | ModernBERT-base pipeline debug run | ✅ PASS | 149M params, Val loss 1.5816, `best_model.pt` saved |
| EXP-006 | 3-4 | Validation set evaluation (`evaluate_detector.py`) | ✅ PASS | 892 examples, Token & Span metrics (Exact + Overlap F1) |
| EXP-007 | 5 | Confidence calibration (Temperature Scaling) | ✅ PASS | T=1.2772, ECE 0.2111 -> 0.1360 (-35.58%), Brier 0.887 -> 0.848 |
| EXP-008 | 6 | Context evidence selector & NLI verification | ✅ PASS | TF-IDF ranking, sentence segmentation tests 3/3 passed |
| EXP-009 | 7 | Type-aware correction controller | ✅ PASS | 4 policies, reverse-offset multi-span editing, tests 4/4 passed |
| EXP-010 | 8 | Post-correction verification & regression guard | ✅ PASS | NHR metric, 2-round bound, reversion on regression verified |
| EXP-011 | 10 | Streamlit interactive web application | ✅ PASS | Color-coded span highlights, calibration & detail tabs |
| EXP-012 | 11 | End-to-end pipeline CLI execution | ✅ PASS | Full flow: input -> detect -> calibrate -> correct -> verify |

---

## Detailed Experiment Entries

### EXP-001: Hardware and Environment Setup
- **Date:** 2026-11-14
- **Phase:** 0
- **Configuration:** AMD Ryzen 7 7435HS, NVIDIA RTX 3050 Laptop GPU (4 GB VRAM), Windows 10 Home 26H2, Python 3.11.9, PyTorch 2.6.0+cu124, Transformers 5.23.0
- **Result:** PASS
- **Notes:** Upgraded to PyTorch 2.6.0+cu124 to satisfy Transformers 5.23 requirements. All 10 phase 0 tests passed.

### EXP-002: RAGTruth Dataset Schema & Offsets Inspection
- **Date:** 2026-11-14
- **Phase:** 1
- **Findings:**
  - 17,790 responses across 2,965 sources (exactly 6 responses per source_id).
  - Official split: 15,090 train / 2,700 test. Zero source_id leakage.
  - Actual label strings: `"Evident Conflict"`, `"Subtle Conflict"`, `"Evident Baseless Info"`, `"Subtle Baseless Info"` ("Info", not "Information").
  - Subtle Conflict has only 201 spans (severe class imbalance).
  - All 14,289 character offsets validated with zero mismatches.
- **Result:** PASS

### EXP-003: Preprocessing & Leakage-Free Validation Split
- **Date:** 2026-11-14
- **Phase:** 2
- **Outputs:**
  - `data/processed/train.jsonl` (14,050 responses, 2,342 source_ids)
  - `data/processed/val.jsonl` (892 responses, 150 source_ids: 50 QA, 50 Summary, 50 Data2txt)
  - `data/processed/test.jsonl` (2,675 responses, 450 source_ids)
  - Leakage check: 0 source_ids shared across splits.
- **Result:** PASS

### EXP-004: Detector Smoke Test
- **Date:** 2026-11-14
- **Phase:** 3
- **Configuration:** `bert-base-uncased`, 20 examples, max_length=256, lr=2e-5, CUDA fp32
- **Loss:** Step 1: 2.5883 -> Step 2: 1.9168 -> Step 3: 1.2721
- **Result:** PASS (Pipeline learns without memory crashes)

### EXP-005: ModernBERT-base Full Pipeline Verification Run
- **Date:** 2026-11-14
- **Phase:** 3 & 4
- **Configuration:** `answerdotai/ModernBERT-base` (149M params), batch_size=8, grad_accum=4, fp16 AMP, lr=2e-5
- **Outputs:** `experiments/checkpoints/debug_ModernBERT-base_ep5_lr2e-05/best_model.pt`
- **Result:** PASS (Epoch 1 completed in 21.2s, Val loss 1.5816, best model checkpoint saved)

### EXP-006: Validation Set Evaluation (`evaluate_detector.py`)
- **Date:** 2026-11-14
- **Phase:** 3 & 4
- **Data:** 892 validation examples
- **Metrics Recorded:**
  - Token Macro-F1, Binary Hallucination F1
  - Per-category F1 (Evident Conflict, Subtle Conflict, Evident Baseless, Subtle Baseless)
  - Exact Span F1 & Overlap Span F1
- **Artifacts:** `experiments/predictions/debug_ModernBERT-base_ep5_lr2e-05_val_preds.jsonl`, `experiments/metrics/debug_ModernBERT-base_ep5_lr2e-05_val_metrics.json`
- **Result:** PASS

### EXP-007: Confidence Calibration with Temperature Scaling
- **Date:** 2026-11-14
- **Phase:** 5
- **Configuration:** Temperature scaling on validation logits (141,029 tokens) using L-BFGS
- **Outcomes:**
  - Learned Temperature: $T = 1.2772$
  - Uncalibrated ECE: $0.2111$ -> Calibrated ECE: $0.1360$ (**35.58% error reduction**)
  - Uncalibrated Brier: $0.8870$ -> Calibrated Brier: $0.8481$
  - Reliability Diagram: `experiments/figures/debug_ModernBERT-base_ep5_lr2e-05_reliability_diagram.png`
  - Validation Operating Thresholds: High=0.75, Low=0.35, Abstain=0.24
- **Artifacts:** `experiments/checkpoints/debug_ModernBERT-base_ep5_lr2e-05/calibration_params.json`
- **Result:** PASS

### EXP-008: Evidence Selection & NLI Factuality Verification
- **Date:** 2026-11-14
- **Phase:** 6
- **Modules:** `src/tacer_rag/verification/evidence_selector.py`, `nli_verifier.py`
- **Tests:** `tests/test_verification.py` (3 passed)
- **Result:** PASS

### EXP-009: Type-Aware Correction Controller
- **Date:** 2026-11-14
- **Phase:** 7
- **Modules:** `src/tacer_rag/correction/policies.py`, `llm_client.py`, `controller.py`
- **Tests:** `tests/test_correction.py` (4 passed)
- **Result:** PASS

### EXP-010: Post-Correction Verification & Regression Guard
- **Date:** 2026-11-14
- **Phase:** 8
- **Behavior:** Calculates New Hallucination Rate (NHR), bounds loops to $\le 2$ rounds, automatically reverts to original answer if correction introduces new errors.
- **Result:** PASS

### EXP-011: Streamlit Interactive Demonstration
- **Date:** 2026-11-14
- **Phase:** 10
- **File:** `app/streamlit_app.py`
- **Features:** Interactive input, preset examples, color-coded span highlights, calibration analysis tab, system architecture overview.
- **Result:** PASS

### EXP-012: End-to-End Pipeline CLI Execution
- **Date:** 2026-11-14
- **Phase:** 11
- **File:** `scripts/run_pipeline.py`
- **Execution:** Successfully processed custom queries through the full chain: ModernBERT span detection -> temperature calibration -> evidence selection -> type-aware correction -> post-verification audit.
- **Result:** PASS
