# TACER-RAG: Paper Outline

> **Status:** Skeleton — to be filled with actual results.
> Do not add fabricated results or unverified novelty claims.

---

## Proposed Title

Type-Aware and Confidence-Calibrated Hallucination Detection and
Evidence-Constrained Correction for Retrieval-Augmented Generation

*(Title may be revised after literature review and experiments.)*

---

## 1. Abstract (Draft)

*(To be written after results are available.)*

---

## 2. Introduction

- Motivation: RAG systems reduce but do not eliminate hallucinations.
- Problem: Existing detection is often binary and not type-aware; correction is not evidence-constrained.
- Research question: Can confidence-calibrated, type-aware correction improve reliability?
- Contributions (to be verified against literature review):
  - [ ] Four-type ModernBERT detector on RAGTruth
  - [ ] Confidence calibration for detection decisions
  - [ ] Type-specific evidence-constrained correction policies
  - [ ] Post-correction regression detection metric
- Paper organisation.

---

## 3. Related Work

*(Populated from literature_review.md after reviews are complete.)*

Topics to cover:
- RAG systems and hallucination
- Hallucination detection benchmarks (RAGTruth, HaluEval, FELM)
- Token/span-level detection (LettuceDetect, RefChecker)
- Detection-guided correction (RAG-HAT, RARR)
- Confidence calibration in NLP
- NLI for factuality

---

## 4. Problem Formulation

*(From methodology.md — formalise after implementation.)*

---

## 5. Proposed Method

5.1 Input Processing and Segmentation
5.2 ModernBERT Type-Aware Detector
5.3 Confidence Calibration
5.4 Evidence Verification
5.5 Type-Aware Correction Controller
5.6 Post-Correction Verification

---

## 6. Experimental Setup

6.1 Dataset: RAGTruth
6.2 Baselines
6.3 Ablation configurations
6.4 Evaluation metrics
6.5 Implementation details

---

## 7. Results

*(Tables and figures generated from actual experiment runs.)*

---

## 8. Ablation Study

*(Ablation results generated from actual experiment runs.)*

---

## 9. Error Analysis

*(Case studies from actual model predictions.)*

---

## 10. Limitations

*(From research/limitations.md)*

---

## 11. Conclusion

*(To be written after results.)*

---

## 12. References

*(To be populated from literature_review.md — no fabricated references.)*
