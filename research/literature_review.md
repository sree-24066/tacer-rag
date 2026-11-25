# TACER-RAG: Literature Review

> **Status:** To be populated during Phase 1 (prior-art review).
>
> **Rules:**
> - Do NOT fabricate paper details, results, or citations.
> - If a paper cannot be accessed, state that explicitly.
> - Record the overlap with our work honestly, even if it reduces novelty.

---

## Review Template

Each entry:

```
### [SHORT_ID] Title
- **Authors:**
- **Year:**
- **Venue:**
- **Link:**
- **Method:**
- **Dataset:**
- **Metrics:**
- **Main Results:**
- **Overlap with TACER-RAG:**
- **What Remains Different:**
- **Challenges Our Novelty Claim:** Yes / No / Partial
- **Notes:**
```

---

## Papers to Review (Priority Order)

### Tier 1 — Direct Overlap (must review before claiming novelty)

1. RAGTruth (Niu et al., 2024) — primary dataset paper
2. LettuceDetect (KRLabs, 2024) — closest existing system
3. RAG-HAT (EMNLP 2024 Industry) — detection-guided correction
4. Self-RAG (Asai et al., 2023) — reflection-based RAG
5. MiniCheck — compact factuality verification

### Tier 2 — Strongly Related

6. FActScoring / SAFE — factual claim verification
7. HaluEval — hallucination evaluation benchmark
8. FELM — factual error localization
9. RefChecker — reference-based claim checking
10. RARR — retrieval-augmented revision

### Tier 3 — Background and Context

11. Survey papers on RAG hallucination (2023–2024)
12. Confidence calibration in NLP models
13. NLI for factuality verification
14. ModernBERT technical report

---

## Completed Reviews

*(None yet — to be added during Phase 1)*

---

## Preliminary Novelty Assessment

Based on the project description and known related work:

| Claimed Contribution | Known Prior Art | Assessment |
|---|---|---|
| Token-level hallucination detection | LettuceDetect, FELM | Not novel alone |
| Four-type hallucination classification | RAGTruth paper itself | Not novel alone |
| ModernBERT for detection | LettuceDetect | Not novel alone |
| Type-aware correction policies | RAG-HAT (partial) | Possibly novel combination |
| Confidence calibration for detection | Standard NLP | Not novel alone |
| Evidence-constrained correction | RARR (partial) | Partially novel |
| Post-correction regression detection | Unknown — to be searched | Possibly novel |
| Combined pipeline | Unknown — to be searched | Needs literature search |

> This table will be updated as the literature review is completed.
> Do not claim novelty based on this preliminary assessment.
