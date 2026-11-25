# TACER-RAG: Known Limitations

> Record limitations honestly. Do not hide them.

---

## Hardware Limitations

- **4 GB VRAM** limits batch size, sequence length, and model size.
  Full fine-tuning of large encoders (> 350M parameters) may cause OOM.
- **Local CPU-only inference** will be slow for LLM-based correction.
- **No multi-GPU setup** — cannot use data parallelism.

## Data Limitations

- RAGTruth is a closed benchmark. Performance may not generalise to other RAG systems.
- Hallucination categories in RAGTruth were annotated by humans; inter-annotator agreement is not 100%.
- The dataset covers specific LLM outputs; distribution shift may occur with other generators.

## Model Limitations

- ModernBERT-base has a fixed context window; very long contexts require chunking, which may miss cross-chunk hallucinations.
- NLI-based verification is an approximation of factual grounding.
- LLM-based correction quality depends on the local model available.

## Methodological Limitations

- Post-correction verification uses the same detector, which may have systematic blind spots.
- Confidence calibration is performed on the validation set; it may not transfer perfectly to the test set.
- Type-specific correction policies are designed based on category definitions; they have not been validated against human judgment in this study.

## Scope Limitations

- This project does not address hallucinations in multi-hop reasoning chains.
- Real-time retrieval is not implemented; contexts are pre-retrieved.
- The Streamlit demo is for demonstration only and should not be used in production.
