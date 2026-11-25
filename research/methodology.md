# TACER-RAG: Research Methodology

**Project:** Type-Aware and Confidence-Calibrated Hallucination Detection and Evidence-Constrained Correction for Retrieval-Augmented Generation (TACER-RAG)

---

## 1. Problem Formulation

Given:
- A user query / question $q$
- A set of retrieved context passages $C = \{c_1, c_2, \dots, c_m\}$
- An unverified model-generated answer $a = (w_1, w_2, \dots, w_n)$ consisting of $n$ tokens or character span $[0, |a|)$

The goal of TACER-RAG is threefold:
1. **Span-Level Detection & Categorization:** Identify all hallucinated spans $\mathcal{H} = \{(s_i, e_i, t_i)\}_{i=1}^{|\mathcal{H}|}$, where $0 \le s_i < e_i \le |a|$ denote character boundaries in $a$, and $t_i \in \mathcal{T}$ represents one of four distinct hallucination types defined by the RAGTruth taxonomy:
   $$\mathcal{T} = \{\text{Evident Conflict}, \text{Subtle Conflict}, \text{Evident Baseless Info}, \text{Subtle Baseless Info}\}$$
2. **Confidence Calibration:** Estimate a well-calibrated confidence score $\hat{p}_i \in [0, 1]$ reflecting the true empirical probability of hallucination, avoiding overconfident false corrections.
3. **Evidence-Constrained Correction:** Produce a corrected answer $a^*$ by modifying exclusively the detected spans using evidence strictly grounded in $C$, minimizing the **New Hallucination Rate (NHR)**:
   $$a^* = \text{Correct}(a, \mathcal{H}, C, \hat{P})$$

---

## 2. System Architecture

```
                      +-----------------------------+
                      | Input: Context C, Answer a  |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | 1. Tokenization & Alignment |
                      |    [CLS] C [SEP] a [SEP]    |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | 2. ModernBERT Token Model   |
                      |    9-class BIO Tagging Head |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | 3. Temperature Calibration  |
                      |    z_scaled = z / T*        |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | 4. Evidence Retrieval & NLI |
                      |    Ranked Sentences in C    |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | 5. Type-Aware Controller    |
                      |    (Replace/Rewrite/Remove) |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | 6. Post-Correction Check    |
                      |    Verify NHR <= 0          |
                      +-----------------------------+
                                     |
                                     v
                      +-----------------------------+
                      | Final Reliable Answer a*    |
                      +-----------------------------+
```

---

## 3. Detailed Module Specifications

### Module 1: Input Processing & Token-Span Alignment
The input sequence is structured as:
$$x = [\text{CLS}] \circ C \circ [\text{SEP}] \circ a \circ [\text{SEP}]$$
To avoid penalizing the model on context tokens, context token positions receive a target label of $-100$ (`IGNORE_INDEX`). Answer tokens are mapped to character offsets using Hugging Face fast tokenizer offset mappings. Overlapping spans are resolved deterministically by earliest start index.

### Module 2: ModernBERT Hallucination Detector
We employ `answerdotai/ModernBERT-base` (149M parameters) equipped with rotary position embeddings (RoPE) and unpadding optimizations, fine-tuned as a token-level sequence tagger over 9 BIO labels:
$$\mathcal{Y}_{\text{BIO}} = \{\text{O}\} \cup \{\text{B-}t, \text{I-}t \mid t \in \mathcal{T}\}$$

Loss is computed via Cross-Entropy with optional inverse-frequency sample weighting:
$$\mathcal{L}_{\text{CE}} = - \sum_{j \in \text{AnswerTokens}} w_{y_j} \log \frac{\exp(z_{j, y_j})}{\sum_{k=1}^9 \exp(z_{j, k})}$$

### Module 3: Confidence Calibration
Modern neural networks often exhibit poor probability calibration on token classification tasks. We implement **Temperature Scaling** (Guo et al., 2017). On a held-out validation set $\mathcal{D}_{\text{val}}$, we learn a scalar temperature $T > 0$ by optimizing negative log-likelihood:
$$\hat{T} = \arg\min_{T > 0} - \sum_{i \in \mathcal{D}_{\text{val}}} \log \sigma\left(\frac{z_i}{T}\right)_{y_i}$$
where $\sigma(\cdot)$ is the softmax function.

Calibration is evaluated using **Expected Calibration Error (ECE)** and **Brier Score**:
$$\text{ECE} = \sum_{b=1}^B \frac{|B_b|}{N} |\text{acc}(B_b) - \text{conf}(B_b)|$$
$$\text{Brier} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (p_{ik} - y_{ik})^2$$

### Module 4: Grounded Context Evidence Retrieval & NLI
For each suspicious span, candidate context sentences are extracted using rule-based sentence segmentation. Sentences are ranked using sub-linear TF-IDF cosine similarity:
$$\text{sim}(e_k, \text{span}) = \frac{\mathbf{v}_{e_k} \cdot \mathbf{v}_{\text{span}}}{\|\mathbf{v}_{e_k}\| \|\mathbf{v}_{\text{span}}\|}$$

The top candidate is verified using an NLI cross-encoder ($e_k$ as premise, span as hypothesis), producing probabilities for $\mathcal{L}_{\text{NLI}} \in \{\text{entailment}, \text{contradiction}, \text{neutral}\}$. If NLI confidence for entailment exceeds $0.70$, the span is flagged as false-positive detection and preserved.

### Module 5: Type-Aware Correction Controller
Rather than blindly invoking generic LLM regeneration, the controller implements decoupled, deterministic policies tailored to error typology:
- **Policy A (Evident Conflict):** Direct contradiction. Replace conflicting span with evidence-grounded fact without introducing peripheral claims.
- **Policy B (Subtle Conflict):** Nuanced misrepresentation. Rewrite the distorted clause to accurately reflect context while leaving surrounding valid sentences unchanged.
- **Policy C (Evident Baseless Info):** Fabricated entity or number. Excise the baseless span directly; do not hallucinate a synthetic replacement.
- **Policy D (Subtle Baseless Info):** Unsupported inference. Qualify the speculative clause (e.g., hedge with attribution) or excise.

**Reverse-Offset Execution:** Edits are performed in decreasing order of start offset ($s_{|\mathcal{H}|} > \dots > s_1$) ensuring character index integrity across multi-span corrections.

### Module 6: Post-Correction Verification & Regression Guard
Post-correction introduces the risk of secondary hallucinations. We introduce the **New Hallucination Rate (NHR)**:
$$\text{NHR} = \frac{|\mathcal{H}_{\text{corrected}} \setminus \mathcal{H}_{\text{original}}|}{\max(|\mathcal{H}_{\text{original}}|, 1)}$$

The pipeline bounds iterative refinement to at most 2 rounds. If NHR $> 0$ (a regression is detected), the controller triggers an automated safety reversion to the original answer or abstains, ensuring corrections never degrade answer reliability.

---

## 4. Evaluation Protocol

1. **Token-Level Metrics:** Precision, Recall, Macro-F1 across all 9 classes, and Binary Hallucination F1 (hallucinated vs. supported).
2. **Span-Level Metrics:** Exact Span F1 (exact character match) and Overlap Span F1 (boundary intersection with matching category).
3. **Calibration Metrics:** Uncalibrated vs. Calibrated ECE and Brier Score.
4. **Correction Quality:** Correction Success Rate, Hallucination Reduction Rate, and New Hallucination Rate (NHR).
5. **Ablation Matrix:**
   - Full TACER-RAG
   - Ablation 1: No Calibration (raw logits)
   - Ablation 2: No Type Awareness (generic replacement)
   - Ablation 3: No NLI Verification
   - Ablation 4: No Post-Correction Verification
   - Ablation 5: No Abstention
