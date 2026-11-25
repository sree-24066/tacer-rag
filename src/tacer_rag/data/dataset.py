"""
RAGTruth PyTorch Dataset for token-classification with ModernBERT.

Design decisions:
- Input format: [CLS] context [SEP] answer [SEP]
  Context tokens get label -100 (ignored in loss).
  Answer tokens get BIO labels or 0 (O) if not hallucinated.
- Character offsets from annotations are aligned to token offsets
  using fast tokenizer's offset_mapping.
- Overlapping spans: if two spans overlap a token, the first span wins.
  This is rare in RAGTruth but documented here.
- Long contexts: truncated to max_length. Spans beyond truncation point
  are dropped and counted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset

# Label constants (must match configs/data.yaml)
LABEL_MAP = {
    "O": 0,
    "B-EVIDENT_CONFLICT": 1, "I-EVIDENT_CONFLICT": 2,
    "B-SUBTLE_CONFLICT":  3, "I-SUBTLE_CONFLICT":  4,
    "B-EVIDENT_BASELESS": 5, "I-EVIDENT_BASELESS": 6,
    "B-SUBTLE_BASELESS":  7, "I-SUBTLE_BASELESS":  8,
}
ID_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}
N_LABELS = len(LABEL_MAP)
IGNORE_INDEX = -100  # tokens not part of the answer (CLS, SEP, context tokens)

RAGTRUTH_TO_BIO_PREFIX = {
    "Evident Conflict":      "EVIDENT_CONFLICT",
    "Subtle Conflict":       "SUBTLE_CONFLICT",
    "Evident Baseless Info": "EVIDENT_BASELESS",
    "Subtle Baseless Info":  "SUBTLE_BASELESS",
}


@dataclass
class RagTruthExample:
    """Single preprocessed example ready for tokenization."""
    record_id: int
    source_id: int
    model: str
    task_type: str
    split: str
    context: str           # source_info text (evidence)
    answer: str            # generated response
    # List of (char_start, char_end, bio_prefix) in answer character space
    hallucinated_spans: list[tuple[int, int, str]]


def load_examples(jsonl_path: str | Path) -> list[RagTruthExample]:
    """Load processed JSONL and convert to RagTruthExample objects."""
    examples = []
    path = Path(jsonl_path)
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            # Build context string from source_info
            source_info = rec.get("source_info", "")
            if isinstance(source_info, dict):
                # QA task: source_info is a dict with 'passages' key
                passages = source_info.get("passages", [])
                if isinstance(passages, list):
                    context = " ".join(
                        p.get("passage", p) if isinstance(p, dict) else str(p)
                        for p in passages
                    )
                else:
                    context = str(source_info)
            else:
                context = str(source_info)

            answer = rec.get("response", "")

            # Build hallucinated span list
            spans = []
            for lbl in rec.get("labels", []):
                lt = lbl.get("label_type", "")
                bio_prefix = RAGTRUTH_TO_BIO_PREFIX.get(lt)
                if bio_prefix is None:
                    continue  # unknown label type — skip
                spans.append((lbl["start"], lbl["end"], bio_prefix))

            # Sort spans by start position for deterministic B/I assignment
            spans.sort(key=lambda x: x[0])

            examples.append(RagTruthExample(
                record_id=rec.get("id", -1),
                source_id=rec.get("source_id", -1),
                model=rec.get("model", ""),
                task_type=rec.get("task_type", ""),
                split=rec.get("split", ""),
                context=context,
                answer=answer,
                hallucinated_spans=spans,
            ))
    return examples


class RagTruthTokenDataset(Dataset):
    """
    Token-classification dataset for ModernBERT.

    Each item is a dict with:
        input_ids, attention_mask, token_type_ids (optional),
        labels (token-level, -100 for non-answer tokens)
    """

    def __init__(
        self,
        examples: list[RagTruthExample],
        tokenizer,
        max_length: int = 512,
        truncation_stats: Optional[dict] = None,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation_stats = truncation_stats or {"dropped_spans": 0, "total_spans": 0}
        self._encodings = self._encode_all()

    def _encode_all(self) -> list[dict]:
        encodings = []
        for ex in self.examples:
            enc = self._encode_one(ex)
            if enc is not None:
                encodings.append(enc)
        return encodings

    def _encode_one(self, ex: RagTruthExample) -> Optional[dict]:
        """
        Tokenize (context, answer) pair and align hallucination labels to tokens.

        Returns dict with tensors, or None if encoding fails.
        """
        tokenizer = self.tokenizer
        answer = ex.answer
        context = ex.context

        # Full encoding: (context, answer) as a pair
        # Use truncation=True (longest-first) — "only_first" crashes when
        # combined length < max_length. Longest-first is safe in all cases.
        encoding = tokenizer(
            context,
            answer,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offset_mapping = encoding["offset_mapping"].squeeze(0).tolist()
        # token_type_ids: 0 = context segment, 1 = answer segment
        token_type_ids = encoding.get("token_type_ids")
        if token_type_ids is not None:
            token_type_ids = token_type_ids.squeeze(0)

        seq_len = len(input_ids)
        labels = [IGNORE_INDEX] * seq_len

        # --- Identify answer token indices ---
        # Strategy: use token_type_ids if available (most tokenizers)
        # Fallback: find the second SEP and treat everything after as answer
        if token_type_ids is not None:
            answer_token_mask = [i for i, t in enumerate(token_type_ids.tolist()) if t == 1]
        else:
            # Find second SEP token
            sep_id = tokenizer.sep_token_id
            sep_positions = [i for i, tid in enumerate(input_ids.tolist()) if tid == sep_id]
            if len(sep_positions) >= 2:
                first_sep = sep_positions[0]
                second_sep = sep_positions[1]
                answer_token_mask = list(range(first_sep + 1, second_sep))
            else:
                answer_token_mask = []

        # --- Re-encode answer alone to get correct character offsets ---
        # The pair encoding offset_mapping refers to individual sequence chars.
        # We need to find, for each answer token in the pair, its char offset
        # within the answer string.
        answer_only_enc = tokenizer(
            answer,
            truncation=True,
            max_length=self.max_length,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        answer_offsets = answer_only_enc["offset_mapping"]  # list of (start, end) in answer

        # Map answer tokens in the pair encoding to answer character positions
        # We match by position: the i-th non-special answer token in the pair
        # corresponds to the i-th token in the standalone answer encoding.
        pair_answer_tokens = [i for i in answer_token_mask
                              if input_ids[i].item() not in
                              (tokenizer.cls_token_id, tokenizer.sep_token_id,
                               tokenizer.pad_token_id)
                              and input_ids[i].item() != 0]

        # Initialize answer tokens to O (not hallucinated)
        for tok_pos in pair_answer_tokens:
            labels[tok_pos] = LABEL_MAP["O"]

        # --- Align hallucination spans to answer tokens ---
        self.truncation_stats["total_spans"] += len(ex.hallucinated_spans)

        for span_start_char, span_end_char, bio_prefix in ex.hallucinated_spans:
            b_tag = LABEL_MAP.get(f"B-{bio_prefix}", LABEL_MAP["O"])
            i_tag = LABEL_MAP.get(f"I-{bio_prefix}", LABEL_MAP["O"])

            first_in_span = True
            for idx, (pair_tok_pos) in enumerate(pair_answer_tokens):
                if idx >= len(answer_offsets):
                    self.truncation_stats["dropped_spans"] += 1
                    break

                tok_char_start, tok_char_end = answer_offsets[idx]
                if tok_char_start is None or tok_char_end is None:
                    continue
                if tok_char_end == 0:
                    continue  # padding / special

                # Check overlap with hallucination span
                overlaps = (tok_char_start < span_end_char and tok_char_end > span_start_char)
                if overlaps:
                    # Only assign if not already labelled (first span wins on overlap)
                    if labels[pair_tok_pos] == LABEL_MAP["O"]:
                        if first_in_span:
                            labels[pair_tok_pos] = b_tag
                            first_in_span = False
                        else:
                            labels[pair_tok_pos] = i_tag

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long),
            "record_id": ex.record_id,
        }
        if token_type_ids is not None:
            result["token_type_ids"] = token_type_ids
        return result

    def __len__(self) -> int:
        return len(self._encodings)

    def __getitem__(self, idx: int) -> dict:
        return self._encodings[idx]
