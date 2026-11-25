"""
ModernBERT Token-Classification Model for Hallucination Detection.

Wraps a ModernBERT (or any BERT-family) encoder with a token-classification head.
Outputs per-token logits over 9 BIO classes (O + 4 types × B/I).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import TokenClassifierOutput


@dataclass
class DetectorConfig:
    encoder_name: str = "answerdotai/ModernBERT-base"
    num_labels: int = 9       # O + 4 × (B + I)
    dropout: float = 0.1
    max_length: int = 512
    class_weights: Optional[list[float]] = None


class HallucinationDetector(nn.Module):
    """
    Token-classification wrapper around a pretrained encoder.

    Can be used with:
        - answerdotai/ModernBERT-base
        - bert-base-uncased (for smoke testing without large downloads)
        - Any HuggingFace BERT-compatible encoder
    """

    def __init__(self, config: DetectorConfig):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.register_buffer(
            "class_weights",
            torch.tensor(config.class_weights, dtype=torch.float)
            if config.class_weights is not None else None,
        )

        hf_config = AutoConfig.from_pretrained(
            config.encoder_name,
            num_labels=config.num_labels,
        )
        self.encoder = AutoModel.from_pretrained(
            config.encoder_name,
            config=hf_config,
            ignore_mismatched_sizes=True,
        )
        hidden_size = hf_config.hidden_size
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(hidden_size, config.num_labels)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> TokenClassifierOutput:

        encoder_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Only pass token_type_ids if the model supports it
        if token_type_ids is not None:
            try:
                outputs = self.encoder(
                    **encoder_kwargs, token_type_ids=token_type_ids
                )
            except TypeError:
                outputs = self.encoder(**encoder_kwargs)
        else:
            outputs = self.encoder(**encoder_kwargs)

        sequence_output = outputs.last_hidden_state   # (B, L, H)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)     # (B, L, num_labels)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(
                weight=self.class_weights,
                ignore_index=-100,
            )
            loss = loss_fn(
                logits.view(-1, self.num_labels),
                labels.view(-1),
            )

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
            attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
        )

    @classmethod
    def from_pretrained_encoder(cls, encoder_name: str, **kwargs) -> "HallucinationDetector":
        config = DetectorConfig(encoder_name=encoder_name, **kwargs)
        return cls(config)
