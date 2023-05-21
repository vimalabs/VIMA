from __future__ import annotations

import torch
import torch.nn as nn
from transformers.models.perceiver.modeling_perceiver import (
    PerceiverConfig,
    PerceiverModel,
)


class ObjectsPerceiverEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        *,
        num_latents: int,
        num_blocks: int,
        num_self_attends_per_block: int,
        num_self_attention_heads: int,
        num_cross_attention_heads: int,
        attention_probs_dropout_prob: float,
    ):
        super().__init__()

        cfg = PerceiverConfig(
            d_model=embed_dim,
            d_latents=embed_dim,
            num_latents=num_latents,
            num_blocks=num_blocks,
            num_self_attends_per_block=num_self_attends_per_block,
            num_self_attention_heads=num_self_attention_heads,
            num_cross_attention_heads=num_cross_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
        )
        self.model = PerceiverModel(cfg)
        self.output_dim = embed_dim
        self._num_queries = num_latents

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        out = self.model(inputs=x, attention_mask=mask).last_hidden_state
        return out
