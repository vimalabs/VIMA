from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class WordEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        model = AutoModel.from_pretrained("t5-base")
        embed_weight = model.get_input_embeddings().weight.data
        _emb_dim = embed_weight.shape[1]
        self._embed_layer = nn.Embedding.from_pretrained(embed_weight)
        del model
        self.output_dim = _emb_dim

    def forward(self, x: torch.Tensor):
        """
        x: any shape
        """
        x = self._embed_layer(x)
        return x
