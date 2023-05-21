from __future__ import annotations

import torch
import torch.nn as nn

from ..utils import build_mlp


class ActionEmbedding(nn.Module):
    def __init__(self, output_dim: int, *, embed_dict: dict[str, nn.Module]):
        super().__init__()
        self._embed_dict = nn.ModuleDict(embed_dict)
        embed_dict_output_dim = sum(
            embed_dict[k].output_dim for k in sorted(embed_dict.keys())
        )
        self._post_layer = (
            nn.Identity()
            if output_dim == embed_dict_output_dim
            else nn.Linear(embed_dict_output_dim, output_dim)
        )
        self._output_dim = output_dim

        self._input_fields_checked = False

    @property
    def output_dim(self):
        return self._output_dim

    def forward(self, x_dict: dict[str, torch.Tensor]):
        if not self._input_fields_checked:
            assert set(x_dict.keys()) == set(self._embed_dict.keys())
            self._input_fields_checked = True
        return self._post_layer(
            torch.cat(
                [self._embed_dict[k](x_dict[k]) for k in sorted(x_dict.keys())], dim=-1
            )
        )


class ContinuousActionEmbedding(nn.Module):
    def __init__(
        self, output_dim: int, *, input_dim: int, hidden_dim: int, hidden_depth: int
    ):
        super().__init__()

        self._layer = build_mlp(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            hidden_depth=hidden_depth,
        )

        self.output_dim = output_dim

    def forward(self, x: torch.Tensor):
        return self._layer(x)
