from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn as nn

from .dists import Categorical, MultiCategorical
from ..utils import build_mlp


class ActionDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        action_dims: dict[str, int | list[int]],
        hidden_dim: int,
        hidden_depth: int,
        activation: str | Callable = "relu",
        norm_type: Literal["batchnorm", "layernorm"] | None = None,
        last_layer_gain: float | None = 0.01,
    ):
        super().__init__()

        self._decoders = nn.ModuleDict()
        for k, v in action_dims.items():
            if isinstance(v, int):
                self._decoders[k] = CategoricalNet(
                    input_dim,
                    action_dim=v,
                    hidden_dim=hidden_dim,
                    hidden_depth=hidden_depth,
                    activation=activation,
                    norm_type=norm_type,
                    last_layer_gain=last_layer_gain,
                )
            elif isinstance(v, list):
                self._decoders[k] = MultiCategoricalNet(
                    input_dim,
                    action_dims=v,
                    hidden_dim=hidden_dim,
                    hidden_depth=hidden_depth,
                    activation=activation,
                    norm_type=norm_type,
                    last_layer_gain=last_layer_gain,
                )
            else:
                raise ValueError(f"Invalid action_dims value: {v}")

    def forward(self, x: torch.Tensor):
        return {k: v(x) for k, v in self._decoders.items()}


def _build_mlp_distribution_net(
    input_dim: int,
    *,
    output_dim: int,
    hidden_dim: int,
    hidden_depth: int,
    activation: str | Callable = "relu",
    norm_type: Literal["batchnorm", "layernorm"] | None = None,
    last_layer_gain: float | None = 0.01,
):
    """
    Use orthogonal initialization to initialize the MLP policy

    Args:
        last_layer_gain: orthogonal initialization gain for the last FC layer.
            you may want to set it to a small value (e.g. 0.01) to have the
            Gaussian centered around 0.0 in the beginning.
            Set to None to use the default gain (dependent on the NN activation)
    """

    mlp = build_mlp(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        hidden_depth=hidden_depth,
        activation=activation,
        weight_init="orthogonal",
        bias_init="zeros",
        norm_type=norm_type,
    )
    if last_layer_gain:
        assert last_layer_gain > 0
        nn.init.orthogonal_(mlp[-1].weight, gain=last_layer_gain)
    return mlp


class CategoricalNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        action_dim: int,
        hidden_dim: int,
        hidden_depth: int,
        activation: str | Callable = "relu",
        norm_type: Literal["batchnorm", "layernorm"] | None = None,
        last_layer_gain: float | None = 0.01,
    ):
        """
        Use orthogonal initialization to initialize the MLP policy

        Args:
            last_layer_gain: orthogonal initialization gain for the last FC layer.
                you may want to set it to a small value (e.g. 0.01) to make the
                Categorical close to uniform random at the beginning.
                Set to None to use the default gain (dependent on the NN activation)
        """
        super().__init__()
        self.mlp = _build_mlp_distribution_net(
            input_dim=input_dim,
            output_dim=action_dim,
            hidden_dim=hidden_dim,
            hidden_depth=hidden_depth,
            activation=activation,
            norm_type=norm_type,
            last_layer_gain=last_layer_gain,
        )
        self.head = CategoricalHead()

    def forward(self, x):
        return self.head(self.mlp(x))


class MultiCategoricalNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        action_dims: list[int],
        hidden_dim: int,
        hidden_depth: int,
        activation: str | Callable = "relu",
        norm_type: Literal["batchnorm", "layernorm"] | None = None,
        last_layer_gain: float | None = 0.01,
    ):
        """
        Use orthogonal initialization to initialize the MLP policy
        Split head, does not share the NN weights

        Args:
            last_layer_gain: orthogonal initialization gain for the last FC layer.
                you may want to set it to a small value (e.g. 0.01) to make the
                Categorical close to uniform random at the beginning.
                Set to None to use the default gain (dependent on the NN activation)
        """
        super().__init__()
        self.mlps = nn.ModuleList()
        for action in action_dims:
            net = _build_mlp_distribution_net(
                input_dim=input_dim,
                output_dim=action,
                hidden_dim=hidden_dim,
                hidden_depth=hidden_depth,
                activation=activation,
                norm_type=norm_type,
                last_layer_gain=last_layer_gain,
            )
            self.mlps.append(net)
        self.head = MultiCategoricalHead(action_dims)

    def forward(self, x):
        return self.head(torch.cat([mlp(x) for mlp in self.mlps], dim=-1))


class CategoricalHead(nn.Module):
    def forward(self, x: torch.Tensor) -> Categorical:
        return Categorical(logits=x)


class MultiCategoricalHead(nn.Module):
    def __init__(self, action_dims: list[int]):
        super().__init__()
        self._action_dims = tuple(action_dims)

    def forward(self, x: torch.Tensor) -> MultiCategorical:
        return MultiCategorical(logits=x, action_dims=self._action_dims)
