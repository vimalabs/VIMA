from __future__ import annotations

import math

import torch
import torch.nn as nn
from transformers.models.openai.modeling_openai import (
    Attention as _Attention,
    Conv1D,
    ACT_FNS,
)


class Block(nn.Module):
    def __init__(self, n_positions, config, scale=False):
        super().__init__()
        nx = config.n_embd
        self.attn = Attention(nx, n_positions, config, scale)
        self.ln_1 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.mlp = MLP(4 * nx, config)
        self.ln_2 = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)

    def forward(self, x, attention_mask=None, head_mask=None, output_attentions=False):
        attn_outputs = self.attn(
            x,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        a = attn_outputs[0]

        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)

        outputs = [h] + attn_outputs[1:]
        return outputs


class Attention(_Attention):
    def _attn(
        self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False
    ):
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        # w = w * self.bias + -1e9 * (1 - self.bias)  # TF implementation method: mask_attn_weights
        # XD: self.b may be larger than w, so we need to crop it
        b = self.bias[:, :, : w.size(-2), : w.size(-1)]
        b = b.to(w.dtype)
        w = w * b + -1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.functional.softmax(w, dim=-1)
        w = w.to(v.dtype)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs


class MLP(nn.Module):
    def __init__(self, n_state, config):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        if config.afn == "geglu":
            self.act = nn.GELU()
            self.gated_layer = nn.Linear(config.n_embd, n_state, bias=False)
        else:
            self.act = ACT_FNS[config.afn]
            self.gated_layer = None
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        if self.gated_layer is not None:
            h = h * self.gated_layer(x)
        h2 = self.c_proj(h)
        return self.dropout(h2)


class XAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        num_heads=1,
        ff_expanding: int = 4,
        kv_n_positions: int,
        detach_qk: bool = False,
        fp32_logits: bool = True,
        auto_add_pos_embd: bool = True,
        use_geglu: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        if dim % num_heads != 0:
            raise ValueError(
                f"dim ({dim}) must be divisible by num_heads ({num_heads})."
            )
        self.dim = dim
        self.dim_per_head = dim // num_heads

        # Layer normalization
        self.layernorm = nn.LayerNorm(dim)
        # Projection matrices
        self.query = nn.Linear(dim, dim, bias=False)
        self.key_value = nn.Linear(dim, 2 * dim, bias=False)
        self.attention_out = nn.Linear(dim, dim, bias=False)

        inner_dim = int(dim * ff_expanding)
        self.ln = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, inner_dim, bias=False)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(inner_dim, dim, bias=False)
        if use_geglu:
            self.gated_layer = nn.Linear(dim, inner_dim, bias=False)
        else:
            self.gated_layer = None
        if auto_add_pos_embd:
            self.kv_positions_embed = nn.Embedding(kv_n_positions, dim)
        else:
            self.kv_positions_embed = None

        self.register_buffer("kv_position_ids", torch.arange(kv_n_positions))

        self._detach_qk = detach_qk
        self._fp32_logits = fp32_logits

    def transpose_for_scores(self, x, channels_per_head):
        new_x_shape = x.size()[:-1] + (self.num_heads, channels_per_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        *,
        q: torch.Tensor,
        kv: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        kv_position_ids: torch.LongTensor | None = None,
    ):
        queries = self.layernorm(q)
        queries = self.query(queries)

        if kv_position_ids is None:
            assert kv.shape[1] <= self.kv_position_ids.shape[0]
            kv_position_ids = self.kv_position_ids[None, : kv.shape[1]]
        if self.kv_positions_embed is not None:
            kv_position_embeds = self.kv_positions_embed(kv_position_ids)
            kv = kv + kv_position_embeds
        keys, values = self.key_value(kv).chunk(2, dim=-1)

        # Reshape channels for multi-head attention.
        # We reshape from (batch_size, time, channels) to (batch_size, num_heads, time, channels per head)
        queries = self.transpose_for_scores(queries, self.dim_per_head)
        keys = self.transpose_for_scores(keys, self.dim_per_head)
        values = self.transpose_for_scores(values, self.dim_per_head)

        # Take the dot product between the queries and keys to get the raw attention scores.
        if self._fp32_logits:
            queries = queries.to(torch.float32)
            keys = keys.to(torch.float32)
        attention_scores = torch.matmul(
            queries, keys.transpose(-1, -2)
        )  # (B, NH, T_q, T_k)

        batch_size, num_heads, q_len, q_head_dim = queries.shape
        _, _, kv_len, v_head_dim = values.shape
        hiddens = self.num_heads * v_head_dim

        attention_scores = attention_scores / math.sqrt(q_head_dim)

        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, kv_len)
            assert attention_mask.dtype == torch.bool
            attention_mask = self.invert_attention_mask(attention_mask)
            attention_mask = attention_mask.to(attention_scores.dtype)
            attention_scores = attention_scores + attention_mask

        if self._detach_qk:
            attention_scores = attention_scores.detach()
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = attention_probs.to(values.dtype)

        context_layer = torch.matmul(attention_probs, values)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (hiddens,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # Output projection
        attention_output = self.attention_out(context_layer)
        attention_output = attention_output + q

        ff_output = self.ln(attention_output)
        ff_output = self.linear1(ff_output)
        ff_output = self.act(ff_output)
        if self.gated_layer is not None:
            ff_output = ff_output * self.gated_layer(attention_output)
        ff_output = self.linear2(ff_output)

        output = ff_output + attention_output
        return output

    def invert_attention_mask(self, encoder_attention_mask):
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.

        Returns:
            `torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(
            dtype=self.dtype
        )  # fp16 compatibility
        encoder_extended_attention_mask = (
            1.0 - encoder_extended_attention_mask
        ) * torch.finfo(self.dtype).min

        return encoder_extended_attention_mask

    @property
    def dtype(self) -> torch.dtype:
        """
        `torch.dtype`: The dtype of the module (assuming that all the module parameters have the same dtype).
        """
        return get_parameter_dtype(self)


def get_parameter_dtype(parameter):
    """
    Returns the first found floating dtype in parameters if there is one, otherwise returns the last dtype it found.
    """
    last_dtype = None
    for t in parameter.parameters():
        last_dtype = t.dtype
        if t.is_floating_point():
            return t.dtype

    if last_dtype is not None:
        # if no floating dtype was found return whatever the first dtype is
        return last_dtype
