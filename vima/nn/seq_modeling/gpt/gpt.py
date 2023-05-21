from __future__ import annotations
from typing import Optional, Union, Tuple
import math
import torch
import torch.nn as nn
from transformers.models.openai import OpenAIGPTConfig, OpenAIGPTPreTrainedModel
from transformers.models.openai.modeling_openai import (
    Attention as _Attention,
    BaseModelOutput,
    Conv1D,
    ACT_FNS,
)


class HFGPT(nn.Module):
    def __init__(
        self,
        *,
        vocab_size=40478,
        n_positions=512,
        n_embd=768,
        n_layer=12,
        n_head=12,
        dropout: float = 0.1,
        use_geglu: bool = False,
    ):
        super().__init__()
        kwargs = {}
        if use_geglu:
            kwargs["afn"] = "geglu"
        cfg = OpenAIGPTConfig(
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            summary_first_dropout=dropout,
            **kwargs,
        )
        self.lm = OpenAIGPTModel(cfg)

    def forward(
        self,
        x: torch.Tensor,
        *,
        custom_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        batch_first: bool = False,
    ):
        """
        x: (L, B, E) if batch_first == False else (B, L, E)
        custom_mask: (B, L_tgt) or (B, 1, L_tgt) concurrently work with the causal mask
            because of self-attention, L_tgt = L
        """
        if batch_first:
            B, L, E = x.shape
        else:
            L, B, E = x.shape
            x = x.transpose(0, 1)

        attention_mask = None
        if custom_mask is not None:
            if custom_mask.dim() == 3:
                custom_mask = custom_mask.squeeze(dim=1)
            attention_mask = custom_mask.float().contiguous()
        out = self.lm(
            inputs_embeds=x.contiguous(),
            attention_mask=attention_mask,
            position_ids=position_ids,
        ).last_hidden_state
        assert out.shape == (B, L, E)
        if not batch_first:
            out = out.transpose(0, 1)
        return out


class OpenAIGPTModel(OpenAIGPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.tokens_embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList(
            [
                Block(config.n_positions, config, scale=True)
                for _ in range(config.n_layer)
            ]
        )

        self.register_buffer("position_ids", torch.arange(config.n_positions))
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.tokens_embed

    def set_input_embeddings(self, new_embeddings):
        self.tokens_embed = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutput]:
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is None:
            # Code is different from when we had a single embedding matrix  from position and token embeddings
            position_ids = self.position_ids[None, : input_shape[-1]]

        # Attention mask.
        if attention_mask is not None:
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.tokens_embed(input_ids)
        position_embeds = self.positions_embed(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.tokens_embed(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                attention_mask,
                head_mask[i],
                output_attentions=output_attentions,
            )
            hidden_states = outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (outputs[1],)

        hidden_states = hidden_states.view(*output_shape)
        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, all_hidden_states, all_attentions]
                if v is not None
            )

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
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
