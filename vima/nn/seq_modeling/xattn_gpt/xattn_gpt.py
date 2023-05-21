from __future__ import annotations

import torch
import torch.nn as nn
from transformers.models.openai.modeling_openai import (
    OpenAIGPTPreTrainedModel,
    OpenAIGPTConfig,
)

from .components import XAttention, Block


class XAttnGPT(OpenAIGPTPreTrainedModel):
    def __init__(
        self,
        embd_dim: int = 768,
        *,
        n_positions: int = 512,
        n_layer: int = 12,
        n_head: int = 12,
        dropout: float = 0.1,
        xattn_n_head: int = 8,
        xattn_ff_expanding: int = 4,
        xattn_detach_qk: bool = False,
        xattn_n_positions: int,
        use_geglu: bool = False,
    ):
        kwargs = {}
        if use_geglu:
            kwargs["afn"] = "geglu"
        cfg = OpenAIGPTConfig(
            n_positions=n_positions,
            n_embd=embd_dim,
            n_layer=n_layer,
            n_head=n_head,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            summary_first_dropout=dropout,
            **kwargs,
        )

        super().__init__(cfg)

        self.positions_embed = nn.Embedding(n_positions, embd_dim)
        self.xattn_positions_embed = nn.Embedding(xattn_n_positions, embd_dim)
        self.drop = nn.Dropout(cfg.embd_pdrop)
        self.h = nn.ModuleList(
            [Block(n_positions, cfg, scale=True) for _ in range(n_layer)]
        )
        self.xattns = nn.ModuleList(
            [
                XAttention(
                    embd_dim,
                    num_heads=xattn_n_head,
                    ff_expanding=xattn_ff_expanding,
                    detach_qk=xattn_detach_qk,
                    kv_n_positions=xattn_n_positions,
                    auto_add_pos_embd=False,
                    use_geglu=use_geglu,
                )
                for _ in range(n_layer)
            ]
        )

        self.register_buffer("position_ids", torch.arange(n_positions))
        self.register_buffer("xattn_position_ids", torch.arange(xattn_n_positions))
        # Initialize weights and apply final processing
        self.post_init()

        self._input_checked = False

    def forward(
        self,
        *,
        obs_action_tokens: torch.Tensor,
        obs_action_position_ids: torch.LongTensor | None = None,
        prompt_tokens: torch.Tensor,
        prompt_mask: torch.Tensor | None = None,
        prompt_position_ids: torch.LongTensor | None = None,
        batch_first: bool = False,
        obs_action_masks: torch.Tensor | None = None,
    ):
        if not self._input_checked:
            self._check_input(
                obs_action_tokens,
                prompt_tokens,
                prompt_mask,
                batch_first,
                obs_action_masks,
            )
            self._input_checked = True
        if batch_first:
            B_oa, L_oa, E_oa = obs_action_tokens.shape
        else:
            L_oa, B_oa, E_oa = obs_action_tokens.shape
            obs_action_tokens = obs_action_tokens.transpose(0, 1)
            prompt_tokens = prompt_tokens.transpose(0, 1)
        input_shape = obs_action_tokens.size()[:-1]

        if obs_action_position_ids is None:
            obs_action_position_ids = self.position_ids[None, : input_shape[-1]]
        position_embeds = self.positions_embed(obs_action_position_ids)

        obs_action_tokens = obs_action_tokens + position_embeds
        obs_action_tokens = self.drop(obs_action_tokens)

        output_shape = input_shape + (obs_action_tokens.size(-1),)

        assert prompt_tokens.size(1) <= self.xattn_position_ids.size(0)
        if prompt_position_ids is None:
            prompt_position_ids = self.xattn_position_ids[None, : prompt_tokens.size(1)]
        prompt_position_embds = self.xattn_positions_embed(prompt_position_ids)
        prompt_tokens = prompt_tokens + prompt_position_embds

        if obs_action_masks is not None:
            obs_action_masks = obs_action_masks.unsqueeze(1).unsqueeze(2)
            obs_action_masks = obs_action_masks.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            obs_action_masks = (1.0 - obs_action_masks) * torch.finfo(self.dtype).min

        for self_attn, xattn in zip(self.h, self.xattns):
            obs_action_tokens = xattn(
                q=obs_action_tokens,
                kv=prompt_tokens,
                attention_mask=prompt_mask,
                kv_position_ids=None,
            )
            obs_action_tokens = self_attn(
                obs_action_tokens, attention_mask=obs_action_masks
            )[0]

        obs_action_tokens = obs_action_tokens.view(*output_shape)
        assert obs_action_tokens.shape == (B_oa, L_oa, E_oa)
        if not batch_first:
            obs_action_tokens = obs_action_tokens.transpose(0, 1)

        return obs_action_tokens

    def _check_input(
        self,
        obs_action_tokens: torch.Tensor,
        prompt_tokens: torch.Tensor,
        prompt_mask: torch.Tensor | None = None,
        batch_first: bool = False,
        obs_action_masks: torch.Tensor | None = None,
    ):
        assert obs_action_tokens.dim() == 3
        assert obs_action_tokens.dtype == torch.float32
        assert prompt_tokens.dim() == 3
        assert prompt_tokens.dtype == torch.float32

        if batch_first:
            B_oa, L_oa, E_oa = obs_action_tokens.shape
            B_p, L_p, E_p = prompt_tokens.shape
        else:
            L_oa, B_oa, E_oa = obs_action_tokens.shape
            L_p, B_p, E_p = prompt_tokens.shape
        assert B_oa == B_p
        assert E_oa == E_p
        B = B_oa

        if prompt_mask is not None:
            # fmt: off
            assert prompt_mask.shape == (B, L_p) or prompt_mask.shape == (B, 1, L_p), \
                f"Expect `prompt_mask` to have shape of either ({B, 1, L_p}) or ({B, L_p}), but got {prompt_mask.shape}"
            # fmt: on
            # a simple sanity check on the mask
            assert torch.all(
                prompt_mask.sum(dim=-1) > 0
            ), "each source token should attend to at least one target token"
            assert prompt_mask.dtype == torch.bool
        if obs_action_masks is not None:
            assert obs_action_masks.shape == (B, L_oa)
            assert torch.all(obs_action_masks.sum(dim=-1) > 0)
            assert obs_action_masks.dtype == torch.bool
