from __future__ import annotations

import torch
import torch.nn as nn

import vima.nn as vnn
from ..utils import *


class VIMAGPTPolicy(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        vocab_size=40478,
        n_positions=512,
        n_layer=12,
        n_head=12,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.transformer = vnn.HFGPT(
            n_embd=embed_dim,
            use_geglu=True,
            vocab_size=vocab_size,
            n_positions=n_positions,
            n_layer=n_layer,
            n_head=n_head,
            dropout=dropout,
        )
        self.prompt_sep_token = nn.Parameter(torch.zeros(embed_dim))

        self.obj_encoder = vnn.MultiViewRGBEncoder(
            img_size=(64, 128),
            emb_dim=embed_dim,
            views=["front", "top"],
            vit_patch_size=32,
            vit_width=768,
            vit_layers=4,
            vit_heads=24,
        )

        self.end_effector_encoder = vnn.Embedding(num_embeddings=2, embedding_dim=2)

        obs_feat_dim = self.obj_encoder.output_dim + 2
        self.obs_fusion_layer = (
            nn.Identity()
            if obs_feat_dim == embed_dim
            else nn.Linear(obs_feat_dim, embed_dim)
        )

        self.action_encoder = vnn.ActionEmbedding(
            output_dim=embed_dim,
            embed_dict={
                "pose0_position": vnn.ContinuousActionEmbedding(
                    output_dim=256,
                    input_dim=2,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
                "pose0_rotation": vnn.ContinuousActionEmbedding(
                    output_dim=256,
                    input_dim=4,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
                "pose1_position": vnn.ContinuousActionEmbedding(
                    output_dim=256,
                    input_dim=2,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
                "pose1_rotation": vnn.ContinuousActionEmbedding(
                    output_dim=256,
                    input_dim=4,
                    hidden_dim=256,
                    hidden_depth=1,
                ),
            },
        )
        self.action_decoder = vnn.ActionDecoder(
            input_dim=embed_dim,
            action_dims={
                "pose0_position": [50, 100],
                "pose0_rotation": [50] * 4,
                "pose1_position": [50, 100],
                "pose1_rotation": [50] * 4,
            },
            hidden_dim=512,
            hidden_depth=2,
            activation="relu",
            norm_type=None,
            last_layer_gain=0.01,
        )

        self.prompt_embedding = vnn.WordEmbedding()
        self.t5_prompt_encoder = vnn.T5PromptEncoder()
        self.t5_prompt_encoder_post_layer = (
            nn.Identity()
            if embed_dim == self.t5_prompt_encoder.output_dim
            else nn.Linear(self.t5_prompt_encoder.output_dim, embed_dim, bias=False)
        )
        self.prompt_obj_post_layer = vnn.build_mlp(
            self.obj_encoder.output_dim,
            hidden_dim=768,
            output_dim=768,
            hidden_depth=2,
        )

        self._views = ["front", "top"]
        self._n_discrete_x_bins = 50
        self._n_discrete_y_bins = 100
        self._n_discrete_z_bins = 50
        self._n_discrete_rot_bins = 50

    def forward(
        self,
        obs_token: torch.Tensor,
        action_token: torch.Tensor | None,
        prompt_token: torch.Tensor,
        prompt_token_mask: torch.Tensor,
    ):
        B = obs_token.shape[1]
        L_obs, L_prompt = obs_token.shape[0], prompt_token.shape[0]
        L_action = 0 if action_token is None else action_token.shape[0]
        L = L_obs + L_action + L_prompt + 1

        tokens = torch.empty(
            L, B, self.embed_dim, dtype=torch.float32, device=self.device
        )
        tokens[:L_prompt] = prompt_token
        tokens[L_prompt] = self.prompt_sep_token.unsqueeze(0).repeat(B, 1)
        tokens[L_prompt + 1 :: 2] = obs_token
        if action_token is not None:
            tokens[L_prompt + 2 :: 2] = action_token
        mask = torch.cat(
            [
                prompt_token_mask,
                torch.ones((B, L - L_prompt), dtype=torch.bool, device=self.device),
            ],
            dim=1,
        )
        mask = mask.unsqueeze(1)
        n_valid_prompt_tokens = prompt_token_mask.sum(dim=1)
        prompt_position_ids = any_stack(
            [
                any_concat(
                    [
                        torch.arange(n_valids, dtype=torch.long, device=self.device),
                        torch.zeros(
                            L_prompt - n_valids, dtype=torch.long, device=self.device
                        ).fill_(n_valids - 1),
                    ],
                    dim=0,
                )
                for n_valids in n_valid_prompt_tokens
            ],
            dim=0,
        )
        seq_position_ids = any_stack(
            [
                torch.arange(
                    start=n_valids,
                    end=n_valids + L_obs + L_action + 1,
                    dtype=torch.long,
                    device=self.device,
                )
                for n_valids in n_valid_prompt_tokens
            ],
            dim=0,
        )
        position_ids = any_concat([prompt_position_ids, seq_position_ids], dim=1)
        tokens_out = self.transformer(
            tokens, custom_mask=mask, batch_first=False, position_ids=position_ids
        )
        predicted_action_tokens = tokens_out[L_prompt + 1 :: 2]
        return predicted_action_tokens

    def forward_prompt_assembly(self, prompts):
        raw_prompts_token_type, word_batch, image_batch = prompts
        B = len(raw_prompts_token_type)
        L_max = 0
        for raw_prompt in raw_prompts_token_type:
            L_this = 0
            for item in raw_prompt:
                if item == 0:
                    L_this += 1
                elif item == 1:
                    L_this += 1
                else:
                    raise ValueError(f"Invalid prompt token type {item}")
            L_max = max(L_max, L_this)
        n_words = word_batch.shape[0]
        batch_word_emb = self.prompt_embedding(word_batch)
        n_img = len(list(image_batch["rgb"].values())[0])
        batch_image_emb = self.obj_encoder(**image_batch)
        batch_image_emb = self.prompt_obj_post_layer(batch_image_emb)
        prompt_tokens, prompt_masks = [], []
        word_ptr, img_ptr = 0, 0
        for raw_prompt in raw_prompts_token_type:
            assembled_prompt = []
            for item in raw_prompt:
                if item == 0:
                    assembled_prompt.append(batch_word_emb[word_ptr])
                    word_ptr += 1
                elif item == 1:
                    assembled_prompt.append(batch_image_emb[img_ptr])
                    img_ptr += 1
                else:
                    raise ValueError(f"Invalid type: {type(item)}")
            valid_tokens = len(assembled_prompt)
            num_padding = L_max - valid_tokens
            assembled_prompt = torch.stack(assembled_prompt, dim=0)
            required_padding = torch.zeros(
                (num_padding, assembled_prompt.shape[1]),
                dtype=torch.float32,
                device=self.device,
            )
            assembled_prompt = torch.cat([assembled_prompt, required_padding], dim=0)
            prompt_tokens.append(assembled_prompt)
            prompt_masks.append(
                torch.cat(
                    [
                        torch.ones(valid_tokens, dtype=torch.bool, device=self.device),
                        torch.zeros(num_padding, dtype=torch.bool, device=self.device),
                    ],
                    dim=0,
                )
            )
        prompt_tokens = torch.stack(prompt_tokens, dim=0)
        prompt_masks = torch.stack(prompt_masks, dim=0)
        prompt_tokens = prompt_tokens.transpose(0, 1)
        prompt_tokens = self.t5_prompt_encoder(
            prompt_tokens, attention_mask=prompt_masks, batch_first=False
        )
        prompt_tokens = self.t5_prompt_encoder_post_layer(prompt_tokens)
        return prompt_tokens, prompt_masks

    def forward_obs_token(
        self,
        obs,
    ):
        rgbs, ee = obs["rgb"], obs["ee"]
        leading_dims = ee.shape[:2]
        rgbs = rgbs.map_structure(func=lambda x: x.reshape(-1, *x.shape[2:]))
        img_feats = self.obj_encoder(rgb=rgbs)
        img_feats = img_feats.reshape(*leading_dims, *img_feats.shape[1:])
        ee_feats = self.end_effector_encoder(ee)
        obs_feats = self.obs_fusion_layer(torch.cat([img_feats, ee_feats], dim=-1))
        return obs_feats

    def forward_action_token(self, action):
        return self.action_encoder(self._de_discretize_actions(action))

    def forward_action_decoder(self, predicted_action_tokens: torch.Tensor):
        return self.action_decoder(predicted_action_tokens)

    def discretize_action(self, action):
        device = action["pose0_position"].device
        boundary_x = torch.linspace(
            start=0, end=1, steps=self._n_discrete_x_bins, device=device
        )
        boundary_y = torch.linspace(
            start=0, end=1, steps=self._n_discrete_y_bins, device=device
        )
        boundary_rot = torch.linspace(
            start=0, end=1, steps=self._n_discrete_rot_bins, device=device
        )

        action["pose0_position"][..., 0] = torch.bucketize(
            action["pose0_position"][..., 0].contiguous(), boundary_x
        )
        action["pose0_position"][..., 1] = torch.bucketize(
            action["pose0_position"][..., 1].contiguous(), boundary_y
        )
        action["pose0_rotation"] = torch.bucketize(
            action["pose0_rotation"].contiguous(), boundary_rot
        )

        action["pose1_position"][..., 0] = torch.bucketize(
            action["pose1_position"][..., 0].contiguous(), boundary_x
        )
        action["pose1_position"][..., 1] = torch.bucketize(
            action["pose1_position"][..., 1].contiguous(), boundary_y
        )
        action["pose1_rotation"] = torch.bucketize(
            action["pose1_rotation"].contiguous(), boundary_rot
        )
        action = {k: v.long() for k, v in action.items()}
        return action

    def _de_discretize_actions(self, actions):
        actions = {k: v.float() for k, v in actions.items()}
        actions["pose0_position"][..., 0] = (
            actions["pose0_position"][..., 0] / self._n_discrete_x_bins
        )
        actions["pose0_position"][..., 1] = (
            actions["pose0_position"][..., 1] / self._n_discrete_y_bins
        )
        actions["pose0_rotation"] = (
            actions["pose0_rotation"] / self._n_discrete_rot_bins
        )

        actions["pose1_position"][..., 0] = (
            actions["pose1_position"][..., 0] / self._n_discrete_x_bins
        )
        actions["pose1_position"][..., 1] = (
            actions["pose1_position"][..., 1] / self._n_discrete_y_bins
        )
        actions["pose1_rotation"] = (
            actions["pose1_rotation"] / self._n_discrete_rot_bins
        )
        return actions
