from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange

import vima.nn as vnn
from ..utils import *


class VIMAPolicy(nn.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        xf_n_layers: int,
        sattn_n_heads: int,
        xattn_n_heads: int,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.xattn_gpt = vnn.XAttnGPT(
            embed_dim,
            n_layer=xf_n_layers,
            n_head=sattn_n_heads,
            dropout=0.1,
            xattn_n_head=xattn_n_heads,
            xattn_ff_expanding=4,
            xattn_n_positions=256,
            use_geglu=True,
        )

        self.obj_encoder = vnn.ObjEncoder(
            transformer_emb_dim=embed_dim,
            views=["front", "top"],
            vit_output_dim=768,
            vit_resolution=32,
            vit_patch_size=16,
            vit_width=768,
            vit_layers=4,
            vit_heads=24,
            bbox_mlp_hidden_dim=768,
            bbox_mlp_hidden_depth=2,
        )

        self.end_effector_encoder = vnn.Embedding(num_embeddings=2, embedding_dim=2)

        self.obs_fusion_layer = nn.Linear(self.obj_encoder.output_dim + 2, embed_dim)

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
        obs_mask: torch.Tensor,
        action_token: torch.Tensor | None,
        prompt_token: torch.Tensor,
        prompt_token_mask: torch.Tensor,
    ):
        L_obs, B = obs_token.shape[:2]
        L_action = 0 if action_token is None else action_token.shape[0]
        n_max_objs = obs_token.shape[-2]
        L = L_obs * n_max_objs + L_action

        tokens = torch.empty(
            L, B, self.embed_dim, dtype=torch.float32, device=obs_token.device
        )
        masks = torch.ones(L, B, dtype=torch.bool, device=obs_token.device)
        obs_token = rearrange(obs_token, "L B Q E -> B L Q E")
        obs_token = rearrange(obs_token, "B L Q E -> B (L Q) E")
        obs_token = rearrange(obs_token, "B L E -> L B E")
        obs_mask = rearrange(obs_mask, "L B Q -> B L Q")
        obs_mask = rearrange(obs_mask, "B L Q -> B (L Q)")
        obs_mask = rearrange(obs_mask, "B L -> L B")
        for q in range(n_max_objs):
            tokens[q :: n_max_objs + 1] = obs_token[q::n_max_objs]
            masks[q :: n_max_objs + 1] = obs_mask[q::n_max_objs]
        if action_token is not None:
            tokens[n_max_objs :: n_max_objs + 1] = action_token

        position_ids = torch.cumsum(masks, dim=0) - 1
        position_ids = position_ids.long()
        prompt_position_ids = torch.cumsum(prompt_token_mask, dim=1) - 1

        tokens_out = self.xattn_gpt(
            obs_action_tokens=tokens,
            prompt_tokens=prompt_token,
            prompt_mask=prompt_token_mask,
            obs_action_masks=masks.transpose(0, 1),
            obs_action_position_ids=position_ids.transpose(0, 1),
            prompt_position_ids=prompt_position_ids,
        )

        predicted_action_tokens = tokens_out[n_max_objs - 1 :: n_max_objs + 1]
        return predicted_action_tokens

    def forward_prompt_assembly(self, prompts):
        raw_prompts_token_type, word_batch, image_batch = prompts
        batch_word_emb = self.prompt_embedding(word_batch)
        batch_image_emb = self.obj_encoder(**image_batch)
        batch_image_emb = self.prompt_obj_post_layer(batch_image_emb)
        n_max_objs = batch_image_emb.shape[-2]

        L_max = 0
        for raw_prompt in raw_prompts_token_type:
            L_this = 0
            for item in raw_prompt:
                if item == 0:
                    L_this += 1
                elif item == 1:
                    L_this += n_max_objs
                else:
                    raise ValueError(f"Invalid prompt token type {item}")
            L_max = max(L_max, L_this)

        prompt_tokens, prompt_masks = [], []
        word_ptr, img_ptr = 0, 0
        for raw_prompt in raw_prompts_token_type:
            assembled_prompt = []
            assembled_mask = []
            for item in raw_prompt:
                if item == 0:
                    assembled_prompt.append(batch_word_emb[word_ptr])
                    word_ptr += 1
                    assembled_mask.append(True)
                elif item == 1:
                    obj_mask = any_concat(
                        [
                            image_batch["mask"][view][img_ptr]
                            for view in sorted(self._views)
                        ],
                        dim=-1,
                    )
                    for q in range(n_max_objs):
                        assembled_prompt.append(batch_image_emb[img_ptr][q])
                        assembled_mask.append(obj_mask[q])
                    img_ptr += 1
                else:
                    raise ValueError(f"Invalid type: {type(item)}")
            num_padding = L_max - len(assembled_prompt)
            assembled_prompt = torch.stack(assembled_prompt, dim=0)
            required_padding = torch.zeros(
                (num_padding, assembled_prompt.shape[1]),
                dtype=torch.float32,
                device=assembled_prompt.device,
            )
            assembled_prompt = torch.cat([assembled_prompt, required_padding], dim=0)
            prompt_tokens.append(assembled_prompt)

            prompt_masks.append(
                torch.cat(
                    [
                        any_to_torch_tensor(
                            assembled_mask,
                            dtype=torch.bool,
                            device=assembled_prompt.device,
                        ),
                        torch.zeros(
                            num_padding,
                            dtype=torch.bool,
                            device=assembled_prompt.device,
                        ),
                    ],
                    dim=0,
                )
            )

        prompt_tokens = torch.stack(prompt_tokens, dim=0)
        prompt_masks = torch.stack(prompt_masks, dim=0)
        prompt_tokens = prompt_tokens.transpose(0, 1)
        if self.t5_prompt_encoder is not None:
            prompt_tokens = self.t5_prompt_encoder(
                prompt_tokens, attention_mask=prompt_masks, batch_first=False
            )
            prompt_tokens = self.t5_prompt_encoder_post_layer(prompt_tokens)
        return prompt_tokens, prompt_masks

    def forward_obs_token(self, obs):
        objects, ee = obs["objects"], obs["ee"]
        leading_dims = ee.shape[:2]

        objects = objects.map_structure(func=lambda x: x.reshape(-1, *x.shape[2:]))
        img_feats = self.obj_encoder(**objects)
        img_feats = img_feats.reshape(*leading_dims, *img_feats.shape[1:])
        obj_mask = {
            k: objects["mask"][k].reshape(*leading_dims, -1) for k in objects["mask"]
        }

        ee_feats = self.end_effector_encoder(ee)
        ee_feats = ee_feats.unsqueeze(2).repeat(1, 1, img_feats.shape[-2], 1)

        obs_feats = self.obs_fusion_layer(torch.cat([img_feats, ee_feats], dim=-1))

        obj_mask = any_concat([obj_mask[view] for view in sorted(self._views)], dim=-1)
        return obs_feats, obj_mask

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
