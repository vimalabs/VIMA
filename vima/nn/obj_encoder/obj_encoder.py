from __future__ import annotations

import torch
import torch.nn as nn

from .vit import ViTEncoder, GatoViTEncoder, ViTEncoderRectangular
from .perceiver import ObjectsPerceiverEncoder
from ..utils import build_mlp


class ObjEncoder(nn.Module):
    bbox_max_h = 128
    bbox_max_w = 256

    def __init__(
        self,
        *,
        transformer_emb_dim: int,
        views: list[str],
        vit_output_dim: int = 512,
        vit_resolution: int,
        vit_patch_size: int,
        vit_width: int,
        vit_layers: int,
        vit_heads: int,
        bbox_mlp_hidden_dim: int,
        bbox_mlp_hidden_depth: int,
    ):
        super().__init__()

        views = sorted(views)
        self._views = views
        self._transformer_emb_dim = transformer_emb_dim

        self.cropped_img_encoder = ViTEncoder(
            output_dim=vit_output_dim,
            resolution=vit_resolution,
            patch_size=vit_patch_size,
            width=vit_width,
            layers=vit_layers,
            heads=vit_heads,
        )

        self.bbox_mlp = nn.ModuleDict(
            {
                view: build_mlp(
                    4,
                    hidden_dim=bbox_mlp_hidden_dim,
                    hidden_depth=bbox_mlp_hidden_depth,
                    output_dim=bbox_mlp_hidden_dim,
                )
                for view in views
            }
        )

        self.pre_transformer_layer = nn.ModuleDict(
            {
                view: nn.Linear(
                    self.cropped_img_encoder.output_dim + bbox_mlp_hidden_dim,
                    transformer_emb_dim,
                )
                for view in views
            }
        )

    def forward(
        self,
        cropped_img,
        bbox,
        mask,
    ):
        """
        out: (..., n_objs * n_views, E)
        """
        img_feats = {
            view: self.cropped_img_encoder(cropped_img[view]) for view in self._views
        }
        # normalize bbox
        bbox = {view: bbox[view].float() for view in self._views}
        _normalizer = torch.tensor(
            [self.bbox_max_w, self.bbox_max_h, self.bbox_max_h, self.bbox_max_w],
            dtype=bbox[self._views[0]].dtype,
            device=bbox[self._views[0]].device,
        )
        bbox = {view: bbox[view] / _normalizer for view in self._views}
        bbox = {view: self.bbox_mlp[view](bbox[view]) for view in self._views}

        in_feats = {
            view: self.pre_transformer_layer[view](
                torch.concat([img_feats[view], bbox[view]], dim=-1)
            )
            for view in self._views
        }
        out = torch.concat([in_feats[view] for view in self._views], dim=-2)
        return out

    @property
    def output_dim(self):
        return self._transformer_emb_dim


class GatoMultiViewRGBEncoder(nn.Module):
    def __init__(
        self,
        *,
        emb_dim: int,
        views: list[str],
        img_size: tuple[int, int],
        vit_patch_size: int | None = None,
        vit_width: int | None = None,
        vit_layers: int | None = None,
        vit_heads: int | None = None,
    ):
        super().__init__()

        views = sorted(views)
        self._views = views
        self.output_dim = emb_dim

        self.cropped_img_encoder = GatoViTEncoder(
            img_size=img_size,
            patch_size=vit_patch_size,
            width=vit_width,
            layers=vit_layers,
            heads=vit_heads,
            output_dim=emb_dim,
        )

    def forward(
        self,
        rgb,
    ):
        """
        input: (..., 3, H, W)
        output: (..., L * n_views, E)
        """
        img_feats = {
            view: self.cropped_img_encoder(rgb[view]) for view in self._views
        }  # dict of (..., L, E)
        out = torch.concat(
            [img_feats[view] for view in self._views], dim=-2
        )  # (..., L * n_views, E)
        return out

    @property
    def img_patch_len(self):
        return self.cropped_img_encoder.vit.img_patch_len * len(self._views)


class MultiViewRGBPerceiverEncoder(nn.Module):
    def __init__(
        self,
        *,
        emb_dim: int,
        views: list[str],
        img_size: tuple[int, int],
        vit_patch_size: int | None = None,
        vit_width: int | None = None,
        vit_layers: int | None = None,
        vit_heads: int | None = None,
        perceiver_num_queries: int,
        perceiver_num_blocks: int,
        perceiver_num_self_attends_per_block: int,
        perceiver_num_self_attention_heads: int,
        perceiver_num_cross_attention_heads: int,
        perceiver_attention_probs_dropout_prob: float,
    ):
        super().__init__()

        views = sorted(views)
        self._views = views
        self._transformer_emb_dim = emb_dim

        self.cropped_img_encoder = GatoViTEncoder(
            img_size=img_size,
            output_dim=emb_dim,
            patch_size=vit_patch_size,
            width=vit_width,
            layers=vit_layers,
            heads=vit_heads,
        )
        self.peceiver = ObjectsPerceiverEncoder(
            emb_dim,
            num_latents=perceiver_num_queries,
            num_blocks=perceiver_num_blocks,
            num_self_attends_per_block=perceiver_num_self_attends_per_block,
            num_self_attention_heads=perceiver_num_self_attention_heads,
            num_cross_attention_heads=perceiver_num_cross_attention_heads,
            attention_probs_dropout_prob=perceiver_attention_probs_dropout_prob,
        )

    def forward(
        self,
        rgb,
    ):
        img_feats = {view: self.cropped_img_encoder(rgb[view]) for view in self._views}
        img_feats = torch.concat([img_feats[view] for view in self._views], dim=-2)
        masks = torch.ones(
            img_feats.shape[:2], device=img_feats.device, dtype=torch.bool
        )
        out = self.peceiver(img_feats, masks)
        return out

    @property
    def output_dim(self):
        return self._transformer_emb_dim


class MultiViewRGBEncoder(nn.Module):
    def __init__(
        self,
        *,
        emb_dim: int,
        views: list[str],
        img_size: tuple[int, int],
        vit_patch_size: int | None = None,
        vit_width: int | None = None,
        vit_layers: int | None = None,
        vit_heads: int | None = None,
    ):
        super().__init__()

        views = sorted(views)
        self._views = views
        self._transformer_emb_dim = emb_dim

        self.cropped_img_encoder = ViTEncoderRectangular(
            img_size=img_size,
            output_dim=emb_dim,
            patch_size=vit_patch_size,
            width=vit_width,
            layers=vit_layers,
            heads=vit_heads,
        )

    def forward(
        self,
        rgb,
    ):
        img_feats = {view: self.cropped_img_encoder(rgb[view]) for view in self._views}
        out = torch.concat([img_feats[view] for view in self._views], dim=-1)
        return out

    @property
    def output_dim(self):
        return self._transformer_emb_dim * len(self._views)
