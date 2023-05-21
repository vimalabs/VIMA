from collections import OrderedDict

import torch
import torch.nn as nn

from .preprocess import basic_image_tensor_preprocess


VIMA_IMG_MEAN = (0.3471, 0.3429, 0.3383)
VIMA_IMG_STD = (0.3011, 0.2961, 0.2956)


class ViTEncoder(nn.Module):
    def __init__(
        self,
        *,
        output_dim: int,
        resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.vit = VisionTransformer(
            resolution=resolution,
            patch_size=patch_size,
            width=width,
            layers=layers,
            heads=heads,
            output_dim=output_dim,
        )

    def forward(self, x):
        """
        x: (..., 3, H, W)
        """
        assert x.dim() >= 4
        leading_dim = x.shape[:-3]
        x = basic_image_tensor_preprocess(x, mean=VIMA_IMG_MEAN, std=VIMA_IMG_STD)
        x = x.flatten(0, x.dim() - 4)
        x = self.vit(x)
        x = x.view(*leading_dim, self.output_dim)
        return x


class GatoViTEncoder(nn.Module):
    def __init__(
        self,
        *,
        img_size: tuple[int, int],
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.vit = GatoVisionTransformerRectangular(
            img_size=img_size,
            patch_size=patch_size,
            width=width,
            layers=layers,
            heads=heads,
            output_dim=output_dim,
        )

    def forward(self, x):
        """
        x: (..., 3, H, W)
        """
        assert x.dim() >= 4
        leading_dim = x.shape[:-3]
        x = basic_image_tensor_preprocess(x, mean=VIMA_IMG_MEAN, std=VIMA_IMG_STD)
        x = x.flatten(0, x.dim() - 4)
        x = self.vit(x)  # (B, L, E)
        x = x.view(*leading_dim, *x.shape[-2:])  # (..., L, E)
        return x


class GatoVisionTransformerRectangular(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        n_patches_height = img_size[0] // patch_size
        n_patches_width = img_size[1] // patch_size
        self.pos_embed = nn.Parameter(
            scale * torch.randn(n_patches_height * n_patches_width, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.blocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )
        self.ln_post = nn.LayerNorm(width)
        self.projection = nn.Parameter(scale * torch.randn(width, output_dim))

        self.img_patch_len = n_patches_height * n_patches_width

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, H_patch, W_patch]
        B = x.size(0)
        x = x.reshape(B, x.shape[1], -1)  # shape = [*, width, H_patch * W_patch]
        x = x.permute(0, 2, 1)  # shape = [*, H_patch * W_patch, width]
        x = x + self.pos_embed
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.blocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        x = x @ self.projection
        return x


class VisionTransformer(nn.Module):
    def __init__(
        self,
        resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self._resolution = resolution
        self._patch_size = patch_size
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.cls_token = nn.Parameter(scale * torch.randn(width))
        self.pos_embed = nn.Parameter(
            scale * torch.randn((resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.blocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )
        self.ln_post = nn.LayerNorm(width)
        self.projection = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B = x.size(0)
        x = x.reshape(B, x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.cls_token.repeat((B, 1, 1)), x], dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.pos_embed
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.blocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.projection is not None:
            x = x @ self.projection

        return x


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = nn.LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        original_dtype = x.dtype
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        out = self.attn(
            x.to(torch.float32),
            x.to(torch.float32),
            x,
            need_weights=False,
            attn_mask=self.attn_mask,
        )[0]
        return out.to(original_dtype)

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ViTEncoderRectangular(nn.Module):
    def __init__(
        self,
        *,
        output_dim: int,
        img_size: tuple[int, int],
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.vit = VisionTransformerRectangular(
            img_size=img_size,
            patch_size=patch_size,
            width=width,
            layers=layers,
            heads=heads,
            output_dim=output_dim,
        )

    def forward(self, x):
        """
        x: (..., 3, H, W)
        """
        assert x.dim() >= 4
        leading_dim = x.shape[:-3]
        x = basic_image_tensor_preprocess(x, mean=VIMA_IMG_MEAN, std=VIMA_IMG_STD)
        x = x.flatten(0, x.dim() - 4)
        x = self.vit(x)
        x = x.view(*leading_dim, self.output_dim)
        return x


class VisionTransformerRectangular(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        scale = width**-0.5
        self.cls_token = nn.Parameter(scale * torch.randn(width))
        n_patches_height = img_size[0] // patch_size
        n_patches_width = img_size[1] // patch_size
        self.pos_embed = nn.Parameter(
            scale * torch.randn(n_patches_height * n_patches_width + 1, width)
        )
        self.ln_pre = nn.LayerNorm(width)
        self.blocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads) for _ in range(layers)]
        )
        self.ln_post = nn.LayerNorm(width)
        self.projection = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        B = x.size(0)
        x = x.reshape(B, x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.cls_token.repeat((B, 1, 1)), x], dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.pos_embed
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.blocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.projection is not None:
            x = x @ self.projection

        return x
