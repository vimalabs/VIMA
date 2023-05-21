from __future__ import annotations

import warnings

import torch
import kornia


@torch.no_grad()
def basic_image_tensor_preprocess(
    img,
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
    std: tuple[float, float, float] = (0.5, 0.5, 0.5),
    shape: tuple[int, int] | None = None,
):
    """
    Check for resize, and divide by 255
    """

    assert torch.is_tensor(img)
    assert img.dim() >= 4
    original_shape = list(img.size())
    img = img.float()
    img = img.flatten(0, img.dim() - 4)
    assert img.dim() == 4

    input_size = img.size()[-2:]
    assert img.max() > 2, "img should be between [0, 255] before normalize"

    if shape and input_size != shape:
        warnings.warn(
            f'{"Down" if shape < input_size else "Up"}sampling image'
            f" from original resolution {input_size}x{input_size}"
            f" to {shape}x{shape}"
        )
        img = kornia.geometry.transform.resize(img, shape).clamp(0.0, 255.0)

    B, C, H, W = img.size()
    assert C % 3 == 0, "channel must divide 3"
    img = img.view(B * C // 3, 3, H, W)
    img = torch_normalize(img / 255.0, mean=mean, std=std)
    original_shape[-2:] = H, W
    return img.view(original_shape)


def torch_normalize(tensor: torch.Tensor, mean, std, inplace=False):
    """
    Adapted from https://pytorch.org/docs/stable/_modules/torchvision/transforms/functional.html#normalize

    Normalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not torch.is_tensor(tensor):
        raise TypeError("tensor should be a torch tensor. Got {}.".format(type(tensor)))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError(
            f"std evaluated to zero after conversion to {dtype}, leading to division by zero."
        )
    if mean.ndim == 1:
        mean = mean[:, None, None]
    if std.ndim == 1:
        std = std[:, None, None]
    tensor.sub_(mean).div_(std)
    return tensor
