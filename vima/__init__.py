import os
import torch

from .policy import *


def create_policy_from_ckpt(ckpt_path, device):
    assert os.path.exists(ckpt_path), "Checkpoint path does not exist"
    ckpt = torch.load(ckpt_path, map_location=device)
    policy_instance = VIMAPolicy(**ckpt["cfg"])
    policy_instance.load_state_dict(
        {k.replace("policy.", ""): v for k, v in ckpt["state_dict"].items()},
        strict=True,
    )
    policy_instance.eval()
    return policy_instance
