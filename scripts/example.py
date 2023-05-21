from __future__ import annotations

import os

import numpy as np
from tokenizers import Tokenizer
from tokenizers import AddedToken
from einops import rearrange
import cv2
from vima.utils import *
from vima import create_policy_from_ckpt
from vima_bench import *
from gym.wrappers import TimeLimit as _TimeLimit
from gym import Wrapper
import torch
import argparse


os.environ["TOKENIZERS_PARALLELISM"] = "true"


_kwargs = {
    "single_word": True,
    "lstrip": False,
    "rstrip": False,
    "normalized": True,
}

PLACEHOLDER_TOKENS = [
    AddedToken("{base_obj}", **_kwargs),
    AddedToken("{base_obj_1}", **_kwargs),
    AddedToken("{base_obj_2}", **_kwargs),
    AddedToken("{dragged_obj}", **_kwargs),
    AddedToken("{dragged_obj_1}", **_kwargs),
    AddedToken("{dragged_obj_2}", **_kwargs),
    AddedToken("{dragged_obj_3}", **_kwargs),
    AddedToken("{dragged_obj_4}", **_kwargs),
    AddedToken("{dragged_obj_5}", **_kwargs),
    AddedToken("{swept_obj}", **_kwargs),
    AddedToken("{bounds}", **_kwargs),
    AddedToken("{constraint}", **_kwargs),
    AddedToken("{scene}", **_kwargs),
    AddedToken("{demo_blicker_obj_1}", **_kwargs),
    AddedToken("{demo_less_blicker_obj_1}", **_kwargs),
    AddedToken("{demo_blicker_obj_2}", **_kwargs),
    AddedToken("{demo_less_blicker_obj_2}", **_kwargs),
    AddedToken("{demo_blicker_obj_3}", **_kwargs),
    AddedToken("{demo_less_blicker_obj_3}", **_kwargs),
    AddedToken("{start_scene}", **_kwargs),
    AddedToken("{end_scene}", **_kwargs),
    AddedToken("{before_twist_1}", **_kwargs),
    AddedToken("{after_twist_1}", **_kwargs),
    AddedToken("{before_twist_2}", **_kwargs),
    AddedToken("{after_twist_2}", **_kwargs),
    AddedToken("{before_twist_3}", **_kwargs),
    AddedToken("{after_twist_3}", **_kwargs),
    AddedToken("{frame_0}", **_kwargs),
    AddedToken("{frame_1}", **_kwargs),
    AddedToken("{frame_2}", **_kwargs),
    AddedToken("{frame_3}", **_kwargs),
    AddedToken("{frame_4}", **_kwargs),
    AddedToken("{frame_5}", **_kwargs),
    AddedToken("{frame_6}", **_kwargs),
    AddedToken("{ring}", **_kwargs),
    AddedToken("{hanoi_stand}", **_kwargs),
    AddedToken("{start_scene_1}", **_kwargs),
    AddedToken("{end_scene_1}", **_kwargs),
    AddedToken("{start_scene_2}", **_kwargs),
    AddedToken("{end_scene_2}", **_kwargs),
    AddedToken("{start_scene_3}", **_kwargs),
    AddedToken("{end_scene_3}", **_kwargs),
]
PLACEHOLDERS = [token.content for token in PLACEHOLDER_TOKENS]
tokenizer = Tokenizer.from_pretrained("t5-base")
tokenizer.add_tokens(PLACEHOLDER_TOKENS)


@torch.no_grad()
def main(cfg):
    assert cfg.partition in ALL_PARTITIONS
    assert cfg.task in PARTITION_TO_SPECS["test"][cfg.partition]

    seed = 42
    policy = create_policy_from_ckpt(cfg.ckpt, cfg.device)
    env = TimeLimitWrapper(
        ResetFaultToleranceWrapper(
            make(
                cfg.task,
                modalities=["segm", "rgb"],
                task_kwargs=PARTITION_TO_SPECS["test"][cfg.partition][cfg.task],
                seed=seed,
                render_prompt=True,
                display_debug_window=True,
                hide_arm_rgb=False,
            )
        ),
        bonus_steps=2,
    )

    while True:
        env.global_seed = seed

        obs = env.reset()
        env.render()

        meta_info = env.meta_info
        prompt = env.prompt
        prompt_assets = env.prompt_assets
        elapsed_steps = 0
        inference_cache = {}
        while True:
            if elapsed_steps == 0:
                prompt_token_type, word_batch, image_batch = prepare_prompt(
                    prompt=prompt, prompt_assets=prompt_assets, views=["front", "top"]
                )
                word_batch = word_batch.to(cfg.device)
                image_batch = image_batch.to_torch_tensor(device=cfg.device)
                prompt_tokens, prompt_masks = policy.forward_prompt_assembly(
                    (prompt_token_type, word_batch, image_batch)
                )

                inference_cache["obs_tokens"] = []
                inference_cache["obs_masks"] = []
                inference_cache["action_tokens"] = []
            obs["ee"] = np.asarray(obs["ee"])
            obs = add_batch_dim(obs)
            obs = prepare_obs(obs=obs, rgb_dict=None, meta=meta_info).to_torch_tensor(
                device=cfg.device
            )
            obs_token_this_step, obs_mask_this_step = policy.forward_obs_token(obs)
            obs_token_this_step = obs_token_this_step.squeeze(0)
            obs_mask_this_step = obs_mask_this_step.squeeze(0)
            inference_cache["obs_tokens"].append(obs_token_this_step[0])
            inference_cache["obs_masks"].append(obs_mask_this_step[0])
            max_objs = max(x.shape[0] for x in inference_cache["obs_tokens"])
            obs_tokens_to_forward, obs_masks_to_forward = [], []
            obs_tokens_this_env, obs_masks_this_env = [], []
            for idx in range(len(inference_cache["obs_tokens"])):
                obs_this_env_this_step = inference_cache["obs_tokens"][idx]
                obs_mask_this_env_this_step = inference_cache["obs_masks"][idx]
                required_pad = max_objs - obs_this_env_this_step.shape[0]
                obs_tokens_this_env.append(
                    any_concat(
                        [
                            obs_this_env_this_step,
                            torch.zeros(
                                required_pad,
                                obs_this_env_this_step.shape[1],
                                device=cfg.device,
                                dtype=obs_this_env_this_step.dtype,
                            ),
                        ],
                        dim=0,
                    )
                )
                obs_masks_this_env.append(
                    any_concat(
                        [
                            obs_mask_this_env_this_step,
                            torch.zeros(
                                required_pad,
                                device=cfg.device,
                                dtype=obs_mask_this_env_this_step.dtype,
                            ),
                        ],
                        dim=0,
                    )
                )
            obs_tokens_to_forward.append(any_stack(obs_tokens_this_env, dim=0))
            obs_masks_to_forward.append(any_stack(obs_masks_this_env, dim=0))
            obs_tokens_to_forward = any_stack(obs_tokens_to_forward, dim=0)
            obs_masks_to_forward = any_stack(obs_masks_to_forward, dim=0)
            obs_tokens_to_forward = obs_tokens_to_forward.transpose(0, 1)
            obs_masks_to_forward = obs_masks_to_forward.transpose(0, 1)

            if elapsed_steps == 0:
                action_tokens_to_forward = None
            else:
                action_tokens_to_forward = any_stack(
                    [any_stack(inference_cache["action_tokens"], dim=0)],
                    dim=0,
                )
                action_tokens_to_forward = action_tokens_to_forward.transpose(0, 1)
            predicted_action_tokens = policy.forward(
                obs_token=obs_tokens_to_forward,
                action_token=action_tokens_to_forward,
                prompt_token=prompt_tokens,
                prompt_token_mask=prompt_masks,
                obs_mask=obs_masks_to_forward,
            )  # (L, B, E)
            predicted_action_tokens = predicted_action_tokens[-1].unsqueeze(
                0
            )  # (1, B, E)
            dist_dict = policy.forward_action_decoder(predicted_action_tokens)
            actions = {k: v.mode() for k, v in dist_dict.items()}
            action_tokens = policy.forward_action_token(actions)  # (1, B, E)
            action_tokens = action_tokens.squeeze(0)  # (B, E)
            inference_cache["action_tokens"].append(action_tokens[0])
            actions = policy._de_discretize_actions(actions)
            action_bounds = [meta_info["action_bounds"]]
            action_bounds_low = [action_bound["low"] for action_bound in action_bounds]
            action_bounds_high = [
                action_bound["high"] for action_bound in action_bounds
            ]
            action_bounds_low = np.asarray(action_bounds_low)
            action_bounds_high = np.asarray(action_bounds_high)
            action_bounds_low = torch.tensor(
                action_bounds_low, dtype=torch.float32, device=cfg.device
            )
            action_bounds_high = torch.tensor(
                action_bounds_high, dtype=torch.float32, device=cfg.device
            )
            actions["pose0_position"] = (
                actions["pose0_position"] * (action_bounds_high - action_bounds_low)
                + action_bounds_low
            )
            actions["pose1_position"] = (
                actions["pose1_position"] * (action_bounds_high - action_bounds_low)
                + action_bounds_low
            )
            actions["pose0_position"] = torch.clamp(
                actions["pose0_position"], min=action_bounds_low, max=action_bounds_high
            )
            actions["pose1_position"] = torch.clamp(
                actions["pose1_position"], min=action_bounds_low, max=action_bounds_high
            )
            actions["pose0_rotation"] = actions["pose0_rotation"] * 2 - 1
            actions["pose1_rotation"] = actions["pose1_rotation"] * 2 - 1
            actions["pose0_rotation"] = torch.clamp(
                actions["pose0_rotation"], min=-1, max=1
            )
            actions["pose1_rotation"] = torch.clamp(
                actions["pose1_rotation"], min=-1, max=1
            )
            actions = {k: v.cpu().numpy() for k, v in actions.items()}
            actions = any_slice(actions, np.s_[0, 0])
            obs, _, done, info = env.step(actions)
            elapsed_steps += 1
            if done:
                break


def prepare_prompt(*, prompt: str, prompt_assets: dict, views: list[str]):
    views = sorted(views)
    encoding = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_ids, prompt_tokens = encoding.ids, encoding.tokens
    assert set(prompt_assets.keys()) == set(
        [token[1:-1] for token in prompt_tokens if token in PLACEHOLDERS]
    )
    filled_prompt = []
    for id, token in zip(prompt_ids, prompt_tokens):
        if token not in PLACEHOLDERS:
            assert "{" not in token and "}" not in token
            filled_prompt.append(id)
        else:
            assert token.startswith("{") and token.endswith("}")
            asset_name = token[1:-1]
            assert asset_name in prompt_assets, f"missing prompt asset {asset_name}"
            asset = prompt_assets[asset_name]
            obj_info = asset["segm"]["obj_info"]
            placeholder_type = asset["placeholder_type"]
            if placeholder_type == "object":
                objects = [obj_info["obj_id"]]
            elif placeholder_type == "scene":
                objects = [each_info["obj_id"] for each_info in obj_info]
            obj_repr = {
                "cropped_img": {view: [] for view in views},
                "bbox": {view: [] for view in views},
            }
            for view in views:
                rgb_this_view = asset["rgb"][view]
                segm_this_view = asset["segm"][view]
                bboxes = []
                cropped_imgs = []
                for obj_id in objects:
                    ys, xs = np.nonzero(segm_this_view == obj_id)
                    if len(xs) < 2 or len(ys) < 2:
                        continue
                    xmin, xmax = np.min(xs), np.max(xs)
                    ymin, ymax = np.min(ys), np.max(ys)
                    x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
                    h, w = ymax - ymin, xmax - xmin
                    bboxes.append([int(x_center), int(y_center), int(h), int(w)])
                    cropped_img = rgb_this_view[:, ymin : ymax + 1, xmin : xmax + 1]
                    if cropped_img.shape[1] != cropped_img.shape[2]:
                        diff = abs(cropped_img.shape[1] - cropped_img.shape[2])
                        pad_before, pad_after = int(diff / 2), diff - int(diff / 2)
                        if cropped_img.shape[1] > cropped_img.shape[2]:
                            pad_width = ((0, 0), (0, 0), (pad_before, pad_after))
                        else:
                            pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
                        cropped_img = np.pad(
                            cropped_img,
                            pad_width,
                            mode="constant",
                            constant_values=0,
                        )
                        assert cropped_img.shape[1] == cropped_img.shape[2], "INTERNAL"
                    cropped_img = rearrange(cropped_img, "c h w -> h w c")
                    cropped_img = np.asarray(cropped_img)
                    cropped_img = cv2.resize(
                        cropped_img,
                        (32, 32),
                        interpolation=cv2.INTER_AREA,
                    )
                    cropped_img = rearrange(cropped_img, "h w c -> c h w")
                    cropped_imgs.append(cropped_img)
                bboxes = np.asarray(bboxes)
                cropped_imgs = np.asarray(cropped_imgs)
                obj_repr["bbox"][view] = bboxes
                obj_repr["cropped_img"][view] = cropped_imgs
            filled_prompt.append(obj_repr)
    raw_prompt = [filled_prompt]
    max_n_objs_prompt = {view: 0 for view in views}
    for prompt in raw_prompt:
        for token in prompt:
            if isinstance(token, dict):
                for view in views:
                    max_n_objs_prompt[view] = max(
                        max_n_objs_prompt[view], len(token["cropped_img"][view])
                    )
    raw_prompt_token_type, word_batch, image_batch = [], [], []
    for prompt in raw_prompt:
        token_type = []
        for token in prompt:
            if isinstance(token, int):
                token_type.append(0)
                word_batch.append(token)
            elif isinstance(token, dict):
                token_type.append(1)
                n_objs_prompt = {
                    view: len(token["cropped_img"][view]) for view in views
                }
                # add mask
                token["mask"] = {
                    view: np.ones((n_objs_prompt[view],), dtype=np.bool)
                    for view in views
                }
                n_objs_to_pad = {
                    view: max_n_objs_prompt[view] - n_objs_prompt[view]
                    for view in views
                }
                objs_pad = {
                    "bbox": {
                        view: np.zeros((n_objs_to_pad[view], 4), dtype=np.int64)
                        for view in views
                    },
                    "cropped_img": {
                        view: np.zeros(
                            (n_objs_to_pad[view], 3, 32, 32),
                            dtype=np.uint8,
                        )
                        for view in views
                    },
                    "mask": {
                        view: np.zeros((n_objs_to_pad[view]), dtype=np.bool)
                        for view in views
                    },
                }
                token = any_concat([token, objs_pad], dim=0)
                image_batch.append(token)
        raw_prompt_token_type.append(token_type)
    assert sum([len(prompt) for prompt in raw_prompt_token_type]) == len(
        word_batch
    ) + len(image_batch)
    word_batch = any_stack(word_batch, dim=0)
    image_batch = any_to_datadict(stack_sequence_fields(image_batch))

    word_batch = any_to_torch_tensor(word_batch)
    image_batch = image_batch.to_torch_tensor()
    return raw_prompt_token_type, word_batch, image_batch


def prepare_obs(
    *,
    obs: dict,
    rgb_dict: dict | None = None,
    meta: dict,
):
    assert not (rgb_dict is not None and "rgb" in obs)
    rgb_dict = rgb_dict or obs.pop("rgb")
    segm_dict = obs.pop("segm")
    views = sorted(rgb_dict.keys())
    assert meta["n_objects"] == len(meta["obj_id_to_info"])
    objects = list(meta["obj_id_to_info"].keys())

    L_obs = get_batch_size(obs)

    obs_list = {
        "ee": obs["ee"],
        "objects": {
            "cropped_img": {view: [] for view in views},
            "bbox": {view: [] for view in views},
            "mask": {view: [] for view in views},
        },
    }

    for l in range(L_obs):
        rgb_dict_this_step = any_slice(rgb_dict, np.s_[l])
        segm_dict_this_step = any_slice(segm_dict, np.s_[l])
        for view in views:
            rgb_this_view = rgb_dict_this_step[view]
            segm_this_view = segm_dict_this_step[view]
            bboxes = []
            cropped_imgs = []
            n_pad = 0
            for obj_id in objects:
                ys, xs = np.nonzero(segm_this_view == obj_id)
                if len(xs) < 2 or len(ys) < 2:
                    n_pad += 1
                    continue
                xmin, xmax = np.min(xs), np.max(xs)
                ymin, ymax = np.min(ys), np.max(ys)
                x_center, y_center = (xmin + xmax) / 2, (ymin + ymax) / 2
                h, w = ymax - ymin, xmax - xmin
                bboxes.append([int(x_center), int(y_center), int(h), int(w)])
                cropped_img = rgb_this_view[:, ymin : ymax + 1, xmin : xmax + 1]
                if cropped_img.shape[1] != cropped_img.shape[2]:
                    diff = abs(cropped_img.shape[1] - cropped_img.shape[2])
                    pad_before, pad_after = int(diff / 2), diff - int(diff / 2)
                    if cropped_img.shape[1] > cropped_img.shape[2]:
                        pad_width = ((0, 0), (0, 0), (pad_before, pad_after))
                    else:
                        pad_width = ((0, 0), (pad_before, pad_after), (0, 0))
                    cropped_img = np.pad(
                        cropped_img, pad_width, mode="constant", constant_values=0
                    )
                    assert cropped_img.shape[1] == cropped_img.shape[2], "INTERNAL"
                cropped_img = rearrange(cropped_img, "c h w -> h w c")
                cropped_img = np.asarray(cropped_img)
                cropped_img = cv2.resize(
                    cropped_img,
                    (32, 32),
                    interpolation=cv2.INTER_AREA,
                )
                cropped_img = rearrange(cropped_img, "h w c -> c h w")
                cropped_imgs.append(cropped_img)
            bboxes = np.asarray(bboxes)
            cropped_imgs = np.asarray(cropped_imgs)
            mask = np.ones(len(bboxes), dtype=bool)
            if n_pad > 0:
                bboxes = np.concatenate(
                    [bboxes, np.zeros((n_pad, 4), dtype=bboxes.dtype)], axis=0
                )
                cropped_imgs = np.concatenate(
                    [
                        cropped_imgs,
                        np.zeros(
                            (n_pad, 3, 32, 32),
                            dtype=cropped_imgs.dtype,
                        ),
                    ],
                    axis=0,
                )
                mask = np.concatenate([mask, np.zeros(n_pad, dtype=bool)], axis=0)
            obs_list["objects"]["bbox"][view].append(bboxes)
            obs_list["objects"]["cropped_img"][view].append(cropped_imgs)
            obs_list["objects"]["mask"][view].append(mask)
    for view in views:
        obs_list["objects"]["bbox"][view] = np.stack(
            obs_list["objects"]["bbox"][view], axis=0
        )
        obs_list["objects"]["cropped_img"][view] = np.stack(
            obs_list["objects"]["cropped_img"][view], axis=0
        )
        obs_list["objects"]["mask"][view] = np.stack(
            obs_list["objects"]["mask"][view], axis=0
        )

    obs = any_to_datadict(any_stack([obs_list], dim=0))
    obs = obs.to_torch_tensor()
    obs = any_transpose_first_two_axes(obs)
    return obs


class ResetFaultToleranceWrapper(Wrapper):
    max_retries = 10

    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        for _ in range(self.max_retries):
            try:
                return self.env.reset()
            except:
                current_seed = self.env.unwrapped.task.seed
                self.env.global_seed = current_seed + 1
        raise RuntimeError(
            "Failed to reset environment after {} retries".format(self.max_retries)
        )


class TimeLimitWrapper(_TimeLimit):
    def __init__(self, env, bonus_steps: int = 0):
        super().__init__(env, env.task.oracle_max_steps + bonus_steps)


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--partition", type=str, default="placement_generalization")
    arg.add_argument("--task", type=str, default="visual_manipulation")
    arg.add_argument("--ckpt", type=str, required=True)
    arg.add_argument("--device", default="cpu")
    arg = arg.parse_args()
    main(arg)
