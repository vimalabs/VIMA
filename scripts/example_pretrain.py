from __future__ import annotations

import os
import pdb 
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
import mediapy as media 
from vima_bench.env.wrappers.recorder import GUIRecorder
os.environ["TOKENIZERS_PARALLELISM"] = "true"


_kwargs = {
    "single_word": True,
    "lstrip": False,
    "rstrip": False,
    "normalized": True,
}

## TODO: 
"""
You simply want to modify the program such that we store a certain number of vima episodes and also record the reward produced by the model 

"""

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
from collections import defaultdict
class VimaRecorder(object): 
    def __init__(self,orientation='front',space='rgb',num_episodes=10) -> None:
        self.orientation = 'front' 
        self.space = space 
        self.orientation = orientation 
        self.num_episodes=num_episodes 
        self._current_ep = 0  
        self.episodes=  defaultdict(list)
        self.isOpen = True 
    def add_step(self,obs,done): 
        current_frame = self.get_frame(obs) 
        if  self.isOpen: 
            self.episodes[self._current_ep].append(current_frame) 
        if self.isOpen and done: 
            self._current_ep +=1 
        if self._current_ep > self.num_episodes: 
            self.isOpen = False  
        
    def get_frame(self,obs): 
        arr = obs[self.space][self.orientation] 
        #resturcture it so its in the same form as expected by matplotlib 
        re_orient = np.moveaxis(arr,[0,1,2],[2,0,1])
        return re_orient


def main(cfg): 
    from glob import glob 
    import pickle as pkl 
    from PIL import Image 
    assert cfg.partition in ALL_PARTITIONS
    assert cfg.task in PARTITION_TO_SPECS["test"][cfg.partition]
    seed = 42
    #i modified the policy loading code to just load a blank policy network with no weights using a flag 
    policy = create_policy_from_ckpt(cfg.ckpt, cfg.device,ignore_statedict=True)
    policy = policy.to('cuda')
    trajectories = glob("/home/rlcorrea/vima_v6/rearrange_then_restore/*")[0:10] 
    for traj in trajectories:
        with open(os.path.join(traj, "obs.pkl"), "rb") as f:
            obs = pkl.load(f)
        rgb_dict = {"front": [], "top": []}
        n_rgb_frames = len(os.listdir(os.path.join(traj, f"rgb_front")))
        for view in ["front", "top"]:
            for idx in range(n_rgb_frames):
                # load {idx}.jpg using PIL
                rgb_dict[view].append(
                    rearrange(
                        np.array(
                            Image.open(os.path.join(traj, f"rgb_{view}", f"{idx}.jpg")),
                            copy=True,
                            dtype=np.uint8,
                        ),
                        "h w c -> c h w",
                    )
                )
        rgb_dict = {k: np.stack(v, axis=0) for k, v in rgb_dict.items()}
        segm = obs.pop("segm")
        end_effector = obs.pop("ee")
        with open(os.path.join(traj, "action.pkl"), "rb") as f:
            action = pkl.load(f)

        with open(os.path.join(traj, "trajectory.pkl"), "rb") as f:
            traj_meta = pkl.load(f)

        prompt = traj_meta.pop("prompt")
        prompt_assets = traj_meta.pop("prompt_assets")
        pdb.set_trace() 
        print('hi')
        #https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/common/base_class.py#L753

if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--partition", type=str, default="placement_generalization")
    arg.add_argument("--task", type=str, default="visual_manipulation")
    arg.add_argument("--ckpt", type=str, required=True)
    arg.add_argument("--device", default="cuda:0")
    arg = arg.parse_args()
    main(arg)
