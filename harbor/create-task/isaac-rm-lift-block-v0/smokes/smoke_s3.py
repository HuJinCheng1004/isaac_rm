# S3 — reset smoke: 5 resets, obs dim 27, joint pos varies across resets
import sys
sys.path.insert(0, "source/chassis_nav")
from isaaclab.app import AppLauncher

app = AppLauncher({"headless": True, "enable_cameras": False}).app

import torch
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import chassis_nav.tasks  # noqa: F401

cfg = parse_env_cfg("Isaac-RM-Lift-Block-v0", device="cuda:0", num_envs=1)
env = gym.make("Isaac-RM-Lift-Block-v0", cfg=cfg, render_mode=None)

joint_snaps = []
for i in range(5):
    obs, _ = env.reset()
    obs_vec = obs["policy"] if isinstance(obs, dict) else obs
    assert obs_vec.shape[-1] == 27, f"obs dim {obs_vec.shape[-1]} != 27"
    # joint_pos is the first 6 entries (arm joint_pos_rel)
    joint_snaps.append(obs_vec[..., :6].clone())

stacked = torch.stack(joint_snaps, dim=0)
spread = (stacked.max(dim=0).values - stacked.min(dim=0).values).max().item()
assert spread > 1e-3, f"joint pos did not vary across resets (spread={spread})"
env.close()
print(f"S3 OK: 5 resets, obs dim 27, joint spread={spread:.4f}")
sys.stdout.flush()
app.close()
