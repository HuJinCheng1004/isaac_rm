# S5 — observation smoke: obs dim 27, all finite, term order matches expected layout
import sys
sys.path.insert(0, "source/chassis_nav")
from isaaclab.app import AppLauncher

app = AppLauncher({"headless": True, "enable_cameras": False}).app

import torch
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import chassis_nav.tasks  # noqa: F401

EXPECTED_TERMS = [
    ("joint_pos", 6),
    ("joint_vel", 6),
    ("gripper_pos", 2),
    ("object_position", 3),
    ("ee_position", 3),
    ("actions", 7),
]

cfg = parse_env_cfg("Isaac-RM-Lift-Block-v0", device="cuda:0", num_envs=1)
env = gym.make("Isaac-RM-Lift-Block-v0", cfg=cfg, render_mode=None)
obs, _ = env.reset()
obs_vec = obs["policy"] if isinstance(obs, dict) else obs

# verify term order + dims via the observation manager
term_dims = env.unwrapped.observation_manager._group_obs_term_dim["policy"]
term_names = env.unwrapped.observation_manager.active_terms["policy"]
got = [(n, int(d[0]) if hasattr(d, "__len__") else int(d)) for n, d in zip(term_names, term_dims)]
assert got == EXPECTED_TERMS, f"obs layout mismatch:\n got={got}\n exp={EXPECTED_TERMS}"

assert obs_vec.shape[-1] == 27, f"obs dim {obs_vec.shape[-1]} != 27"
assert torch.isfinite(obs_vec).all(), "obs contains non-finite values"
env.close()
print(f"S5 OK: shape={list(obs_vec.shape)}, layout verified, all finite")
sys.stdout.flush()
app.close()
