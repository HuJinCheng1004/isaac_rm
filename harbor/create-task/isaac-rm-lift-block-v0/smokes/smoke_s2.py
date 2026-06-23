# S2 — action shape smoke (6 arm + 1 binary gripper = 7) for Isaac-RM-Lift-Block-v0
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
obs, _ = env.reset()
# IsaacLab's ManagerBasedRLEnv.step calls .to(device) on the action -> needs a torch tensor.
act = torch.from_numpy(env.action_space.sample()).to("cuda:0")
obs2, rew, term, trunc, info = env.step(act)
assert act.shape[-1] == 7, f"action dim {act.shape[-1]} != 7"
env.close()
print("S2 OK:", tuple(act.shape))
sys.stdout.flush()
app.close()
