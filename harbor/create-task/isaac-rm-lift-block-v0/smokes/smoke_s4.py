# S4 — termination smoke: 210 steps, episode times out at step 200
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
env.reset()

# Hold a zero action (no-op-ish) and count steps until truncation (time_out).
act = torch.zeros(env.action_space.shape, device="cuda:0")
timeout_step = None
for step in range(1, 211):
    obs, rew, term, trunc, info = env.step(act)
    if bool(torch.as_tensor(trunc).any()):
        timeout_step = step
        break

assert timeout_step is not None, "episode never truncated within 210 steps"
assert 195 <= timeout_step <= 205, f"timeout at step {timeout_step}, expected ~200"
env.close()
print(f"S4 OK: time_out at step {timeout_step}")
sys.stdout.flush()
app.close()
