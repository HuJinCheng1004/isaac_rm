# S1 — env instantiation smoke for Isaac-RM-Lift-Block-v0
import sys
sys.path.insert(0, "source/chassis_nav")
from isaaclab.app import AppLauncher

app = AppLauncher({"headless": True, "enable_cameras": False}).app

import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import chassis_nav.tasks  # noqa: F401

cfg = parse_env_cfg("Isaac-RM-Lift-Block-v0", device="cuda:0", num_envs=1)
env = gym.make("Isaac-RM-Lift-Block-v0", cfg=cfg, render_mode=None)
print("obs_space:", env.observation_space)
print("act_space:", env.action_space)
env.close()
# Print the verdict BEFORE app.close(): Isaac Sim's simulation_app.close() hard-exits
# the interpreter, so any statement after it never runs / flushes.
print("S1 OK: env instantiated")
sys.stdout.flush()
app.close()
