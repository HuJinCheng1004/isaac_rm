"""S6 render smoke — records a short random-action MP4 for human acceptance review.

Runs at num_envs=4, render_seconds=8, saves to:
  harbor/create-task/isaac-rm-lift-block-v0/smoke_s6_render.mp4

Usage:
    cd /home/shihao/isaac_rm
    OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python \
        harbor/create-task/isaac-rm-lift-block-v0/smokes/smoke_s6_render.py
"""
from __future__ import annotations
import os, sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "source", "chassis_nav"))

from isaaclab.app import AppLauncher
import argparse

_p = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(_p)
_a, _ = _p.parse_known_args(["--headless", "--enable_cameras"])
app_launcher = AppLauncher(_a)
simulation_app = app_launcher.app

import numpy as np
import torch
import gymnasium as gym
import imageio.v2 as imageio

import chassis_nav.tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

TASK = "Isaac-RM-Lift-Block-v0"
OUT = os.path.join(os.path.dirname(__file__), "..", "smoke_s6_render.mp4")
NUM_ENVS = 4
RENDER_SECONDS = 12

env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=NUM_ENVS)
# Camera: positioned far in the -y direction (front of the scene), elevated, looking at the
# center of the 4-env grid (2×2, spacing 2.5 m, envs at ≈ ±1.25 in x/y world).
# With origin_type="world" the eye/lookat are absolute world coordinates.
# This angle puts all 4 robots spread left-to-right across the image frame.
env_cfg.viewer.eye = (0.0, -9.0, 6.0)       # far in front, high up
env_cfg.viewer.lookat = (0.0, 0.0, 0.5)      # center of the 4-env grid
env_cfg.viewer.resolution = (1920, 1080)

env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array")
raw_env = env.unwrapped
fps = max(1, int(round(1.0 / float(raw_env.step_dt))))
max_steps = max(1, int(round(RENDER_SECONDS * fps)))

obs, _ = env.reset(seed=0)
frames: list[np.ndarray] = []
device = raw_env.device
ACTION_SCALE = 0.08  # very small perturbations → slow continuous motion, arm stays above table

for t in range(max_steps):
    # Near-zero actions: tiny random perturbations around home pose, arm stays above table
    a = env.action_space.sample()
    a = a * ACTION_SCALE
    a = torch.as_tensor(a, dtype=torch.float32, device=device)
    obs, _, _, _, _ = env.step(a)
    frame = raw_env.render()
    if frame is not None:
        frames.append(np.asarray(frame, dtype=np.uint8))

if not frames:
    print("ERROR: no frames captured — check enable_cameras flag")
    sys.exit(1)

out_path = os.path.abspath(OUT)
imageio.mimsave(out_path, frames, fps=fps)
size_kb = os.path.getsize(out_path) / 1024.0
print(f"S6 render OK: {len(frames)} frames -> {out_path} ({size_kb:.1f} KB)")
env.close()
simulation_app.close()
