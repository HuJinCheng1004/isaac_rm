"""渲染推方块任务的 reset 初始姿态，保存为 PNG，用于人工检查关节 home 位是否合理。

用法：
    cd /home/shihao/isaac_rm
    .venv/bin/python harbor/scripts/peek_reset.py task=Isaac-RM-Push-Block-v0
"""
from __future__ import annotations
import os, sys

def _parse(argv):
    out = {}
    for tok in argv:
        t = tok.lstrip("+")
        if "=" in t:
            k, v = t.split("=", 1)
            out[k] = v
    return out

_ARGS = _parse(sys.argv[1:])

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

import chassis_nav  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

def main():
    task = _ARGS.get("task", "Isaac-RM-Push-Block-v0")
    out_path = _ARGS.get("output", "/home/shihao/isaac_rm/source/chassis_nav/chassis_nav/tasks/push/test_view/peek_reset.png")
    num_envs = int(_ARGS.get("num_envs", 4))
    device = "cuda:0"

    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    # 从桌子 +y 侧斜上方俯视：能同时看到手臂姿态、桌面、方块
    env_cfg.viewer.eye = (-3.0, 8.0, 4.0)
    env_cfg.viewer.lookat = (-0.3, -0.2, -0.3)
    # 提高输出图片分辨率（默认 1280x720），可用 width/height 覆盖
    _w = int(_ARGS.get("width", 2560))
    _h = int(_ARGS.get("height", 1440))
    env_cfg.viewer.resolution = (_w, _h)
    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array")
    raw_env = env.unwrapped

    # reset → warm up 几步让渲染器稳定，再取帧
    env.reset()
    warmup = int(_ARGS.get("warmup", 5))
    zero_action = torch.zeros(num_envs, env.action_space.shape[-1], device=device)
    frame = None
    for _ in range(warmup):
        env.step(zero_action)
        f = raw_env.render()
        if f is not None:
            frame = f

    if frame is None:
        print("[peek] ERROR: render() returned None after warmup", flush=True)
        os._exit(1)

    frame = np.asarray(frame, dtype=np.uint8)
    imageio.imwrite(out_path, frame)
    print(f"[peek] saved {frame.shape} → {out_path}", flush=True)
    sys.stdout.flush()
    os._exit(0)

if __name__ == "__main__":
    main()
