"""chassis_nav — harbor local_implementation 渲染入口（薄壳，包住 skrl 策略回放）。

加载一个 skrl checkpoint，在少量并行环境里跑确定性 rollout，用 IsaacLab 视口
``env.render()``（rgb_array）收帧写成 MP4，供 ``/harbor:reward-tune`` 的 ANALYZE 步
抽帧做行为分析。

harbor 调用约定::

    python harbor/scripts/rl/skrl_local/render.py \
    checkpoint=/home/shihao/isaac_rm/harbor/outputs/ppo_Isaac-RM-Lift-Block-v0_20260622-154131/skrl/checkpoints/best_agent.pt \
    task=Isaac-RM-Lift-Block-v0 render_seconds=20 +gpu_sim=true

默认把 MP4 写到 checkpoint 同目录的 ``render.mp4``。

Isaac Sim 5.1 + IsaacLab 的 ``sim_app.close()`` 会在 USD stage detach 处挂死，因此 MP4
落盘后立即 ``os._exit(0)`` 硬退出（编排方无需 pkill 看门狗）。
"""
from __future__ import annotations

import os
import sys


def _parse(argv: list[str]) -> dict:
    out: dict[str, str] = {}
    for tok in argv:
        t = tok.lstrip("+")
        if t.startswith("--config-name="):
            continue
        if "=" in t:
            k, v = t.split("=", 1)
            out[k] = v
    return out


_ARGS = _parse(sys.argv[1:])

# --- 启动 Isaac Sim（开相机渲染）---
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

import chassis_nav  # noqa: F401  (registers tasks)
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner


@torch.no_grad()
def main() -> int:
    task = _ARGS.get("task", "Isaac-RM-Push-Block-v0")
    checkpoint = _ARGS["checkpoint"]
    num_envs = int(_ARGS.get("render_num_envs", _ARGS.get("num_envs", 4)))
    out_path = _ARGS.get("output", os.path.join(os.path.dirname(os.path.abspath(checkpoint)), "render.mp4"))
    device = "cuda:0"

    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)

    # --- 渲染视角（IsaacLab 视口相机）---
    # eye/lookat 为世界坐标 (x, y, z)；origin_type/env_index 决定跟随哪个 env。
    # 场景布局（env-local，env0 origin=(0,0,0)）：
    #   机器人底座 (0.5, -0.3, -0.805) · 桌面 (-0.5, -0.2, 0.0) · 方块 (-0.1, -0.1, 0.055)
    # 相机设在机器人右前方、桌面高度以上，注视 EE-方块 交互区。
    env_cfg.viewer.eye = (0.8, 0.5, 0.6)        # 右前上方，贴近交互区
    env_cfg.viewer.lookat = (-0.05, -0.15, 0.12) # 注视方块上方 ~12cm 处
    env_cfg.viewer.resolution = (1280, 720)     # 输出分辨率 (w, h)
    env_cfg.viewer.origin_type = "env"          # "world" | "env" | "asset_root"
    env_cfg.viewer.env_index = 0                # origin_type="env" 时跟随的环境

    agent_cfg = load_cfg_from_registry(task, "skrl_cfg_entry_point")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["wandb"] = False

    env = gym.make(task, cfg=env_cfg, render_mode="rgb_array")
    raw_env = env.unwrapped
    env = SkrlVecEnvWrapper(env, ml_framework="torch")

    runner = Runner(env, agent_cfg)
    runner.agent.load(checkpoint)
    runner.agent.set_running_mode("eval")
    print(f"[render] loaded {checkpoint}; rolling out {task} x{num_envs}", flush=True)

    frames: list[np.ndarray] = []
    obs, _ = env.reset()
    fps = int(1.0 / float(raw_env.step_dt)) if raw_env.step_dt else 10
    # render_seconds: 目标视频时长（秒）。默认跑满一个 episode；给了就按秒数算步数
    # （环境到 episode 末会自动 reset，所以可以超过单个 episode 长度）。
    render_seconds = _ARGS.get("render_seconds")
    if render_seconds is not None:
        max_steps = max(1, int(round(float(render_seconds) * fps)))
    else:
        max_steps = int(raw_env.max_episode_length) + 1
    moved = False
    for t in range(max_steps):
        actions = runner.agent.act(obs, t, max_steps)[0]
        if float(actions.abs().sum()) > 0:
            moved = True
        obs, _, _, _, _ = env.step(actions)
        frame = raw_env.render()
        if frame is not None:
            frames.append(np.asarray(frame, dtype=np.uint8))

    if not frames:
        print("[render] ERROR: env.render() produced no frames", flush=True)
        os._exit(1)
    # 帧间差异性 sanity（避免静帧/冻结失败模式）
    diff = float(np.abs(frames[-1].astype(np.int32) - frames[0].astype(np.int32)).mean())
    imageio.mimsave(out_path, frames, fps=fps)
    size_kb = os.path.getsize(out_path) / 1024.0
    print(f"[render] wrote {len(frames)} frames -> {out_path} ({size_kb:.1f} KB); "
          f"inference_moved={moved} frame_diff={diff:.2f}", flush=True)

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
