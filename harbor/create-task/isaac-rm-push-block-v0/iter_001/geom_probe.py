"""iter-1 geometry probe: verify the goal-frame fix.

Reset + settle (2 zero-action steps), num_envs=8, then read:
  - robot articulation root z (expected ~-0.805, the underground chassis root)
  - block z (expected ~table height)
  - goal_w via env-origin frame (expected z ~0.04, table height, NOT negative)
  - xy distance block->goal (expected <= ~0.25 m, in the reachable push region)
Writes results to /tmp/geom_probe.txt then os._exit(0).
"""
import argparse
import os
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
args.headless = True
args.enable_cameras = False
sim_app = AppLauncher(args).app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import chassis_nav.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK_ID = "Isaac-RM-Push-Block-v0"

cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=8, use_fabric=True)
env = gym.make(TASK_ID, cfg=cfg)
u = env.unwrapped

env.reset(seed=0)
act_dim = env.action_space.shape[-1]
zero = torch.zeros((cfg.scene.num_envs, act_dim), device=u.device)
for _ in range(2):
    env.step(zero)

robot = u.scene["robot"]
obj = u.scene["object"]
cmd = u.command_manager.get_command("object_pose")
des_pos_b = cmd[:, :3]

env_origins = u.scene.env_origins
goal_w = env_origins + des_pos_b                       # iter-1 fix: env-origin frame
block_w = obj.data.root_pos_w
root_w = robot.data.root_pos_w

dist_xy = torch.norm(goal_w[:, :2] - block_w[:, :2], dim=1)
dist_3d = torch.norm(goal_w - block_w, dim=1)

lines = []
lines.append("=== iter-1 GEOMETRY PROBE (num_envs=8, reset+2 zero steps) ===")
for i in range(cfg.scene.num_envs):
    lines.append(
        f"env{i}: root_w=({root_w[i,0]:.3f},{root_w[i,1]:.3f},{root_w[i,2]:.3f}) "
        f"block_w=({block_w[i,0]:.3f},{block_w[i,1]:.3f},{block_w[i,2]:.3f}) "
        f"goal_w=({goal_w[i,0]:.3f},{goal_w[i,1]:.3f},{goal_w[i,2]:.3f}) "
        f"dist_xy={dist_xy[i]:.3f} dist_3d={dist_3d[i]:.3f}"
    )
lines.append(
    f"SUMMARY: goal_w.z mean={goal_w[:,2].mean():.3f} "
    f"(expect ~0.04, NOT negative); "
    f"dist_xy mean={dist_xy.mean():.3f} max={dist_xy.max():.3f} "
    f"(expect <= ~0.25); root_w.z mean={root_w[:,2].mean():.3f}"
)
out = "\n".join(lines) + "\n"
with open("/tmp/geom_probe.txt", "w") as f:
    f.write(out)
print(out, flush=True)

os._exit(0)
