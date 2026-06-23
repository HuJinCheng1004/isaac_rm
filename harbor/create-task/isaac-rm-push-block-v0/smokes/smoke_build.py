"""§1-§5 build smoke for Isaac-RM-Push-Block-v0.

验证：
  1. env 可创建（gym.make 不报错）
  2. obs(28) / action(6) 维度正确
  3. 步进 20 步随机动作无 NaN/Inf
  4. 打印 EE 位置、方块位置 — 用于工作空间校准

运行方式：
  cd /home/shihao/isaac_rm
  .venv/bin/python harbor/create-task/isaac-rm-push-block-v0/smokes/smoke_build.py
"""
import os, sys
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
sys.path.insert(0, "/home/shihao/isaac_rm/source/chassis_nav")

from isaaclab.app import AppLauncher
app_launcher = AppLauncher({"headless": True, "enable_cameras": False})
simulation_app = app_launcher.app

import torch
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import chassis_nav.tasks

ENV_ID = "Isaac-RM-Push-Block-v0"
N = 16

errors = []
lines = [f"=== smoke_build: {ENV_ID} ==="]

# --- 1. 创建环境 -------------------------------------------------------------
try:
    env_cfg = parse_env_cfg(ENV_ID, device="cuda:0", num_envs=N)
    env = gym.make(ENV_ID, cfg=env_cfg, render_mode=None)
    lines.append(f"  [build] gym.make OK, num_envs={N}")
except Exception as e:
    lines.append(f"FAIL: gym.make: {e}")
    print("\n".join(lines), flush=True)
    simulation_app.close(); sys.exit(1)

raw = env.unwrapped

# --- 2. 维度检查（从 manager 读）-------------------------------------------
act_dim = raw.action_manager.total_action_dim
lines.append(f"  [dims] action={act_dim} (expect 6)")
if act_dim != 6:
    errors.append(f"act_dim={act_dim}, expected 6")

# --- 3. Reset + 几何校准 ----------------------------------------------------
obs_raw, _ = env.reset()
obs_vec = obs_raw["policy"] if isinstance(obs_raw, dict) else obs_raw
obs_dim = obs_vec.shape[-1]
lines.append(f"  [dims] obs={obs_dim} (expect 28)")
if obs_dim != 28:
    errors.append(f"obs_dim={obs_dim}, expected 28")

robot = raw.scene["robot"]
obj   = raw.scene["object"]

ee_ids, _ = robot.find_bodies("r_link8")
ee_pos   = robot.data.body_pos_w[:, ee_ids[0]].mean(dim=0)
blk_pos  = obj.data.root_pos_w.mean(dim=0)
root_pos = robot.data.root_pos_w.mean(dim=0)

lines.append(f"  [init] robot_root  = [{root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f}]")
lines.append(f"  [init] ee r_link8  = [{ee_pos[0]:.3f},  {ee_pos[1]:.3f},  {ee_pos[2]:.3f}]")
lines.append(f"  [init] block       = [{blk_pos[0]:.3f},  {blk_pos[1]:.3f},  {blk_pos[2]:.3f}]")
ee_to_blk = torch.norm(ee_pos - blk_pos).item()
lines.append(f"  [init] |ee-block|  = {ee_to_blk:.3f} m (goal: < 1.0 m after workspace fix)")
if ee_to_blk > 1.5:
    errors.append(f"|ee-block|={ee_to_blk:.3f} m too large — workspace needs calibration")

# --- 4. 随机步进 20 步 -------------------------------------------------------
total_rew = torch.zeros(N, device=raw.device)
finite_ok = True
for step_i in range(20):
    act_t = torch.rand(N, act_dim, device=raw.device) * 2 - 1  # uniform [-1, 1]
    obs, rew, term, trunc, info = env.step(act_t)
    rew_t = rew if isinstance(rew, torch.Tensor) else torch.tensor(rew, device=raw.device)
    total_rew += rew_t
    if not torch.isfinite(rew_t).all():
        finite_ok = False
        errors.append(f"non-finite reward at step {step_i}")

if finite_ok:
    lines.append(f"  [steps] 20 random steps OK, total_rew mean={total_rew.mean().item():.4f}")
else:
    lines.append("  [steps] WARN: non-finite reward")

ee_pos_f = robot.data.body_pos_w[:, ee_ids[0]].mean(dim=0)
lines.append(f"  [final] ee r_link8 = [{ee_pos_f[0]:.3f},  {ee_pos_f[1]:.3f},  {ee_pos_f[2]:.3f}]")

# --- 5. 奖励项验证 -----------------------------------------------------------
term_names = list(raw.reward_manager.active_terms)
lines.append(f"  [reward] terms = {term_names}")
for need in ["reaching_block", "block_to_goal_tracking", "block_to_goal_tracking_fine_grained", "action_rate", "joint_vel"]:
    if need not in term_names:
        errors.append(f"missing reward term: {need}")

# --- 6. 命令维度 -------------------------------------------------------------
cmd = raw.command_manager.get_command("object_pose")
lines.append(f"  [cmd] object_pose shape={list(cmd.shape)} (expect [{N}, 7])")
if list(cmd.shape) != [N, 7]:
    errors.append(f"cmd shape {list(cmd.shape)} != [{N}, 7]")

env.close()

# --- 输出 -------------------------------------------------------------------
if errors:
    lines.append("FAIL: " + "; ".join(errors))
else:
    lines.append(f"OK: obs={obs_dim}, action={act_dim}, 20 steps finite, {len(term_names)} reward terms")

result = "\n".join(lines)
print(result, flush=True)
os.makedirs("/tmp/rm_push_smoke", exist_ok=True)
with open("/tmp/rm_push_smoke/build_result.txt", "w") as f:
    f.write(result + "\n")

simulation_app.close()
if errors:
    sys.exit(1)
