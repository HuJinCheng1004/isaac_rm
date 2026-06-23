"""S6 smoke for Isaac-RM-Push-Block-v0 (iter 3 — contact-gated precision tracking).

Verifies:
  1. Total reward is finite for every env across a rollout (no NaN/Inf).
  2. Total reward (mean across envs) is non-constant over time.
  3. Per-term reward values (read from reward_manager._step_reward / _episode_sums)
     are finite and correctly signed:
       - reaching_block, block_to_goal_tracking, *_mid_band, *_fine_grained, success >= 0
       - action_rate, joint_vel <= 0 (regularizer penalties)
  4. block_at_goal sets env._reach_success (harbor eval reads this), shape (num_envs,)
     with values in {0,1}.
  5. Composer = sum: per-term step contributions sum to the total reward.
  6. CONTACT GATE (iter 3): block_to_goal_distance_contact_gated is ~0 when the EE is
     far from the block and substantially > the far value when the EE is near the block,
     at an IDENTICAL block->goal distance (so the change is the gate, not the tracking).

Runs on the PLAY task (16 envs) — deterministic, no obs corruption.
"""
import argparse
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
args.headless = True
args.enable_cameras = False
sim_app = AppLauncher(args).app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import chassis_nav.tasks  # noqa: F401, E402  (registers Isaac-RM-Push-Block-*)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from chassis_nav.tasks.push.mdp.rewards import (  # noqa: E402
    block_to_goal_distance_contact_gated,
)
from chassis_nav.robots.arm import EE_BODY_NAME  # noqa: E402

TASK_ID = "Isaac-RM-Push-Block-Play-v0"


def _to_np(t):
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


cfg = parse_env_cfg(TASK_ID, device="cuda:0", num_envs=16, use_fabric=True)
env = gym.make(TASK_ID, cfg=cfg)
unwrapped = env.unwrapped
rm = unwrapped.reward_manager

# Expected sign per term: +1 = reward (>=0), -1 = penalty (<=0).
EXPECTED_SIGN = {
    "reaching_block": +1,
    "block_to_goal_tracking": +1,
    "block_to_goal_tracking_mid_band": +1,
    "block_to_goal_tracking_fine_grained": +1,
    "success": +1,
    "action_rate": -1,
    "joint_vel": -1,
}

obs, info = env.reset(seed=0)

step_means = []
all_finite = True
saw_reach_success = False
composer_max_err = 0.0
n_steps = 30

act_dim = env.action_space.shape[-1]
for _ in range(n_steps):
    a = torch.rand((cfg.scene.num_envs, act_dim), device=unwrapped.device) * 2.0 - 1.0
    obs, r, term, trunc, info = env.step(a)

    r_np = _to_np(r).reshape(-1)
    if not np.all(np.isfinite(r_np)):
        all_finite = False
    step_means.append(float(r_np.mean()))

    # --- per-term step contributions ---
    # NOTE: RewardManager stores `_step_reward[:, j] = (weight * dt * func) / dt`
    # i.e. the NOMINAL per-step value (sign-preserving). The actual contribution
    # that sums to the returned reward (_reward_buf) is `_step_reward * dt`.
    dt = unwrapped.step_dt
    step_terms = rm._step_reward  # (num_envs, num_terms), nominal (pre-dt) values
    term_names = rm.active_terms
    step_terms_np = _to_np(step_terms)
    composed = (step_terms_np * dt).sum(axis=1).reshape(-1)
    composer_max_err = max(composer_max_err, float(np.max(np.abs(composed - r_np))))

    # finite + sign check per term
    for j, name in enumerate(term_names):
        col = step_terms_np[:, j]
        assert np.all(np.isfinite(col)), f"non-finite values in term {name}"
        sign = EXPECTED_SIGN.get(name)
        if sign == +1:
            assert np.all(col >= -1e-6), f"term {name} expected >=0, got min={col.min():.4f}"
        elif sign == -1:
            assert np.all(col <= 1e-6), f"term {name} expected <=0, got max={col.max():.4f}"

    # --- env._reach_success side-effect from block_at_goal ---
    rs = getattr(unwrapped, "_reach_success", None)
    if rs is not None:
        saw_reach_success = True
        rs_np = _to_np(rs).reshape(-1)
        assert rs_np.shape[0] == cfg.scene.num_envs, (
            f"_reach_success shape {rs_np.shape} != num_envs {cfg.scene.num_envs}"
        )
        assert np.all((rs_np == 0.0) | (rs_np == 1.0)), "._reach_success not binary"

assert all_finite, "NaN or Inf in total reward over rollout"
arr = np.asarray(step_means)
assert arr.std() > 0, f"reward (mean across envs) constant over {n_steps} steps (mean={arr.mean()})"
assert composer_max_err < 1e-4, f"composer != sum: max|sum(terms)-reward|={composer_max_err:.6f}"
assert saw_reach_success, "block_at_goal never set env._reach_success"

# -----------------------------------------------------------------------------
# CONTACT-GATE BEHAVIOR (iter 3). The gated term = tracking(d_block_goal) * gate(d_ee_block).
# We isolate the GATE factor (independent of tracking magnitude) by comparing the gated
# reward to the UNGATED tracking at the SAME block placement, via the gate RATIO
# r = gated / ungated  (= gate value, in [0,1]):
#   (B) EE NEAR block (block teleported to the EE)  -> gate ~1  -> ratio ~1
#   (A) EE FAR  from block (block 0.4 m below EE)    -> gate ~0  -> ratio ~0
# This proves the gate opens on contact and closes on abandonment regardless of how
# far the block is from the goal at the probe placement.
# -----------------------------------------------------------------------------
unwrapped.reset(seed=1)
robot = unwrapped.scene["robot"]
block = unwrapped.scene["object"]
ee_cfg = SceneEntityCfg("robot", body_names=[EE_BODY_NAME])
ee_cfg.resolve(unwrapped.scene)
ee_w = robot.data.body_pos_w[:, ee_cfg.body_ids[0]]  # (N,3)
# resolved cfg for direct reward-function calls below (defaults are not auto-resolved)
gate_robot_cfg = SceneEntityCfg("robot", body_names=[EE_BODY_NAME])
gate_robot_cfg.resolve(unwrapped.scene)

def _set_block_xyz(pos_w):
    # Teleport the block and refresh its data buffer WITHOUT advancing physics
    # (a sim.step would let the block fall / depenetrate away from the EE, corrupting
    # the synthetic d_ee_block). write_root_pose_to_sim + a buffer refresh is enough:
    # the reward funcs read block.data.root_pos_w, which we re-derive here.
    root_pose = block.data.root_pose_w.clone()
    root_pose[:, 0:3] = pos_w
    block.write_root_pose_to_sim(root_pose)
    vel = torch.zeros_like(block.data.root_vel_w)
    block.write_root_velocity_to_sim(vel)
    # refresh the cached data view so root_pos_w reflects the write immediately
    block.update(dt=0.0)

def _ungated_tracking(std):
    # 1 - tanh(d_block_goal / std) in env-origin frame (matches block_to_goal_distance)
    command = unwrapped.command_manager.get_command(cmd_name)
    des_pos_w = unwrapped.scene.env_origins + command[:, :3]
    d = torch.norm(des_pos_w - block.data.root_pos_w, dim=1)
    return _to_np(1 - torch.tanh(d / std)).reshape(-1)

cmd_name = "object_pose"
EPS = 1e-6

# (B) NEAR: block at the EE position -> d_ee_block ~ 0 -> gate ~1
_set_block_xyz(ee_w.clone())
near_mid = _to_np(block_to_goal_distance_contact_gated(
    unwrapped, std=0.12, std_gate=0.05, command_name=cmd_name, robot_cfg=gate_robot_cfg)).reshape(-1)
near_fine = _to_np(block_to_goal_distance_contact_gated(
    unwrapped, std=0.08, std_gate=0.05, command_name=cmd_name, robot_cfg=gate_robot_cfg)).reshape(-1)
near_gate_mid = near_mid / (_ungated_tracking(0.12) + EPS)   # ~= gate value
near_gate_fine = near_fine / (_ungated_tracking(0.08) + EPS)
# re-read EE (unchanged by block teleport) for the FAR placement
ee_w = robot.data.body_pos_w[:, ee_cfg.body_ids[0]]

# (A) FAR: block 0.4 m below the EE in z (off-contact).
far_pos = ee_w.clone()
far_pos[:, 2] = far_pos[:, 2] - 0.40
_set_block_xyz(far_pos)
far_mid = _to_np(block_to_goal_distance_contact_gated(
    unwrapped, std=0.12, std_gate=0.05, command_name=cmd_name, robot_cfg=gate_robot_cfg)).reshape(-1)
far_fine = _to_np(block_to_goal_distance_contact_gated(
    unwrapped, std=0.08, std_gate=0.05, command_name=cmd_name, robot_cfg=gate_robot_cfg)).reshape(-1)
far_gate_mid = far_mid / (_ungated_tracking(0.12) + EPS)
far_gate_fine = far_fine / (_ungated_tracking(0.08) + EPS)

# Gate must CLOSE (~0) when EE is 0.4 m off-contact (1-tanh(0.4/0.05)=1-tanh(8)~3e-7).
assert np.all(far_gate_mid < 1e-2), f"gate NOT closed when EE far: max ratio={far_gate_mid.max():.4f}"
assert np.all(far_gate_fine < 1e-2), f"gate NOT closed when EE far: max ratio={far_gate_fine.max():.4f}"
# Gate must OPEN (~1) when the EE is in contact with the block.
assert np.all(near_gate_mid > 0.9), f"gate NOT open on contact: min ratio={near_gate_mid.min():.4f}"
assert np.all(near_gate_fine > 0.9), f"gate NOT open on contact: min ratio={near_gate_fine.min():.4f}"

verdict = (
    f"active reward terms: {list(rm.active_terms)}\n"
    f"composer max|sum(terms)-reward| = {composer_max_err:.3e}  (sum-verified)\n"
    f"_reach_success: shape ({cfg.scene.num_envs},), binary, present\n"
    f"contact-gate ratio: far(mid={far_gate_mid.mean():.4f},fine={far_gate_fine.mean():.4f}) ~0 "
    f"(gate closed) ; near(mid={near_gate_mid.mean():.4f},fine={near_gate_fine.mean():.4f}) ~1 "
    f"(gate open)\n"
    f"S6 OK: {n_steps} steps x {cfg.scene.num_envs} envs, "
    f"reward mean={arr.mean():.3f} std={arr.std():.3f} composer=sum-verified"
)
# Isaac's sim_app.close() can swallow trailing stdout; persist the verdict to a file
# so the result is reliably visible regardless of stdout teardown.
with open("/tmp/s6_verdict.txt", "w") as f:
    f.write(verdict + "\n")
print(verdict, flush=True)
env.close()
sim_app.close()
