"""S6 smoke — reward is finite, non-constant, and composer matches per-term decomposition.

Runs at num_envs=128. Reward is a (128,) tensor each step; the contract checks
that EVERY env returns a finite scalar, that the cross-step time series of the
mean-across-envs has non-zero std, and that the composer match holds element-wise
across all envs at every step (when `info["detailed_reward"]` is present).

Substitutions:
  Isaac-RM-Lift-Block-v0 — gym task id
"""
import argparse
import os
import sys

import numpy as np

# Make the in-repo chassis_nav package importable (registers Isaac-RM-* gym ids).
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "source", "chassis_nav"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
args.headless = True
args.enable_cameras = False
sim_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import isaaclab_tasks    # noqa: F401, E402
import chassis_nav.tasks  # noqa: F401, E402  (registers Isaac-RM-* gym ids)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _to_np(t):
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


cfg = parse_env_cfg("Isaac-RM-Lift-Block-v0", device="cuda:0", num_envs=128, use_fabric=True)
env = gym.make("Isaac-RM-Lift-Block-v0", cfg=cfg)
obs, info = env.reset(seed=0)

step_means = []                  # mean reward across envs at each step (for time-std)
term_sums = {}                   # name -> running sum (mean across envs)
saw_detailed = False
all_finite = True
reach_success_ok = False

import torch  # noqa: E402  (env expects a torch action tensor, not numpy)

# Manager-based env: read per-term contributions straight off the RewardManager
# (no _DetailedRewardWrapper in raw gym.make create mode, so info["detailed_reward"]
# is absent). reward_manager._step_reward holds the dt-scaled, weighted per-term reward
# from the LAST compute() call -> shape (num_envs, num_terms); summing across terms must
# reproduce the env's scalar reward (composer = sum).
rm = env.unwrapped.reward_manager
term_names = list(rm.active_terms)
# RewardManager stores _step_reward as value/dt (un-scaled), while the returned scalar
# reward accumulates the dt-scaled value. Multiply back by step_dt to reconstruct it.
_step_dt = env.unwrapped.step_dt

for _ in range(30):
    a = env.action_space.sample()
    a = torch.as_tensor(a, dtype=torch.float32, device=env.unwrapped.device)
    obs, r, term, trunc, info = env.step(a)

    r_np = _to_np(r).reshape(-1)                      # (num_envs,)
    if not np.all(np.isfinite(r_np)):
        all_finite = False
    step_means.append(float(r_np.mean()))

    # --- per-term decomposition + composer=sum check via the RewardManager ---
    step_reward = _to_np(rm._step_reward) * _step_dt  # (num_envs, num_terms), dt-scaled
    saw_detailed = True
    composed = step_reward.sum(axis=1)
    assert np.allclose(composed, r_np, atol=1e-5), (
        "composer mismatch (RewardManager terms do not sum to scalar reward): "
        f"max|sum(terms) - reward|={np.max(np.abs(composed - r_np)):.6f}"
    )
    for j, name in enumerate(term_names):
        col = step_reward[:, j]
        assert np.all(np.isfinite(col)), f"non-finite per-term reward: {name}"
        term_sums[name] = term_sums.get(name, 0.0) + float(col.mean())

    # --- success side-effect: env._reach_success must be set every step ---
    rs = getattr(env.unwrapped, "_reach_success", None)
    assert rs is not None and hasattr(rs, "shape"), "env._reach_success not set by success term"
    assert int(rs.shape[0]) == r_np.shape[0], "env._reach_success has wrong env count"
    reach_success_ok = True
    # IsaacLab auto-resets terminated envs internally — no manual reset needed.

assert all_finite, "NaN or Inf in reward over 30 steps"
assert reach_success_ok, "env._reach_success was never observed"
arr = np.asarray(step_means)
assert arr.std() > 0, f"reward (mean across envs) is constant over 30 steps (mean={arr.mean()})"

print("per-term episodic mean (summed over 30 steps, dt-scaled, mean across envs):")
for k, v in sorted(term_sums.items(), key=lambda kv: -abs(kv[1])):
    print(f"  {k}: {v:+.6f}")

print(
    f"S6 OK: 30 steps, reward mean={arr.mean():.3f} std={arr.std():.3f} "
    f"composer={'sum-verified' if saw_detailed else 'passthrough'} "
    f"reach_success=set"
)
env.close()
sim_app.close()
