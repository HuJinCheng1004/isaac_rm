# Reward state — `Isaac-RM-Push-Block-v0` (iter 3, 2026-06-18T05:55:38Z)

| Field | Value |
|---|---|
| composer | sum (RewardManager sums `weight_i * dt * func_i` per step; dt = 2/60 s) |
| per_term_logging | yes (repo wiring: `reward_manager._step_reward` / `_episode_sums` → metrics.jsonl `reward/<term>/episodic_return_mean`) |
| success metric | `eval/success_rate` from `env._reach_success` (set by `block_at_goal`, threshold 0.05 m, **XY** block→goal distance) |
| reward_path | `source/chassis_nav/chassis_nav/tasks/push/mdp/rewards.py` |
| env_cfg_path | `source/chassis_nav/chassis_nav/tasks/push/push_env_cfg.py` (RewardsCfg block) |

## Iter-3 change (CONTROL fix — incentivize SUSTAINED EE-block contact)

iters 1-2 plateaued at success ~0.10: the policy learned an **open-loop swat** (one ballistic
shove, EE retracts to ~31 cm, block coasts to a stop ~10 cm short). Reward-gradient shaping is
exhausted. iter 3 makes closed-loop pushing beat the swat with the brief's (a)+(b) combo:

1. **(a) reaching_block weight 1.0 → 4.0** — sustained EE-block proximity is now worth keeping
   each step (was dwarfed by tracking total 29); still << coarse 16 so the goal signal leads.
2. **(b) mid_band + fine precision tracking are now CONTACT-GATED** via the new
   `block_to_goal_distance_contact_gated`: tracking × `g=1-tanh(d_ee_block/0.05)`. Precision
   reward is paid only while the EE touches the block (g≈0 at the iter-2 31 cm abandon distance),
   removing the shove-and-leave exploit. COARSE attractor stays UNGATED for long-range initiation.

No (c) ee_z_to_block — iter-2 visual shows lateral contact loss (arm retracts), not vertical
hover/lift-off, so vertical alignment is not the failure mode.

## Term list

| name | weight | shape | std / params | gate |
|---|---:|---|---|---|
| reaching_block | 4.0 | `1 - tanh(d_ee_block / std)` | std=0.1, body=`r_link8` | none; **w 1.0→4.0 (iter 3)** |
| block_to_goal_tracking | 16.0 | `1 - tanh(d_block_goal / std)` | std=0.3 (coarse), env-origin goal | **UNGATED** (long-range) |
| block_to_goal_tracking_mid_band | 8.0 | tracking × contact-gate | std=0.12, **gate std_gate=0.05 (iter 3)** | EE within ~5-8 cm of block |
| block_to_goal_tracking_fine_grained | 5.0 | tracking × contact-gate | std=0.08, **gate std_gate=0.05 (iter 3)** | EE within ~5-8 cm of block |
| success | 1e-6 | `1.0 if d_xy_block_goal < thr else 0.0` | thr=0.05, **XY**; sets `env._reach_success` | none (logging-only) |
| action_rate | -1e-4 | `-‖a_t - a_{t-1}‖²` | library `action_rate_l2` | none |
| joint_vel | -1e-4 | `-‖q̇‖²` | library `joint_vel_l2`, arm joints | none |

Composer = SUM. Contact gate `g = 1 - tanh(d_ee_block / 0.05)`: g(0)=1.0, g(5cm)=0.24,
g(10cm)=0.04, g(31cm)≈0. Coarse → mid → fine kernels hand off monotonically; mid/fine now
only pay while in contact.

## Magnitude budget (per-step saturated maxima, pre-dt-scale)

- reaching_block: 4.0 × 1.0 = **4.0** (hand on block, sustained)
- block_to_goal_tracking (coarse): 16.0 × 1.0 = **16.0** (block at goal)
- block_to_goal_tracking_mid_band: 8.0 × 1.0 × g = **8.0 × g** (block at goal AND in contact)
- block_to_goal_tracking_fine_grained: 5.0 × 1.0 × g = **5.0 × g** (block at goal AND in contact)
- success: 1e-6 (logging-only side-effect)
- action_rate / joint_vel: small negative regularizers
- Saturated total ≈ **33.0/step** (pre-dt) when block sits at goal AND hand stays in contact;
  drops to ~20/step (coarse + reaching only) if the hand leaves — the swat is now penalized.

`ee_z_to_block` (exp z-alignment, std=0.5) remains DEFINED in rewards.py but NOT wired — held
in reserve for a vertical-contact-loss failure mode (not observed; iter-2 loss is lateral).

## Cross-section edits in this iter (if any)

| Section | File | Change | Rationale |
|---|---|---|---|
| (none) | — | — | permit_env_edits granted but unused; contact gate reuses the existing `r_link8` EE body + object — no new sensor / obs / cfg field required. Whole change is in §6. |
