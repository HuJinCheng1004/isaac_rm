# Reward state — `Isaac-RM-Lift-Block-v0` (iter 0, 2026-06-22)

| Field | Value |
|---|---|
| composer | sum |
| per_term_logging | yes (RewardManager `_step_reward` decomposes; `info["detailed_reward"]` once `/harbor:reward-add-log` wraps the env) |
| dt_scaling | LEFT IN PLACE (push/Franka repo convention; success weight 1e-6 calibrated for it) |
| reward_path | `source/chassis_nav/chassis_nav/tasks/lift/mdp/rewards.py` |
| env_cfg_path | `source/chassis_nav/chassis_nav/tasks/lift/lift_env_cfg.py` (RewardsCfg block) |
| smoke | `harbor/create-task/isaac-rm-lift-block-v0/smokes/smoke_s6.py` (S6 OK) |

## Term list

| name | weight | func | shape | gate | params |
|---|---:|---|---|---|---|
| reaching_block | 1.0 | `object_ee_distance_body` | tanh [0,1] | none | std=0.1, robot_cfg=r_link8 |
| lifting_object | 15.0 | `object_is_lifted` | binary {0,1} | none | minimal_height=0.04 |
| cube_height | 16.0 | `cube_height_reward` | tanh [0,1] | cube z > 0.04 | target_height=0.20, std=0.15, minimal_height=0.04, robot_cfg=r_link8 |
| grasping | 5.0 | `grasping_reward` | tanh [0,1] | cube z > 0.04 | std_gate=0.05, minimal_height=0.04, robot_cfg=r_link8 |
| success | 1e-6 | `block_lifted_and_grasped` | binary {0,1} | — | lift_height=0.15, contact_threshold=0.08, robot_cfg=r_link8; sets `env._reach_success` |
| action_rate | -1e-4 | `action_rate_l2` (stock) | L2 penalty | none | — |
| joint_vel | -1e-4 | `joint_vel_l2` (stock) | L2 penalty | none | asset_cfg=arm joints |

## Cross-section edits in this iter

None. §1–§5 already exposed everything the reward needs (EE body id, cube world pose, arm-joint cfg). §7 DR untouched.

## Open questions for next iter (training-evidence dependent)

- `cube_height` target 0.20 m vs success line 0.15 m: chosen so the tanh kernel still has gradient at success. If the policy parks the cube just under 0.15 m, consider raising target or tightening std.
- `grasping`/`success` use an EE->cube distance proxy for finger contact (no contact sensor wired). If the policy "lifts" by scooping/wedging without a real grasp, add a true contact sensor on `r_Link_finger1/2` (§1 cross-section edit) and gate `success` on it.
- `lifting_object` (w=15) dominates early — watch for the policy bumping the cube up momentarily to farm it without sustaining the lift; if so, gate it on contact or fold it into the height term.
- No reward curriculum (regularizers constant -1e-4, matching push). Add one only if jitter/energy becomes a problem late in training.
