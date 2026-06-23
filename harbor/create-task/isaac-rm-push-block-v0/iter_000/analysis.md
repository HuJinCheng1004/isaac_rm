# Iter 0 — Post-train analysis

trial_dir: harbor/outputs/ppo_Isaac-RM-Push-Block-v0_20260618-071012
train: 20M steps @ 2048 envs, 401s. W&B run ikgayb0b.

## Numerical (metrics.jsonl, final)
| key | value |
|---|---|
| eval/success_rate | **0.000** |
| eval/episodes | 1364 |
| reward/total/episodic_return_mean | 3.606 |
| reward/reaching_block | 3.290 |
| reward/block_to_goal_tracking (coarse, w=16) | **0.321** |
| reward/block_to_goal_tracking_fine_grained (w=5) | **0.000** |
| reward/success | 0.000 |
| reward/action_rate | -0.0002 |
| reward/joint_vel | -0.0051 |

Interpretation: reaching is ~half-saturated (EE avg ~5–6 cm from block, sustained → block
is NOT knocked away). But coarse tracking ≈ 0 and fine-grained == 0 → the block stays
~1 m from the goal the entire episode. The policy keeps the EE near the block but never
moves the block toward the goal.

## Geometry diagnostic (THE root cause — task is geometrically unsolvable)
Ran a reset+settle probe (num_envs=8). World-frame positions (env 0):
- robot articulation **root_w z = -0.805** (root anchored ~0.8 m below ground — fixed
  chassis root sits at the lift-pole base, far below the table surface)
- block_w z = 0.028 (on table, correct)
- **goal_w z = -0.765** (UNDERGROUND, ~0.79 m below the block)
- dist_block_goal ≈ **1.0 m for every env** — almost entirely the bogus vertical offset.

Cause: `block_to_goal_distance` / `block_at_goal` (mdp/rewards.py) build the goal via
`combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)`.
The command z (0.04) is interpreted as "4 cm above the robot root", but the root is at
z=-0.805, so the goal lands at z=-0.765 — underground. block_at_goal uses a 3D-distance
< 5 cm test, so **success can never fire** regardless of policy or reward shaping. The
0% success rate is forced by geometry, not by the reward design.

The XY offset is also off: goal_w xy ≈ (4.21,-1.36) vs block (3.59,-1.39) → ~0.6 m in
+x (toward the robot), because root_w xy=(4.35,-1.55) ≠ env_origin (3.75,-1.25).

## Verified fix direction
If the goal is computed in the **env-origin frame** (z=0, identity rotation) instead of
the robot root frame, i.e. `goal_w = env.scene.env_origins + des_pos_b`:
- env 0: env_origin (3.75,-1.25,0) + cmd (-0.188,-0.143,0.04) = (3.562,-1.393,0.04)
- block (3.585,-1.392,0.028) → **dist ≈ 2.6 cm** — on the block, at table height.
- The cmd ranges pos_x/pos_y = (-0.3,-0.1) (env-relative) put goals 0–0.2 m from the
  block's reset region (±5 cm around (-0.2,-0.1)) → reachable by pushing.

## Visual (12 frames @ stride 18)
First frame black (camera warmup). Remaining frames show the fixed-chassis robot, the
SeattleLabTable (black box), the goal-pose axis triad, and the arm hovering near the
table top. The DexCube is small/low-res; motion is consistent with "EE hovers near block,
block barely moves" — matching the numbers. No block-flies-off behavior observed.

## Verdict
Reward TERM LADDER is sound (kept from proven library base). The blocker is a §1/§4
**goal-frame geometry bug**: the goal command is resolved in the underground robot-root
frame. Iter 1 MUST fix the goal frame (use env-origin / table frame so the goal sits at
table height in the block's reachable region) BEFORE any further reward shaping. Secondary:
make success xy-only per the task description ("block-to-goal xy distance < 5 cm").
