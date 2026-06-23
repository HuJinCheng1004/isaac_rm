# Iter 1 — Post-train analysis

trial_dir: harbor/outputs/ppo_Isaac-RM-Push-Block-v0_20260618-072844
train: 20M steps @ 2048 envs, 308s.
Change vs iter 0: goal frame fixed (robot-root → env-origin); block_at_goal now XY-only.
Reward term ladder UNCHANGED.

## Numerical (metrics.jsonl, final)
| key | iter 0 | iter 1 |
|---|---|---|
| eval/success_rate | 0.000 | **0.098** |
| reward/total | 3.61 | 67.50 |
| reaching_block (w=1.0) | 3.29 | 0.027 |
| block_to_goal_tracking coarse (w=16, std=0.3) | 0.32 | **65.37** |
| block_to_goal_tracking_fine (w=5, std=0.05) | 0.0 | 2.12 |
| success | 0 | ~0 |

## Interpretation
The geometry fix worked: the block now moves to the goal (coarse tracking 0.32→65.4).
- Implied avg block→goal distance ≈ **0.12 m** (coarse: 65.4/106.7=0.61 → 1-tanh(d/0.3)=0.61 → d≈0.12).
- fine-grained (std=0.05) returns only 2.12/33.3=6.4% → block rarely within ~5–8 cm.
- reaching_block collapsed (3.29→0.027): EE avg ~31 cm from block. The policy gives the
  block a shove then leaves — correct for a push task; reaching was only a contact primer.

**Bottleneck = the last centimeters.** The block plateaus at ~10–12 cm from the goal and
the policy can't close it: the coarse kernel (std=0.3) is nearly flat there and the fine
kernel (std=0.05) is so sharp it gives almost no gradient until <5 cm. There is a gradient
"dead band" in the 5–15 cm range exactly where the block stalls. Only 9.8% of episodes
nail XY<5 cm.

## Visual (12 frames @ stride 18)
Yellow DexCube visible on the table top; goal axis-triad rendered on the table surface near
the block (goal now at table height — confirms the geometry fix visually). The arm approaches
and nudges the block toward the goal, but fine positioning is imprecise; the block ends up
near but not on the goal. No knock-off / drop behavior.

## Verdict — IMPROVED (success 0 → 0.098), best so far
Geometry fix validated. Next bottleneck is fine positioning. Iter 2 should add a **mid-band**
tracking gradient to bridge coarse (std=0.3) and fine (std=0.05) — e.g. a third tanh term
with std≈0.1–0.15 (and/or widen the fine std), so the block gets a pulling gradient through
the 5–15 cm plateau. Keep the proven coarse/reaching structure. Consider whether sustained
EE-block contact (reaching currently abandoned) is needed for fine micro-adjustments — but
prefer the minimal kernel-band change first to isolate its effect.
