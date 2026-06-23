# Iter 2 — Post-train analysis

trial_dir: harbor/outputs/ppo_Isaac-RM-Push-Block-v0_20260618-074028
train: 20M steps @ 2048 envs, 275s.
Change vs iter 1: added mid_band tracking (std=0.12, w=8); fine std 0.05→0.08. (§6 only.)

## Numerical
| key | iter 1 | iter 2 |
|---|---|---|
| eval/success_rate | 0.098 | **0.100** (flat) |
| reward/total | 67.50 | 84.70 |
| reaching_block (w=1.0) | 0.027 | 0.003 |
| coarse tracking (w=16, std=0.3) | 65.37 | 65.05 |
| mid_band (w=8, std=0.12) | — | 14.62 |
| fine (w=5, std=0.08) | 2.12 | 5.05 |

Implied avg block→goal distance (from each kernel): coarse → ~12 cm, mid_band → ~11 cm,
fine → ~10 cm. **All three agree: the block still plateaus at ~10–11 cm from the goal**,
unchanged from iter 1. The added gradient bands raised total reward (the same ballistic
push now earns more) but did NOT move the converged block position or success rate.

## Conclusion — the bottleneck is CONTROL, not reward gradient
reaching_block collapsed to ~0 (EE ends ~31 cm from block). The policy learned an **open-loop
"swat"**: one shove, then the EE retracts. The block coasts and stops ~10 cm from the goal via
friction. No closed-loop fine pushing happens because the arm abandons contact after the shove.
More reward gradient in the 5–15 cm band cannot fix this — there is no agent in contact to
respond to it. To get the block within 5 cm the arm must **stay in contact and push
incrementally**. This is the classic push "last-cm needs sustained contact" failure.

## Visual (17 frames @ stride 12)
Yellow DexCube clearly on the table; goal axis-triad on the table surface beside it (geometry
correct). The arm descends, contacts the block, shoves it toward the goal region, then the EE
drifts away while the block coasts to a stop short of the goal. Motion is consistent with a
single ballistic push rather than continuous contact pushing.

## Verdict — technically improved (best so far) but success effectively flat at 0.10
Reward-gradient shaping is exhausted; further kernel tweaks won't help. Iter 3 must change the
CONTROL incentive so the arm maintains EE-block contact for closed-loop pushing. Candidate
levers (pick the minimal that targets contact):
  (a) raise reaching weight substantially (1.0 → ~3–5) so the EE stays on the block, OR
  (b) contact-gate the fine/mid precision reward (only credit fine tracking when EE is within
      ~5–8 cm of the block) so the policy cannot farm it with a single ballistic shove and must
      stay engaged as the block approaches — keep the coarse attractor UNGATED for long-range
      shove initiation.
Hold the env-origin goal frame, XY success, coarse w=16. The ee_z_to_block term may finally be
warranted if contact is being lost vertically (EE rising off the block) — let the generator judge.
