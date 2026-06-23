# Iter 3 — Post-train analysis (CONVERGED)

trial_dir: harbor/outputs/ppo_Isaac-RM-Push-Block-v0_20260618-075725
train: 20M steps @ 2048 envs, 373s.
Change vs iter 2: reaching weight 1.0→4.0; mid_band + fine tracking CONTACT-GATED
(× (1-tanh(d_ee_block/0.05))). coarse tracking left ungated.

## Numerical
| key | iter 2 | iter 3 |
|---|---|---|
| eval/success_rate | 0.100 | **0.507 ✓ (≥0.5 threshold)** |
| reward/total | 84.70 | 85.94 |
| reaching_block (w=4.0) | 0.003 | 0.631 |
| coarse tracking (w=16, std=0.3) | 65.05 | **85.30** |
| mid_band (gated, w=8) | 14.62 | 0.014 |
| fine (gated, w=5) | 5.05 | 0.006 |

## Interpretation — convergence is genuine
- coarse tracking 65→85.3 → implied avg block→goal distance dropped from ~12 cm to **~6 cm**
  (85.3/106.7=0.80 → 1-tanh(d/0.3)=0.80 → d≈0.06 m). With success = XY<5 cm, ~50% of episodes
  now finish inside the threshold. Consistent: success_rate=0.507 over 1282 eval episodes.
- reaching 0.003→0.631 → the arm now stays engaged near the block (the w=1→4 bump made sustained
  proximity worth holding) instead of swatting and retracting.
- The gated mid/fine terms collapsed to ~0 — the policy can no longer farm precision reward with
  a ballistic shove (gate is ~0 unless the EE is touching the block). Removing that exploit, plus
  the stronger reaching incentive, redirected the policy toward keeping contact and pushing the
  block genuinely closer. The win came from the CONTACT incentive, exactly as hypothesized in iter 2.

## Visual (17 frames @ stride 12)
Yellow DexCube on the table; goal axis-triad on the table surface. The arm descends, makes and
KEEPS contact with the block, and pushes it onto the goal marker — the block ends co-located with
the goal triad. Genuine closed-loop pushing, not a swat. No knock-off / drop.

## Verdict — CONVERGED
success_rate 0.507 ≥ 0.5 threshold. The reward is validated end-to-end by training. Tune stops.

Progression: iter0 0.000 (goal underground — geometry bug) → iter1 0.098 (goal-frame fix) →
iter2 0.100 (kernel bands, flat — diagnosed control bottleneck) → iter3 0.507 (contact-gating +
reaching weight = sustained-contact pushing).

## Possible further gains (not required — threshold met)
- success sits right at 0.5; to push higher (e.g. 0.7+), the now-dead gated precision terms could
  be re-tuned (wider std_gate ~0.08–0.10 so they fire during real contact and supply last-cm pull),
  or reaching weight tuned, or train longer than 20M. Left as future work.
