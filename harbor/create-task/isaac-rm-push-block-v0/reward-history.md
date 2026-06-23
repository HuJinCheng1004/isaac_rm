# Reward-tune history — Isaac-RM-Push-Block-v0

Task: RM robot right arm (6-DOF, fixed chassis) pushes a DexCube block on a table
to a commanded 2D goal position. Success when block-to-goal xy distance < 5 cm; horizon 200 steps.

- Algorithm: ppo (skrl_local), default num_envs=2048
- Success threshold (eval/success_rate): 0.5
- Timesteps / iter: 20,000,000
- Library base: push-block-franka-IsaacLab.md (adapt-first)

Shared log — one `## Iter <N>` section appended each iteration. Latest reward state
lives in `handoff-reward-generator.md` (overwritten each iter).

---

## Iter 0

**Started:** 2026-06-18T05:03:00Z
**Prior reward state:** real (already-authored RewardsCfg, near-identical to library base)
**Reward path:** `source/chassis_nav/chassis_nav/tasks/push/mdp/rewards.py`
**Env cfg path:** `source/chassis_nav/chassis_nav/tasks/push/push_env_cfg.py` (RewardsCfg)

### Adaptation delta

**Base spec:** `experiences/task-library/manipulation/single-arm-manipulation/push-block-franka-IsaacLab.md` §6 (proven, adapt-first).
Secondary reference: `reach-franka-IsaacLab.md` (dual-bandwidth tanh tracking pattern).

**Kept as-is from the proven base (term ladder, weights, kernels, composer):**
- `reaching_block`: `1 - tanh(d_ee_block / std)`, std=0.1, weight=1.0.
- `block_to_goal_tracking`: `1 - tanh(d_block_goal / std)`, std=0.3 (coarse), weight=16.0.
- `block_to_goal_tracking_fine_grained`: same kernel, std=0.05 (sharp), weight=5.0.
- `action_rate` = `action_rate_l2`, weight=-1e-4.
- `joint_vel` = `joint_vel_l2`, weight=-1e-4.
- Composer = SUM; no staging gates (planar single-stage push); dual-bandwidth tracking attractor.
- The proven base's RewardManager-native dt-scaling convention is kept (NO dt-cancel loop installed),
  because the base's weights are calibrated for it and the base is proven. The magnitude-budget
  docstring reads in pre-dt-scale (nominal) units, matching the base.

**Changes from the base (enumerated):**

1. **EE body name `panda_hand` → `r_link8`** (geometry-forced). The base targets the Franka
   `panda_hand`; this repo's RM arm uses `r_link8` as its palm/EE body (`EE_BODY_NAME` in
   `robots/arm.py`). Applied to `reaching_block.params.robot_cfg.body_names`. Already present in
   the repo cfg before this iter; retained.

2. **`joint_vel` asset_cfg filtered to the 6 active arm joints** (mechanics-forced). The base
   penalizes all robot joints; this repo's chassis is fixed and only `r_joint1..r_joint6` are
   actuated, so `_ARM_CFG = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)` scopes the
   penalty to the active DOFs. Already present in the repo cfg; retained.

3. **Goal command ranges + object/table poses** differ from the base (geometry-forced): RM arm
   reaches to -x side (home EE ≈ (-0.615,-0.224,0.879)), so block/table/goal live on -x. These are
   §1/§4 facts, not §6; untouched this iter.

4. **`success` weight `0.0` → `1e-6`** (MECHANICAL DEVIATION, the only §6 value I changed this iter).
   The base uses weight=0.0 (purely logging via `info["detailed_reward"]`). In THIS repo,
   `block_at_goal` carries a load-bearing side-effect — it sets `env._reach_success`, which harbor
   eval reads for `eval/success_rate`. IsaacLab's `RewardManager` SKIPS a term's `func` entirely when
   `weight == 0.0` (reward_manager.py:146), so at the base's 0.0 the side-effect would never fire and
   success_rate would always read stale/unset. A negligible positive weight (1e-6 → ~3e-8/step after
   dt-scale) forces the func to execute so the side-effect fires, while contributing no meaningful
   optimization signal — preserving the proven "effectively logging-only" intent. This is forced by a
   repo-vs-library mechanical difference (the base's `block_at_goal` has no side-effect), NOT a design
   choice. Verified in smoke: `env._reach_success` shape (16,), binary, present.

5. **`ee_z_to_block` left UNWIRED** (matches base). The function is defined in `rewards.py` (exp
   z-alignment, std=0.5, documenting the "hover failure" mode) but the proven base uses no
   z-alignment / contact term, so per adapt-first it stays out of the iter-0 base. Candidate fix held
   in reserve if a future iter diagnoses the hover failure (EE pushes straight down → no lateral
   block motion) from training evidence.

No de-novo budget re-planning: the base is proven, so its weights/budget are adopted verbatim
(per adapt-first; reward-experience #2 budget-planning applies only to pure-creation mode).

### Decisions resolved

| Decision | Value | Source |
|---|---|---|
| primary term | block_to_goal_tracking (w=16) + fine_grained (w=5) | proven base §6 |
| reaching prior | reaching_block (w=1, std=0.1) | proven base §6 |
| regularizers | action_rate (-1e-4), joint_vel (-1e-4, arm joints only) | proven base §6 + repo DOF scope |
| success weight | 1e-6 (effectively logging-only; forces side-effect) | mechanical deviation from base 0.0 |
| z-alignment term | not wired | proven base §6 (no z term) |
| composer | sum | proven base §6 |
| dt-scaling | RewardManager-native (no cancel loop) | proven base convention |

### Files modified

| Path | Action | Notes |
|---|---|---|
| `source/chassis_nav/chassis_nav/tasks/push/push_env_cfg.py` | edit | RewardsCfg.success weight 1.0 → 1e-6 + comment (only §6 change) |
| `harbor/create-task/isaac-rm-push-block-v0/smokes/smoke_s6.py` | write | S6 smoke (PLAY task, 16 envs) |
| `harbor/create-task/isaac-rm-push-block-v0/handoff-reward-generator.md` | write | latest reward state |

No edits to §1–§5 or §7. No edits to `rewards.py` (functions already correct).

### Smoke

```bash
.venv/bin/python harbor/create-task/isaac-rm-push-block-v0/smokes/smoke_s6.py
```

Smoke runs the PLAY task (16 envs), random actions, 30 steps. Verifies: total reward
finite per env, non-constant across time, per-term values finite + correctly signed
(reaching/tracking/success ≥ 0, action_rate/joint_vel ≤ 0), composer = sum
(`sum(_step_reward * dt) == reward`), and `block_at_goal` sets `env._reach_success`
(shape (16,), binary).

```
active reward terms: ['reaching_block', 'block_to_goal_tracking', 'block_to_goal_tracking_fine_grained', 'success', 'action_rate', 'joint_vel']
composer max|sum(terms)-reward| = 2.328e-10  (sum-verified)
_reach_success: shape (16,), binary, present
S6 OK: 30 steps x 16 envs, reward mean=0.002 std=0.000 composer=sum-verified
```

(Isaac's `sim_app.close()` swallows trailing stdout; verdict mirrored to `/tmp/s6_verdict.txt`.
EXIT=0. The across-env mean rounds to std=0.000 at 3 dp but the `arr.std() > 0` assert passed —
random actions on the high-stiffness arm barely move it, so the near-saturated reaching reward
varies only slightly; the reward IS non-constant.)

**Verdict:** pass
**Iterations:** 3 (1: numpy-action build error fixed → tensor actions; 2: weight=0.0 skipped block_at_goal func → env._reach_success never set, raised to 1e-6; 3: pass)

**Finished:** 2026-06-18T05:08:30Z · status: pass

---

## Iter 1

**Started:** 2026-06-18T05:24:00Z
**Prior reward state:** iter_0 handoff (real RewardsCfg, proven library base, 0% success)
**Reward path:** `source/chassis_nav/chassis_nav/tasks/push/mdp/rewards.py`

### Delta vs iter 0 (why)

iter 0 trained to 0% success because of a GEOMETRY BUG, not a reward-shaping issue.
`block_to_goal_distance` and `block_at_goal` resolved the goal command against the robot
articulation root via `combine_frame_transforms(robot.data.root_pos_w, root_quat_w, des_pos_b)`.
But this fixed-chassis robot's articulation root sits at world **z=-0.805** (lift-pole base,
~0.8 m below the table). The command z (0.04, intended "4 cm above table") was added to the
root, so the goal landed at world **z≈-0.765 (~0.8 m UNDERGROUND)**, ~1.0 m from the block.
`block_at_goal` tested 3D distance < 5 cm → success was geometrically impossible → 0% forced.

**Fix (this iter):** resolve the goal in the **env-origin / table frame** instead of the
robot-root frame. In BOTH funcs, the root-frame transform is replaced by
`des_pos_w = env.scene.env_origins + des_pos_b` (identity rotation, z=0 origin). Per the task
spec ("block-to-goal XY distance < 5 cm"), `block_at_goal` now tests **XY distance only**
(z dropped); threshold kept at 0.05. The dense tracking terms keep full 3D distance — the
goal z is now correct (table height) so 3D vs xy is immaterial for them.

**Reward TERM LADDER is UNCHANGED** (proven library base, kept verbatim): reaching_block
(w=1.0, std=0.1), block_to_goal_tracking coarse (w=16, std=0.3), fine (w=5, std=0.05),
success (w=1e-6, logging-only side-effect), action_rate/joint_vel (-1e-4). No weight
rebalance. `ee_z_to_block` remains defined-but-unwired (held in reserve). This isolates
whether the geometry fix ALONE yields nonzero success on the next training run.

### Cross-section edits to §1–§5

None. The fix lives entirely in §6 (`mdp/rewards.py`). The §1–§5 cfg classes were inspected
under `permit_env_edits: true` but needed no change: `env.scene.env_origins` is already
available on the env, the command ranges (pos_x/pos_y ∈ (-0.3,-0.1)) and block reset region
already place reachable goals once the frame is corrected, and the obs terms
`object_position` (robot-root frame) + `target_object_position` (raw env-frame command) stay
sensible for the policy. The §1 docstring's stale "桌面中心 (-0.5,-0.2)" note (actual block
init is env-relative (-0.2,-0.1)) was left untouched — cosmetic only, not load-bearing.

### Decisions resolved

| Decision | Value | Source |
|---|---|---|
| Goal world-frame resolution | `env.scene.env_origins + des_pos_b` (table frame) | iter-0 diagnosis + verified fix |
| Success distance dimensionality | XY only (z dropped) | task description ("XY distance < 5 cm") |
| Success threshold | 0.05 m (unchanged) | proven base / spec |
| Term ladder + weights | unchanged from iter-0 proven base | dispatch instruction (isolate geometry fix) |

### Files modified

| Path | Action | Notes |
|---|---|---|
| `source/chassis_nav/chassis_nav/tasks/push/mdp/rewards.py` | Edit | both goal funcs → env-origin frame; `block_at_goal` → xy distance; removed now-unused `combine_frame_transforms` import |

### Geometry probe (MANDATORY verification — reset+settle, num_envs=8)

```bash
.venv/bin/python harbor/create-task/isaac-rm-push-block-v0/iter_001/geom_probe.py
```

```
=== iter-1 GEOMETRY PROBE (num_envs=8, reset+2 zero steps) ===
env0: root_w=(4.350,-1.550,-0.805) block_w=(3.502,-1.323,0.028) goal_w=(3.540,-1.407,0.040) dist_xy=0.093 dist_3d=0.094
env1: root_w=(4.350,0.950,-0.805) block_w=(3.536,1.157,0.028) goal_w=(3.619,0.961,0.040) dist_xy=0.212 dist_3d=0.213
env2: root_w=(1.850,-1.550,-0.805) block_w=(1.014,-1.310,0.028) goal_w=(1.132,-1.434,0.040) dist_xy=0.171 dist_3d=0.172
env3: root_w=(1.850,0.950,-0.805) block_w=(1.015,1.165,0.028) goal_w=(0.990,1.003,0.040) dist_xy=0.164 dist_3d=0.165
env4: root_w=(-0.650,-1.550,-0.805) block_w=(-1.472,-1.392,0.028) goal_w=(-1.508,-1.457,0.040) dist_xy=0.074 dist_3d=0.075
env5: root_w=(-0.650,0.950,-0.805) block_w=(-1.420,1.111,0.028) goal_w=(-1.437,1.073,0.040) dist_xy=0.042 dist_3d=0.043
env6: root_w=(-3.150,-1.550,-0.805) block_w=(-3.924,-1.322,0.028) goal_w=(-3.867,-1.519,0.040) dist_xy=0.205 dist_3d=0.206
env7: root_w=(-3.150,0.950,-0.805) block_w=(-3.964,1.168,0.028) goal_w=(-3.944,0.988,0.040) dist_xy=0.182 dist_3d=0.182
SUMMARY: goal_w.z mean=0.040 (expect ~0.04, NOT negative); dist_xy mean=0.143 max=0.212 (expect <= ~0.25); root_w.z mean=-0.805
```

**Probe verdict: PASS.** goal_w.z = 0.040 for every env (table height, was z≈-0.765 underground
in iter 0). root_w.z = -0.805 confirms the underground chassis root (the bug cause). block→goal
xy distance 0.042–0.212 m (mean 0.143, max 0.212) — all within the reachable push region (≤0.25 m).
The goal is now reachable by pushing.

### Smoke

```bash
.venv/bin/python harbor/create-task/isaac-rm-push-block-v0/smokes/smoke_s6.py
```

```
active reward terms: ['reaching_block', 'block_to_goal_tracking', 'block_to_goal_tracking_fine_grained', 'success', 'action_rate', 'joint_vel']
composer max|sum(terms)-reward| = 5.960e-08  (sum-verified)
_reach_success: shape (16,), binary, present
S6 OK: 30 steps x 16 envs, reward mean=0.336 std=0.001 composer=sum-verified
```

**Verdict:** pass
**Iterations:** 1

**Finished:** 2026-06-18T05:26:00Z · status: pass

---

## Iter 2

**Started:** 2026-06-18T05:38:00Z
**Prior reward state:** iter_1 handoff (geometry fixed; dual-bandwidth tracking coarse w=16/std=0.3 + fine w=5/std=0.05; success XY-only 1e-6; regularizers -1e-4)
**Reward path:** `source/chassis_nav/chassis_nav/tasks/push/mdp/rewards.py`
**Env cfg path:** `source/chassis_nav/chassis_nav/tasks/push/push_env_cfg.py` (RewardsCfg, lines ~161-216)

### Delta vs iter 1 — bridge the 5-15 cm gradient dead-band

iter-1 (20M steps): success_rate=0.098. Block reaches ~12 cm avg from goal (coarse
tracking 65.4) then plateaus at ~10-12 cm. DIAGNOSIS (iter_001/analysis.md): a gradient
dead-band in 5-15 cm — the coarse kernel (std=0.3) is nearly flat there and the fine
kernel (std=0.05) is too sharp to pull until <5 cm. No strong gradient to close the last
centimetres.

**Minimal, isolated change (two edits, both kernel-band only):**

1. **ADDED `block_to_goal_tracking_mid_band`** — `func=mdp.block_to_goal_distance`,
   `std=0.12`, `weight=8.0`. A std=0.12 tanh has its steepest slope exactly across
   3-15 cm: `1-tanh(0.12/0.12)=0.24`, `1-tanh(0.05/0.12)=0.40` → a strong pulling
   gradient through the plateau. Weight 8 sits between coarse (16) and fine (5): real
   pull without overwhelming the proven coarse attractor. Added ON TOP of the coarse
   term (reward-experience #9: never average a working attractor — `coarse + mid + fine`).

2. **WIDENED fine std 0.05 → 0.08** (weight unchanged 5.0). At std=0.05 the fine kernel
   gives `1-tanh(0.10/0.05)=0.02` at 10 cm — effectively zero pull until <5 cm. At
   std=0.08 it gives `1-tanh(0.10/0.08)=0.16` at 10 cm, so it engages from ~8-10 cm and
   takes the hand-off from the mid-band earlier.

NOT changed (per dispatch): coarse w=16 / std=0.3 (proven), reaching w=1.0, success
XY-only threshold=0.05 weight=1e-6 (logging-only side-effect), action_rate/joint_vel
-1e-4, env-origin goal frame, composer=sum. No `ee_z_to_block` z-alignment term added
(reaching collapse is expected for push; no hover problem observed). dt-scaling left in
place (proven base + iter-1 budget were computed pre-dt-scale; cancelling now would
perturb every term and break the one-change isolation).

### Final term ladder (per-step saturated maxima, pre-dt-scale)

| name | weight | shape | std/params | sat. max | gate |
|---|---:|---|---|---:|---|
| reaching_block | 1.0 | `1 - tanh(d_ee_block/std)` | std=0.1, body=`r_link8` | 1.0 | none |
| block_to_goal_tracking | 16.0 | `1 - tanh(d_block_goal/std)` | std=0.3 (coarse), env-origin goal | 16.0 | none |
| block_to_goal_tracking_mid_band | 8.0 | `1 - tanh(d_block_goal/std)` | std=0.12 (NEW), env-origin goal | 8.0 | none |
| block_to_goal_tracking_fine_grained | 5.0 | `1 - tanh(d_block_goal/std)` | std=0.08 (was 0.05), env-origin goal | 5.0 | none |
| success | 1e-6 | `1.0 if d_xy < thr` | thr=0.05 XY; sets `env._reach_success` | ~0 | none (logging) |
| action_rate | -1e-4 | `-‖a_t-a_{t-1}‖²` | library `action_rate_l2` | <0 | none |
| joint_vel | -1e-4 | `-‖q̇‖²` | library `joint_vel_l2`, arm joints | <0 | none |

Saturated max per-step ≈ 30.0 (pre-dt) when block sits at goal (16+8+5+1). The three
tracking bands now monotonically hand off: coarse (long-range 0.3) → mid (0.12, peak
gradient 3-15 cm) → fine (0.08, last cm). reward-experience #9 honoured (added on top,
no proven gradient reduced).

### Files modified

| Path | Action | Notes |
|---|---|---|
| `source/chassis_nav/chassis_nav/tasks/push/push_env_cfg.py` | edit | added `block_to_goal_tracking_mid_band` (std=0.12, w=8.0); fine std 0.05→0.08 |
| `harbor/create-task/isaac-rm-push-block-v0/smokes/smoke_s6.py` | edit | added mid_band to EXPECTED_SIGN (+1) |

### Smoke

```bash
.venv/bin/python harbor/create-task/isaac-rm-push-block-v0/smokes/smoke_s6.py
```

```
Active Reward Terms: 7 (reaching_block 1.0, block_to_goal_tracking 16.0,
  block_to_goal_tracking_mid_band 8.0, block_to_goal_tracking_fine_grained 5.0,
  success 1e-06, action_rate -1e-4, joint_vel -1e-4)
active reward terms: ['reaching_block', 'block_to_goal_tracking',
  'block_to_goal_tracking_mid_band', 'block_to_goal_tracking_fine_grained',
  'success', 'action_rate', 'joint_vel']
composer max|sum(terms)-reward| = 1.192e-07  (sum-verified)
_reach_success: shape (16,), binary, present
S6 OK: 30 steps x 16 envs, reward mean=0.423 std=0.002 composer=sum-verified
```

**Verdict:** pass
**Iterations:** 1 (no retry needed)

**Finished:** 2026-06-18T05:39:00Z · status: pass

### Cross-section edits in this iter

None. Change lives entirely in §6 (`RewardsCfg`); reuses the existing
`block_to_goal_distance` function unchanged. No new cfg fields needed.

---

## Iter 3

**Started:** 2026-06-18T05:50:00Z
**Prior reward state:** real (iter-2 handoff — coarse + ungated mid_band + ungated fine + reaching w=1.0)
**Reward path:** `source/chassis_nav/chassis_nav/tasks/push/mdp/rewards.py`
**Env cfg path:** `source/chassis_nav/chassis_nav/tasks/push/push_env_cfg.py` (RewardsCfg)

### Diagnosis driving this iter (CONTROL, not reward gradient)

iters 1-2 both plateaued at success ~0.10 with the block stalling ~10-11 cm from the goal
regardless of tracking-kernel bands. iter-2 analysis (root_cause memory) is conclusive:
**reaching_block collapsed to ~0 (EE ends ~31 cm from block)** — the policy learned an
**open-loop "swat"**: one ballistic shove, then the EE retracts while the block coasts to a
stop ~10 cm short. The ungated mid/fine precision tracking was farmable by that single shove
(once the block drifts goal-ward, the policy keeps banking precision reward with the arm
already withdrawn). Reward-gradient shaping is exhausted; the fix must make CLOSED-LOOP
sustained-contact pushing strictly beat the swat. This iter applies the brief's preferred
(a)+(b) combo. (c) ee_z_to_block NOT wired — iter-2 visual shows lateral contact, not a
vertical hover/lift-off, so the contact is lost laterally (arm retracts), which (a)+(b) target
directly; vertical alignment is not the failure mode.

### Delta vs iter 2

**(a) Raise reaching weight 1.0 → 4.0.** Reaching was dwarfed by the tracking total
(16+8+5=29), so abandoning the block after a shove cost almost nothing. At w=4.0, sustained
EE-block proximity is worth keeping each step, yet stays well below the coarse attractor (16)
so it does not dominate the goal signal. Kernel unchanged (`1-tanh(d_ee_block/0.1)`).

**(b) Contact-gate the precision tracking (mid_band + fine).** New helper
`block_to_goal_distance_contact_gated` multiplies the tracking kernel by a soft contact gate
`g = 1 - tanh(d_ee_block / std_gate)`, std_gate=0.05:
  - g(0)=1.00, g(0.05cm)=0.24, g(0.10)=0.04, g(0.31 — iter-2 abandon dist) ≈ 0.
So the policy can collect the mid/fine precision pull ONLY while the EE is touching the block;
abandoning it after a shove zeroes both terms → the swat exploit no longer pays the precision
reward, and riding the block in (closed-loop) is the only way to bank it. std (0.12 mid / 0.08
fine) and weights (8 / 5) are UNCHANGED from iter 2 — only the contact gate is added.

**Kept (per brief):** coarse `block_to_goal_tracking` w=16/std=0.3 **UNGATED** (long-range shove
initiation keeps its gradient — reward-experience #9), env-origin goal frame, XY success
(thr=0.05, w=1e-6, sets `env._reach_success`), action_rate/joint_vel = -1e-4, composer=sum.
dt-scaling convention left as the proven base / iters 1-2 use it (RewardManager multiplies
weight×dt per step; no cancellation loop introduced — that would re-scale the whole proven
ladder and invalidate the iter-1/2 tuning).

### New mdp/rewards.py functions

- `block_to_goal_distance_contact_gated(env, std, command_name, std_gate=0.05, robot_cfg=..., object_cfg=...)`
  — env-origin-frame tracking `1-tanh(d_block_goal/std)` × contact gate `1-tanh(d_ee_block/std_gate)`.
  Re-exported in `tasks/push/mdp/__init__.py`. NOTE: `robot_cfg` is passed EXPLICITLY in the cfg
  params (not left as default) because RewardManager only resolves SceneEntityCfg body_ids for
  params it is handed — an unresolved default `body_names` slice breaks `body_pos_w` indexing.

### Files modified

| Path | Action | Notes |
|---|---|---|
| `source/chassis_nav/chassis_nav/tasks/push/mdp/rewards.py` | edit | added `block_to_goal_distance_contact_gated` |
| `source/chassis_nav/chassis_nav/tasks/push/mdp/__init__.py` | edit | re-export new func |
| `source/chassis_nav/chassis_nav/tasks/push/push_env_cfg.py` | edit | reaching 1.0→4.0; mid_band + fine switched to gated func (explicit robot_cfg) |
| `harbor/create-task/isaac-rm-push-block-v0/smokes/smoke_s6.py` | edit | gate-ratio assertion (open on contact, closed when far) |

### Final term ladder

| name | weight | shape | std / gate |
|---|---:|---|---|
| reaching_block | 4.0 | `1-tanh(d_ee_block/std)` | std=0.1; **w 1.0→4.0** |
| block_to_goal_tracking | 16.0 | `1-tanh(d_block_goal/std)` | std=0.3, UNGATED |
| block_to_goal_tracking_mid_band | 8.0 | tracking × contact-gate | std=0.12, **gate std=0.05 (NEW)** |
| block_to_goal_tracking_fine_grained | 5.0 | tracking × contact-gate | std=0.08, **gate std=0.05 (NEW)** |
| success | 1e-6 | `1.0 if d_xy<thr` | thr=0.05, XY, sets `env._reach_success` |
| action_rate | -1e-4 | `-‖a_t-a_{t-1}‖²` | — |
| joint_vel | -1e-4 | `-‖q̇‖²` | arm joints |

### Cross-section edits (§1–§5)

None. permit_env_edits=true was granted, but the contact gate reuses the existing EE body
(`r_link8`) and object — no new sensor, obs term, or cfg field was required. The whole change
lives in §6.

### Smoke

```bash
.venv/bin/python harbor/create-task/isaac-rm-push-block-v0/smokes/smoke_s6.py
```

```
active reward terms: ['reaching_block', 'block_to_goal_tracking', 'block_to_goal_tracking_mid_band', 'block_to_goal_tracking_fine_grained', 'success', 'action_rate', 'joint_vel']
composer max|sum(terms)-reward| = 2.980e-08  (sum-verified)
_reach_success: shape (16,), binary, present
contact-gate ratio: far(mid=0.0000,fine=0.0000) ~0 (gate closed) ; near(mid=0.9997,fine=0.9910) ~1 (gate open)
S6 OK: 30 steps x 16 envs, reward mean=0.322 std=0.001 composer=sum-verified
```

**Verdict:** pass
**Iterations:** 3 (1: body_ids slice unresolved → pass robot_cfg explicitly; 2: gate test conflated tracking magnitude → isolate via gate ratio + no-physics-step teleport; 3: pass)

**Finished:** 2026-06-18T05:55:38Z · status: pass

---

## Final summary — CONVERGED (best = iter 3, success_rate 0.507)

**Progression**
| iter | change | success_rate | note |
|---|---|---|---|
| 0 | proven library base ladder (push-block-franka), success XY w=1e-6 | 0.000 | goal computed in robot-root frame (z=-0.805) → goal ~0.8 m underground; success geometrically impossible |
| 1 | goal frame robot-root → **env-origin**; block_at_goal XY-only | 0.098 | geometry fixed; block now reaches ~12 cm from goal |
| 2 | + mid_band tracking (std=0.12,w=8); fine std 0.05→0.08 | 0.100 | flat — reward-gradient shaping exhausted; block still plateaus ~10 cm. Diagnosed open-loop swat (reaching collapsed to ~0) |
| 3 | reaching w 1→4; **contact-gate** mid/fine tracking | **0.507** | sustained-contact pushing; coarse 65→85 (block ~6 cm); converged |

**What worked**
1. **Goal-frame geometry fix (iter 1)** — the single biggest unlock. The goal must be resolved in
   the env-origin/table frame, NOT this robot's articulation root (sunk to z=-0.805). Without it,
   success was impossible regardless of reward.
2. **Contact incentive (iter 3)** — raising reaching weight (1→4) + contact-gating the precision
   tracking terms (× (1-tanh(d_ee_block/0.05))) so they can't be farmed by a ballistic shove. This
   converted an open-loop "swat-and-retract" policy into closed-loop contact pushing and broke the
   0.10 plateau to 0.51.

**What didn't move the needle**
- Adding reward-gradient bands (iter 2 mid_band + wider fine) raised total reward but NOT success —
  pure kernel shaping cannot fix a control problem (no agent in contact to respond to the gradient).

**Final reward state** — see handoff-reward-generator.md. Ladder: reaching_block (w=4.0) ·
block_to_goal_tracking coarse (w=16, std=0.3, ungated) · mid_band (w=8, std=0.12, contact-gated) ·
fine (w=5, std=0.08, contact-gated) · success (w=1e-6, XY thr=0.05, logging-only) ·
action_rate/joint_vel (-1e-4). composer = sum. Goal frame = env-origin.

**Outstanding / recommended next direction (optional — threshold met)**
- success_rate sits right at 0.50; the contact-gated mid/fine terms are currently near-0 because
  sustained contact is partial. To reach 0.7+: widen std_gate to ~0.08–0.10 so the precision terms
  fire during genuine contact and supply last-cm pull, and/or train beyond 20M steps.
