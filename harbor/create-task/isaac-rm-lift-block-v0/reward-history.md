# Isaac-RM-Lift-Block-v0 — Reward Tune History

Task: RM robot arm grasps a 5cm cube on the table and lifts it to 15cm above the table surface. Gripper closes to grasp. Success when the cube is lifted above 0.15m AND held (finger contact). Horizon 200 steps. Use absolute joint-position control plus binary gripper.

Algorithm: ppo | num_envs: 4096 | timesteps_per_iter: 20_000_000 | success_threshold: 0.5

| Field | Value |
|---|---|
| task_id | Isaac-RM-Lift-Block-v0 |
| benchmark_family | isaaclab-manager-based (chassis_nav) |
| started_at | 2026-06-22 |

---

## Iter 0

**Started:** 2026-06-22
**Prior reward state:** placeholder (constant-zero `zero_reward`, weight 0.0)
**Reward path:** `source/chassis_nav/chassis_nav/tasks/lift/mdp/rewards.py`

### Adaptation delta

**Base spec:** `experiences/task-library/manipulation/single-arm-manipulation/lift-cube-franka-IsaacLab.md` §6 (proven trained Franka cube-lift). Repo idioms from `push-block-isaac_rm.md` / `source/.../tasks/push/`.

**Kept as-is from the Franka base:**
- Composer = sum; RewardManager dt scaling LEFT IN PLACE (matches the proven push/Franka repo convention; the success weight 1e-6 is calibrated for it — weight==0 is skipped by the manager, a tiny positive weight survives so the side-effect runs).
- Term ladder reach -> lift indicator -> (height/contact) with strictly increasing magnitude budget reach(1) -> lift(15) -> height(16).
- `reaching_block` w=1.0 (tanh std=0.1), `lifting_object` w=15.0 (minimal_height=0.04), regularizers `action_rate` w=-1e-4 / `joint_vel` w=-1e-4.

**Changed (each forced by this task's geometry / lack of a command):**
- This task has NO CommandsCfg / `object_pose` command -> the Franka base's dynamic 3D goal-tracking pair (`object_goal_tracking` w=16, `object_goal_tracking_fine_grained` w=5) is replaced by:
  - `cube_height` (w=16.0): tanh pull of the cube's WORLD z toward a hard-coded 0.20 m, gated on lifted. Computes height directly — never calls `env.command_manager` (would crash).
  - `grasping` (w=5.0): soft EE->cube contact gate `1-tanh(d/std_gate)`, gated on lifted — replaces the fine-grained tracking band with a "stay in contact during lift" signal.
- EE position read via `robot.data.body_pos_w[:, robot_cfg.body_ids[0]]` instead of FrameTransformer `ee_frame` (this robot has no FrameTransformer wired) — mirrors the push task's `object_ee_distance_body`. `robot_cfg` with `body_names=[EE_BODY_NAME]` is passed EXPLICITLY in every term's `params` so RewardManager resolves `body_ids`.
- Added `success` term (`block_lifted_and_grasped`, w=1e-6): cube z > 0.15 m AND EE within 0.08 m (finger-contact proxy; no contact sensor wired). Sets `env._reach_success` for harbor eval — this is the §4 success metric, not present in the Franka training cfg.
- Dropped the Franka base's CurriculumCfg (ramping penalty weights at 10k steps) — kept regularizers at constant -1e-4, matching the sibling push task which has no reward curriculum.

### Decisions resolved

| Decision | Value | Source |
|---|---|---|
| Composer | sum | Franka base + IsaacLab family default |
| dt scaling | left in place | push/Franka repo convention (proven), success weight calibrated for it |
| Lift target height | world z 0.20 m (success line 0.15 m) | description ("15cm above table") + height-kernel gradient at success |
| reaching std | 0.1 | Franka base + push task |
| lifting minimal_height | 0.04 m | Franka base |
| grasping contact std_gate | 0.05 | push task contact gate |
| success contact_threshold | 0.08 m | task brief (finger-contact proxy) |
| success weight | 1e-6 | push task (side-effect must run; weight!=0) |

### Files modified

| Path | Action | Notes |
|---|---|---|
| `source/chassis_nav/chassis_nav/tasks/lift/mdp/rewards.py` | rewrite | placeholder -> 5 reward funcs (kept `zero_reward`) |
| `source/chassis_nav/chassis_nav/tasks/lift/mdp/__init__.py` | edit | export new reward funcs |
| `source/chassis_nav/chassis_nav/tasks/lift/lift_env_cfg.py` | edit | `RewardsCfg` placeholder -> real 7-term wiring |
| `harbor/create-task/isaac-rm-lift-block-v0/smokes/smoke_s6.py` | create | rendered from template + repo-path/import/torch-action/RewardManager-decompose fixes |

No §1–§5 cross-section edits (the EE body_id, cube world pose, and arm-joint cfg the reward needs were already present in §1/§5). §7 DR untouched.

### Smoke

```bash
cd /home/shihao/isaac_rm
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python -u harbor/create-task/isaac-rm-lift-block-v0/smokes/smoke_s6.py
```

```
[INFO] Reward Manager:  <RewardManager> contains 7 active terms.
 reaching_block 1.0 | lifting_object 15.0 | cube_height 16.0 | grasping 5.0
 success 1e-06 | action_rate -1e-4 | joint_vel -1e-4
per-term episodic mean (summed over 30 steps, dt-scaled, mean across envs):
  lifting_object: +0.765625
  cube_height: +0.267370
  reaching_block: +0.021113
  joint_vel: -0.005367
  action_rate: -0.001372
  grasping: +0.000892
  success: +0.000000
S6 OK: 30 steps, reward mean=0.035 std=0.107 composer=sum-verified reach_success=set
```

**Verdict:** pass
**Iterations:** 1 build attempt + 2 smoke-script fixes (numpy->torch action; `_step_reward` is value/dt so multiply by step_dt) — reward code itself unchanged.

**Finished:** 2026-06-22 · status: pass

---

---

## Iter 1

**Started:** 2026-06-22
**Prior reward state:** iter-0 reward (7 terms: reaching/lifting/height/grasping/success + action_rate/joint_vel)

### Changes from iter 0 (delta for new description requirements)

**New reward requirements from user description:**
1. Motion smoothness: penalize high joint velocity AND high jerk (sudden velocity changes)
2. Table avoidance: soft penalty for arm links contacting the table surface

**Adaptation delta (added terms):**

| Term | Change | Rationale |
|------|--------|-----------|
| `joint_vel` weight | -1e-4 → -5e-4 | Increase 5× to encourage slow compliant motion (description emphasis) |
| `joint_jerk` (NEW) | mean-abs(joint_acc), w=-5e-4 | Jerk proxy via PhysX joint_acc; mean-abs avoids L2-squared amplification; weight matches joint_vel for equal penalty scale |
| `table_avoidance` (NEW) | tanh gate on arm body z vs table height, w=-0.5 | Soft penalty for structural links (r_link1–r_link6) near table surface (z<0.02m); end-effector excluded (must approach cube) |

**Scene adjustment (acceptance gate):**
- Robot moved from X=0.3 to X=0.5 (+20cm away from table) — user request for chassis-table clearance
- Block adjusted from (-0.2,-0.1) to (-0.1,-0.1) (+10cm X) — compensates for longer reach
- Robot-to-table gap: 0.8m → 1.0m ✓

**S6 smoke (9 terms):**
```
  lifting_object: +0.871094
  cube_height: +0.232980
  reaching_block: +0.046303
  joint_vel: -0.029109
  joint_jerk: -0.021435
  grasping: +0.002346
  table_avoidance: -0.001357
  action_rate: -0.001355
  success: +0.000000
S6 OK: 30 steps, reward mean=0.037 std=0.101 composer=sum-verified reach_success=set
```
Verdict: **PASS**. jerk ≈ joint_vel in magnitude ✓, table_avoidance near-zero with arm above table ✓.

### Training

```bash
OMNI_KIT_ACCEPT_EULA=YES .venv/bin/python harbor/scripts/rl/skrl_local/train.py \
    task=Isaac-RM-Lift-Block-v0 seed=42 num_envs=2048 total_timesteps=20000000 \
    wandb=reward-tune-Isaac-RM-Lift-Block-v0
```
Log: `harbor/create-task/isaac-rm-lift-block-v0/iter_001/train.log`
WandB: `reward-tune-Isaac-RM-Lift-Block-v0`

**Status:** Running...

