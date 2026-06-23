# Task History — Isaac-RM-Lift-Block-v0

| field | value |
|-------|-------|
| task_id | Isaac-RM-Lift-Block-v0 |
| slug | isaac-rm-lift-block-v0 |
| mode | create |
| sections | 1, 2, 3, 4, 5 |
| started_at | 2026-06-22 |
| status | pass |
| finished_at | 2026-06-22 |

## Adaptation delta

- **Base (repo idioms):** `experiences/task-library/manipulation/single-arm-manipulation/push-block-isaac_rm.md`
  (in-repo `tasks/push/`) — robot cfg, URDF path, viewer, table/plane/light positions, event
  structure, obs noise, agents yaml, gym registration pattern.
- **Design pattern (lift):** `experiences/task-library/manipulation/single-arm-manipulation/lift-cube-franka-IsaacLab.md`
  — gripper as binary action term, EE-in-root-frame obs, height-based lift goal.

**Kept as-is from push base:**
- Robot articulation idioms (`ARM_GRASP_CFG` from `robots/arm.py`, fixed chassis, URDF gripper variant).
- Table pos `[-0.5, -0.2, 0.0]`, plane z=-1.05, DomeLight (0.75) intensity 3000.
- Object = DexCube `dex_cube_instanceable.usd` scale 0.8, init `[-0.2, -0.1, 0.055]`.
- `decimation=2`, `episode_length_s = 200/30`, `sim.dt = 1/60`, physx settings.
- `object_position_in_robot_root_frame` obs fn copied verbatim.
- Obs noise ±0.01 rad on joint_pos / joint_vel; `enable_corruption=True`, `concatenate_terms=True`.
- num_envs=1024, env_spacing=2.5.
- Reset: `reset_robot_joints` (arm offset ±0.3), `reset_object_position` (pose ±0.05).

**Changed (each with reason):**
- Robot cfg: `ARM_PUSH_CFG` -> `ARM_GRASP_CFG` (8 DOF: 6 arm + 2 prismatic fingers) — task requires
  grasping, gripper must be actuated.
- §2 actions: added `gripper_action` = `BinaryJointPositionActionCfg` on `GRIPPER_JOINT_NAMES`
  (open=0.0325, close=0.0) alongside the 6-DOF arm action -> action dim 7 (6 arm + 1 binary).
- §4 goal/termination: REMOVED `CommandsCfg` (push had a dynamic 2D goal). Lift goal is a fixed
  world height (0.15 m); no command to resample. Dropped `target_object_position` obs term.
- §5 obs: removed goal-command term; added `gripper_pos` (absolute joint_pos, 2) and `ee_position`
  (EE body in robot root frame, 3) so the agent can grasp + sense lift height. New obs dim = 27
  (6 arm pos + 6 arm vel + 2 gripper + 3 object + 3 ee + 7 action).
- §3 reset: extended `reset_robot_joints` `asset_cfg` to include both arm + gripper joints so the
  gripper is re-opened to its default each episode (`reset_joints_by_offset` re-centres on default).
  Kept ±0.3 offset on arm; gripper default = OPEN.
- ee_position obs: new task-local fn `ee_position_in_robot_root_frame` (uses `body_pos_w` +
  `subtract_frame_transforms`, mirrors the proven `object_position` fn). Chose body_pos_w over a
  FrameTransformer to keep obs dim deterministic and avoid extra scene wiring (per brief option).
- Viewer: eye=(1.5, 0.8, 1.2), lookat=(-0.2, -0.1, 0.3) — raised to frame the cube pickup/lift area.
- §6 reward: PLACEHOLDER constant-zero only (not authored). §7 DR: empty.
- agents yaml: action_size 6->7, obs comment 28->27, experiment dir rm_push_block->rm_lift_block.

## Authoring + files written

Files written (repo-relative):
- `source/chassis_nav/chassis_nav/tasks/lift/__init__.py`
- `source/chassis_nav/chassis_nav/tasks/lift/lift_env_cfg.py`
- `source/chassis_nav/chassis_nav/tasks/lift/agents/__init__.py`
- `source/chassis_nav/chassis_nav/tasks/lift/agents/skrl_ppo_cfg.yaml`
- `source/chassis_nav/chassis_nav/tasks/lift/mdp/__init__.py`
- `source/chassis_nav/chassis_nav/tasks/lift/mdp/observations.py`
- `source/chassis_nav/chassis_nav/tasks/lift/mdp/rewards.py` (PLACEHOLDER zero reward)
- `source/chassis_nav/chassis_nav/tasks/__init__.py` (added `from . import lift`)

### §1 Scene — decisions
- Robot `ARM_GRASP_CFG` (8 DOF) [pre-resolved]; DexCube at `[-0.2,-0.1,0.055]`, table/plane/light from push base [canonical].
- Did NOT add FrameTransformer; used `body_pos_w` for ee_position (deterministic obs dim, brief option).
- Viewer eye=(1.5,0.8,1.2) lookat=(-0.2,-0.1,0.3) [pre-resolved].

### §2 Actions — decisions
- arm_action JointPositionActionCfg (6, scale 0.5, use_default_offset); gripper_action
  BinaryJointPositionActionCfg (open 0.0325 / close 0.0). Action dim = 7. [pre-resolved]

### §3 Reset — decisions
- reset_robot_joints arm offset ±0.3; reset_gripper_open offset (0,0) on finger joints
  (re-opens to default each episode); reset_object_position pose ±0.05. [canonical/pre-resolved]

### §4 Goal + Termination — decisions
- Option A chosen: NO CommandsCfg, height-based lift goal. Terminations: time_out (200 steps) +
  object_dropping (root_height_below_minimum, -0.05). No success early-termination in this version.

### §5 Observation — decisions
- Layout (27): joint_pos(6)+joint_vel(6)+gripper_pos(2)+object_position(3)+ee_position(3)+actions(7).
- gripper_pos uses absolute `joint_pos` (interpretable open~0.0325 / closed 0.0).
- ee_position task-local fn; robot_cfg passed EXPLICITLY in params (see bug below).
- noise ±0.01 on joint_pos/joint_vel only; enable_corruption=True, concatenate_terms=True.

## Smoke block

Smokes run unbuffered in background (foreground stdout capture truncates Isaac boot logs).

| smoke | file | attempts | verdict |
|-------|------|----------|---------|
| S1 | smokes/smoke_s1.py | 2 | pass — `S1 OK: env instantiated` (obs Box(1,27), act Box(1,7)) |
| S2 | smokes/smoke_s2.py | 2 | pass — `S2 OK: (1, 7)` |
| S3 | smokes/smoke_s3.py | 1 | pass — `S3 OK: 5 resets, obs dim 27, joint spread=0.5008` |
| S4 | smokes/smoke_s4.py | 1 | pass — `S4 OK: time_out at step 200` |
| S5 | smokes/smoke_s5.py | 1 | pass — `S5 OK: shape=[1, 27], layout verified, all finite` |

### Iterations / diagnoses
- **S1 attempt 1 -> fail:** `TypeError: 'slice' object is not subscriptable` in
  `ee_position_in_robot_root_frame`. Root cause: a `SceneEntityCfg` left as a function-default is
  NOT resolved by the ObservationManager, so `body_ids` stayed `slice(None)`. Fix: pass
  `robot_cfg=SceneEntityCfg("robot", body_names=[EE_BODY_NAME])` EXPLICITLY in the obs term's
  `params` (mirrors how the push rewards pass robot_cfg). S1 attempt 2 -> pass.
- **S1/S2 verdict-line ordering:** Isaac Sim `simulation_app.close()` hard-exits the interpreter,
  so the `S<N> OK` print after `app.close()` never flushed. Fix: print verdict + flush BEFORE
  `app.close()` in all 5 smokes.
- **S2 attempt 1 -> fail:** `AttributeError: 'numpy.ndarray' object has no attribute 'to'` —
  `env.action_space.sample()` returns numpy but ManagerBasedRLEnv.step calls `.to(device)`. Fix:
  wrap with `torch.from_numpy(...).to("cuda:0")`. S2 attempt 2 -> pass.

No `task-implementation.md` patches were required (its smoke templates target the 6-DOF push task;
the differences above are task-specific to the 7-DOF lift env, handled in the rendered smokes).

## Final verdict

| section | verdict |
|---------|---------|
| §1 | pass |
| §2 | pass |
| §3 | pass |
| §4 | pass |
| §5 | pass |

Total files written: 8 (7 new lift-task files + 1 line in tasks/__init__.py).
Total doc patches applied: 0.
status: pass
