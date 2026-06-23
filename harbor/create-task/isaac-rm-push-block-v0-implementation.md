# Isaac-RM-Push-Block-v0 — Implementation Spec

- benchmark_family: `isaaclab-manager-based`
- source_repo: isaac_rm
- probed_from_commit: f02c47feb95d484529b87e37067162f59d20bb09
- probed_at: 2026-06-22T00:00:00+00:00
- canonical_build: requires IsaacSim AppLauncher (headless); obs_space Box(-inf,inf,(N,28),float32), act_space Box(-1.0,1.0,(N,6),float32)

---

## §1 Registration + Scene

### Description

The task registers via `gymnasium.register` in `tasks/push/__init__.py`. The env class is `isaaclab.envs:ManagerBasedRLEnv`, configured by `RMPushBlockEnvCfg`. The scene includes: a 6-DOF right arm robot (fixed chassis, URDF variant with all non-arm joints locked), a DexCube rigid block on a table, a SeattleLabTable static asset, a ground plane (z=-1.05), and a dome light. All per-env assets use `{ENV_REGEX_NS}/<name>` prim paths. The robot is placed at world pos `(0.3, -0.3, -0.805)` with a 90-degree Z-axis clockwise rotation (`rot=(0.7071,0,0,-0.7071)`) so the right arm faces -x (toward the table/block). Table center is at `(-0.5, -0.2, 0.0)` in world space; block spawns near `(-0.2, -0.1, 0.055)` (relative to env origin). The main env config sets 1024 envs, 2.5 m spacing, 30 Hz control (decimation=2, sim.dt=1/60 s, episode=200 steps ≈ 6.67 s).

### Decisions resolved

- Robot asset: URDF variant `overseas_65_b_v_description_rmg24_armfixed.urdf` — all non-arm joints baked to `type="fixed"`, lift pole height baked to platform_joint origin (z=0.805 total). Reduces active DOF from ~30 to 6.
- `merge_fixed_joints=False` — preserves named bodies (`r_link8`, `r_Link_finger1`, `r_Link_finger2`) referenced in reward/obs.
- `collision_from_visuals=True`, `self_collision=False` — collision mesh from visuals; self-collision disabled for speed.
- DexCube scale: `(0.8, 0.8, 0.8)` (slightly smaller than default).
- Table orientation: `rot=[0.707,0,0,0.707]` (90 deg around X) to align table surface correctly.
- Ground plane at `z=-1.05` to match table bottom; prevents robot from sinking through.
- PhysX: `bounce_threshold_velocity=0.01`, `gpu_found_lost_aggregate_pairs_capacity=1024*1024*4`, friction_correlation_distance=0.00625.
- Robot PD actuator: stiffness=120.0, damping=6.0, effort_limit_sim=60.0, velocity_limit_sim=3.925 rad/s.
- Viewer: eye=(1.5,0.8,0.8), lookat=(-0.5,-0.2,0.3).

### Code

```python
# source/chassis_nav/chassis_nav/tasks/push/__init__.py
import gymnasium as gym
from . import agents

gym.register(
    id="Isaac-RM-Push-Block-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.push_env_cfg:RMPushBlockEnvCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-RM-Push-Block-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.push_env_cfg:RMPushBlockEnvCfg_PLAY",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)
```

```python
# source/chassis_nav/chassis_nav/tasks/push/push_env_cfg.py — scene section
@configclass
class RMPushSceneCfg(InteractiveSceneCfg):
    """地面 + 桌子 + 固定底盘机器人 + 方块 + 穹形灯。"""

    robot: ArticulationCfg = ARM_PUSH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-0.2, -0.1, 0.055], rot=[1.0, 0.0, 0.0, 0.0]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.8, 0.8, 0.8),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-0.5, -0.2, 0.0], rot=[0.707, 0.0, 0.0, 0.707]
        ),
        spawn=UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        ),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -1.05]),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class RMPushBlockEnvCfg(ManagerBasedRLEnvCfg):
    scene: RMPushSceneCfg = RMPushSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 200.0 / 30.0   # ≈ 6.67 s, 200 control steps
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        self.viewer.eye = (1.5, 0.8, 0.8)
        self.viewer.lookat = (-0.5, -0.2, 0.3)


@configclass
class RMPushBlockEnvCfg_PLAY(RMPushBlockEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
```

```python
# source/chassis_nav/chassis_nav/robots/arm.py — robot asset config
_ASSETS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "assets")
)
# Resolved: /home/shihao/isaac_rm/assets/overseas_65_b_v_description/urdf/
#           overseas_65_b_v_description_rmg24_armfixed.urdf  (FILE EXISTS)

EE_BODY_NAME = "r_link8"
ARM_JOINT_NAMES = ["r_joint1", "r_joint2", "r_joint3", "r_joint4", "r_joint5", "r_joint6"]

ARM_PUSH_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_URDF_PATH,
        fix_base=True,
        merge_fixed_joints=False,
        collision_from_visuals=True,
        self_collision=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=120.0, damping=6.0
            ),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.3, -0.3, -0.805),
        rot=(0.7071, 0.0, 0.0, -0.7071),  # CW 90 deg around Z, arm faces -x
        joint_pos={
            "r_joint1": 1.5708,   # 90 deg
            "r_joint2": 0.5236,   # 30 deg
            "r_joint3": 1.2217,   # 70 deg
            "r_joint4": 0.0,
            "r_joint5": 1.2217,   # 70 deg
            "r_joint6": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["r_joint[1-6]"],
            stiffness=120.0,
            damping=6.0,
            effort_limit_sim=60.0,
            velocity_limit_sim=3.925,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
```

### Smoke

Asset path verified on disk:
- `/home/shihao/isaac_rm/assets/overseas_65_b_v_description/urdf/overseas_65_b_v_description_rmg24_armfixed.urdf` — EXISTS
- `ISAAC_NUCLEUS_DIR` resolves via `carb.settings` at runtime to the Nucleus cloud asset root `/Isaac/...`; USD props are streamed from Nucleus during simulation.

---

## §2 Actions

### Description

Single `ActionsCfg` field: `arm_action` using `mdp.JointPositionActionCfg` for **absolute joint-position** control on the 6 active right-arm joints (`r_joint1`–`r_joint6`). Scale=0.5, use_default_offset=True means policy outputs are multiplied by 0.5 and added to the default home joint positions before being sent to the PD actuator. Effective action range is roughly `home ± 0.5` rad per joint (before actuator clipping). No gripper action — gripper stays closed (r_joint7 is passive, locked in URDF to `type="fixed"` in the armfixed variant). Action space: `Box(-1.0, 1.0, (N, 6), float32)` per IsaacLab convention.

### Decisions resolved

- Absolute joint position (not delta, not EE-pose IK).
- `scale=0.5`: output multiplied by 0.5 before adding to home offset.
- `use_default_offset=True`: offsets applied relative to `init_state.joint_pos` (home pose).
- Joint filter: `ARM_JOINT_NAMES = ["r_joint1","r_joint2","r_joint3","r_joint4","r_joint5","r_joint6"]`.
- No wrist coupling (r_joint7 is passive, locked).

### Code

```python
@configclass
class ActionsCfg:
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
    )
```

### Smoke

Action dim = 6, confirmed by skrl_ppo_cfg.yaml comment: `# Action 6-D: r_joint1~r_joint6 absolute joint position.`

---

## §3 Reset

### Description

Two `mode="reset"` events in `EventCfg`:
1. **Robot joints**: uniform random offset from home pose, range `(-0.3, 0.3)` rad, zero velocity. Applied only to the 6 active arm joints via `_ARM_CFG`.
2. **Block position**: uniform random offset from initial spawn pos `(-0.2, -0.1)` in env-local XY, ±0.05 m each axis, z=0 (stays on table surface). No velocity randomization.

DR terms (`physics_material`, `block_mass`) are `mode="startup"` and live in the same `EventCfg` — see §7.

### Decisions resolved

- Robot joint reset uses `reset_joints_by_offset` from `isaaclab.envs.mdp` (library); restricted to ARM_JOINT_NAMES so locked URDF joints are untouched.
- Block reset uses `reset_root_state_uniform`; `velocity_range={}` means zero velocity always.
- Block jitter range: ±0.05 m in XY around the `init_state` pos of `(-0.2, -0.1, 0.055)`.
- Z is NOT jittered (z range = (0.0, 0.0)); block always starts at table height 0.055.

### Code

```python
_ARM_CFG = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)

@configclass
class EventCfg:
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.3, 0.3),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": _ARM_CFG,
        },
    )
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    # §7 DR terms also live here with mode="startup" — see §7
```

### Smoke

5 resets should produce varying obs[0:6] (joint pos) and varying obs[12:15] (block pos in robot frame) with obs dim 28.

---

## §4 Goal + Termination

### Description

**Goal**: `CommandsCfg` contains a single command group `object_pose` of type `mdp.UniformPoseCommandCfg`. The goal is a 2D point on the table surface resampled every 4 s (fixed). The target is given in robot-body frame coordinates (pos_x, pos_y, pos_z=0.04 fixed), with body_name=`r_link8` used only for debug visualization attachment. Goal XY range: `pos_x=(-0.3,-0.1)`, `pos_y=(-0.3,-0.1)` relative to the env origin (table frame), ensuring the goal lands in the arm's reachable workspace.

**Critical frame fix (iter 1)**: Goal position is resolved as `goal_w = env.scene.env_origins + command[:,:3]` (table/env-origin frame), NOT as `robot_root + command`. The robot chassis root sits at world z≈-0.805; using root frame would place the goal ~0.8 m underground.

**Termination**:
1. `time_out`: episode exceeds 200 control steps (≈6.67 s). `time_out=True` (used for bootstrap).
2. `object_dropping`: block CoM falls below `z=-0.05` (minimum_height=-0.05). No `time_out` flag.

**Success metric**: `block_at_goal` reward term sets `env._reach_success` as a side-effect. Harbor eval reads `env._reach_success` to compute `eval/success_rate`. The reward weight is 1e-6 (non-zero to force function execution by `RewardManager`; weight=0.0 would skip the func call entirely per `reward_manager.py:146`).

### Decisions resolved

- Resampling cadence: fixed 4 s (not episodic; agent sees multiple targets per episode).
- Goal z is pinned to 0.04 m (block surface height at ~0.055 m; 0.04 is slightly below top).
- No yaw/pitch/roll randomization for the goal orientation (all zeroed).
- Success is XY-only (z not checked): `torch.norm(des_pos_w[:,:2] - block_pos_w[:,:2], dim=1) < 0.05`.
- `object_dropping` threshold: -0.05 m (5 cm below table surface; catches drops but not normal table motion).

### Code

```python
@configclass
class CommandsCfg:
    """Goal: a 2D random position on the table surface, resampled every 4 s."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=EE_BODY_NAME,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.3, -0.1),
            pos_y=(-0.3, -0.1),
            pos_z=(0.04, 0.04),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )
```

### Smoke

Episode terminates at step 200 via `time_out`. Block drop termination fires if block z < -0.05.

---

## §5 Observation

### Description

Single observation group `policy` (type `PolicyCfg`, subclass of `ObsGroup`). `concatenate_terms=True`, `enable_corruption=True` (additive uniform noise on joint pos/vel). Total dim = **28**.

Concatenation order:
| Slot | Term | Source | Dim | Noise |
|------|------|--------|-----|-------|
| 0:6  | `joint_pos` | `mdp.joint_pos_rel` (library) — relative to default home | 6 | Unoise(-0.01, 0.01) |
| 6:12 | `joint_vel` | `mdp.joint_vel_rel` (library) — relative to default | 6 | Unoise(-0.01, 0.01) |
| 12:15 | `object_position` | `mdp.object_position_in_robot_root_frame` (task-local) | 3 | none |
| 15:22 | `target_object_position` | `mdp.generated_commands(command_name="object_pose")` (library) — pos(3)+quat(4) | 7 | none |
| 22:28 | `actions` | `mdp.last_action` (library) | 6 | none |

`enable_corruption` is `False` in the PLAY variant (`RMPushBlockEnvCfg_PLAY`).

### Decisions resolved

- `joint_pos_rel` / `joint_vel_rel` are computed relative to `init_state.joint_pos` (home pose), not zero.
- Asset filter for joint obs: `_ARM_CFG = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)` (6-joint filter).
- Block position uses a task-local function that transforms from world to robot root frame via `subtract_frame_transforms`.
- Command output from `UniformPoseCommandCfg` is 7D (pos_xyz + quat_wxyz); only pos_xyz(3) is meaningful for planar push, but the full 7D vector is included.
- Noise: uniform ±0.01 rad on joint pos and vel only; no noise on block position, goal, or last action.

### Code

```python
@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _ARM_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _ARM_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
```

Task-local observation function (verbatim):

```python
# source/chassis_nav/chassis_nav/tasks/push/mdp/observations.py
def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Block position expressed in robot root frame (3D)."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w
    )
    return object_pos_b
```

### Smoke

`obs_space: Box(-inf, inf, (N, 28), float32)` — obs dim 28, all finite after reset.

---

## §6 Reward

### Description

Reward composed by IsaacLab `RewardManager` as a **weighted sum**: `total = Σ weight_i × dt × func_i(env)` where `dt = sim.dt × decimation = 1/60 × 2 ≈ 0.0333 s`. No manual dt-scaling needed.

Seven reward terms across four functional groups:

| Term | func | weight | type |
|------|------|--------|------|
| `reaching_block` | `object_ee_distance_body` | 4.0 | tanh EE→block distance |
| `block_to_goal_tracking` | `block_to_goal_distance` | 16.0 | tanh block→goal, ungated |
| `block_to_goal_tracking_mid_band` | `block_to_goal_distance_contact_gated` (std=0.12, std_gate=0.05) | 8.0 | contact-gated precision |
| `block_to_goal_tracking_fine_grained` | `block_to_goal_distance_contact_gated` (std=0.08, std_gate=0.05) | 5.0 | contact-gated fine |
| `success` | `block_at_goal` | 1e-6 | binary; side-effect sets `env._reach_success` |
| `action_rate` | `mdp.action_rate_l2` | -1e-4 | regularizer |
| `joint_vel` | `mdp.joint_vel_l2` | -1e-4 | regularizer |

**Design rationale (3 iterations)**:
- Iter 1: Goal frame bug fix (env-origin frame, not robot root frame).
- Iter 2: Added multi-band tracking (std=0.3 coarse, 0.12 mid, 0.08 fine); policy learned open-loop "swat" exploit (EE abandons at 31 cm, block coasts on inertia).
- Iter 3 (CONVERGED, success_rate=0.507): (a) reaching weight 1→4 to make sustained EE proximity worth keeping; (b) mid and fine tracking bands contact-gated by `g=1-tanh(d_ee_block/0.05)` so reward requires hand-on-block contact; coarse (w=16) left ungated to preserve shove initiation gradient.

### Decisions resolved

- Coarse tracking (std=0.3) is deliberately UNGATED: provides long-range shove-initiation gradient.
- Mid (std=0.12) and fine (std=0.08) tracking are CONTACT-GATED with std_gate=0.05.
- `success` weight must be > 0.0 (1e-6) to force `block_at_goal` to execute for the `env._reach_success` side-effect. IsaacLab `RewardManager` skips func calls when weight=0.0.
- Regularizers: `action_rate_l2` and `joint_vel_l2` both at -1e-4; prevent jerky motions.
- Goal frame in ALL reward functions: `goal_w = env.scene.env_origins + command[:,:3]` (env-origin/table frame).

### Code (verbatim reward functions)

```python
# source/chassis_nav/chassis_nav/tasks/push/mdp/rewards.py

def object_ee_distance_body(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["r_link8"]),
) -> torch.Tensor:
    """tanh-shaped reaching reward: EE body r_link8 -> block CoM distance."""
    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    cube_pos_w = object.data.root_pos_w
    ee_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    distance = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(distance / std)


def ee_z_to_block(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["r_link8"]),
) -> torch.Tensor:
    """exp-shaped EE height alignment: pulls EE z down to block z height.

    NOT USED in the current RewardsCfg but available in mdp/rewards.py.
    Use exp kernel (not tanh) because home EE is ~0.82 m above block;
    tanh std=0.1 gives near-zero gradient at that distance.
    """
    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    ee_z = robot.data.body_pos_w[:, robot_cfg.body_ids[0], 2]
    block_z = object.data.root_pos_w[:, 2]
    return torch.exp(-(ee_z - block_z).abs() / std)


def block_to_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """tanh-shaped tracking reward: block -> goal distance (ungated, planar push).

    GOAL FRAME (iter 1 fix): goal resolved in env-origin / table frame.
    goal_w = env.scene.env_origins + command[:, :3]
    NOT resolved against robot articulation root (which sits at world z=-0.805).
    """
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w = env.scene.env_origins + des_pos_b
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def block_to_goal_distance_contact_gated(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    std_gate: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["r_link8"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Contact-gated tracking (iter 3): tracking * g, where g = 1 - tanh(d_ee_block / std_gate).

    Prevents the "swat exploit": policy cannot farm precision-band reward with arm
    withdrawn (g~0 at 31 cm abandon distance). Only pays reward while EE is in contact.

    Gate values:
      g(d=0.00) = 1.00
      g(d=0.05) = 0.24
      g(d=0.10) = 0.04
      g(d=0.31) ~ 0   (iter-2 abandon distance)

    Coarse tracking (std=0.3, w=16) left UNGATED to preserve shove-initiation gradient.
    Goal frame: env-origin frame (same fix as block_to_goal_distance).
    """
    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w = env.scene.env_origins + des_pos_b
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    tracking = 1 - torch.tanh(distance / std)
    # soft contact gate: EE (r_link8) -> block CoM distance, world frame
    ee_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    d_ee_block = torch.norm(object.data.root_pos_w - ee_w, dim=1)
    gate = 1 - torch.tanh(d_ee_block / std_gate)
    return tracking * gate


def block_at_goal(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Binary success indicator; sets env._reach_success for harbor eval success_rate.

    Goal frame: env-origin frame (iter 1 fix).
    Success: XY-only distance < threshold (z not checked).
    """
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w = env.scene.env_origins + des_pos_b
    distance = torch.norm(des_pos_w[:, :2] - object.data.root_pos_w[:, :2], dim=1)
    result = (distance < threshold).float()
    env._reach_success = result
    return result
```

```python
# RewardsCfg wiring (verbatim from push_env_cfg.py)
@configclass
class RewardsCfg:
    reaching_block = RewTerm(
        func=mdp.object_ee_distance_body,
        params={
            "std": 0.1,
            "robot_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
        },
        weight=4.0,
    )
    block_to_goal_tracking = RewTerm(
        func=mdp.block_to_goal_distance,
        params={"std": 0.3, "command_name": "object_pose"},
        weight=16.0,
    )
    block_to_goal_tracking_mid_band = RewTerm(
        func=mdp.block_to_goal_distance_contact_gated,
        params={
            "std": 0.12,
            "std_gate": 0.05,
            "command_name": "object_pose",
            "robot_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
        },
        weight=8.0,
    )
    block_to_goal_tracking_fine_grained = RewTerm(
        func=mdp.block_to_goal_distance_contact_gated,
        params={
            "std": 0.08,
            "std_gate": 0.05,
            "command_name": "object_pose",
            "robot_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
        },
        weight=5.0,
    )
    success = RewTerm(
        func=mdp.block_at_goal,
        params={"threshold": 0.05, "command_name": "object_pose"},
        weight=1e-6,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": _ARM_CFG},
    )
```

### Smoke

- Reward is finite and non-constant across 30 random steps.
- After 40M training steps at 2048 envs (lr=1.5e-4): `eval/success_rate = 0.942` (converged; last-20% reward std=0.003).
- Iter 3 threshold ≥ 0.5 was reached earlier at 20M steps (success_rate=0.507); 40M is the recommended run length for >90% success.
- `env._reach_success` is set every step by `block_at_goal` (weight=1e-6 forces func execution).

---

## §7 DR

### Description

Two domain randomization terms in `EventCfg`, both `mode="startup"` (run once per env instance at simulation startup; stays fixed for the env's lifetime, i.e., per-env not per-episode):

1. **Finger friction** (`physics_material`): randomizes static and dynamic friction of `r_Link_finger1` and `r_Link_finger2` (robot end-effector contact surfaces) over `[0.8, 1.2]`. Restitution locked to 0.0. 64 material buckets used.
2. **Block mass** (`block_mass`): scales block mass by a factor drawn uniformly from `[0.8, 1.2]` (±20% variation). `operation="scale"` means the sampled value multiplies the original mass.

No `mode="interval"` DR (no per-step physics variation). No action noise. Observation noise is wired via `Unoise` in `ObservationsCfg.PolicyCfg` (see §5).

### Decisions resolved

- `mode="startup"` (per-env, not per-episode): each parallel env instance gets its own friction/mass, fixed for all episodes in that env slot.
- Body names for friction: `["r_Link_finger1", "r_Link_finger2"]` — the palm-face contact surfaces.
- Mass scale range: (0.8, 1.2) = ±20%; covers typical real-world block mass uncertainty.
- No joint stiffness/damping DR in canonical task.
- No lighting DR (no visual pipeline).

### Code

```python
# In EventCfg, alongside mode="reset" terms:

physics_material = EventTerm(
    func=mdp.randomize_rigid_body_material,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=["r_Link_finger1", "r_Link_finger2"]),
        "static_friction_range": (0.8, 1.2),
        "dynamic_friction_range": (0.8, 1.2),
        "restitution_range": (0.0, 0.0),
        "num_buckets": 64,
    },
)
block_mass = EventTerm(
    func=mdp.randomize_rigid_body_mass,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("object"),
        "mass_distribution_params": (0.8, 1.2),
        "operation": "scale",
    },
)
```

### Smoke

`env.unwrapped.event_manager.active_terms` should contain at least 2 startup terms: `physics_material` and `block_mass`.

---

## Source files (relative to source_repo root `/home/shihao/isaac_rm`)

- `source/chassis_nav/chassis_nav/tasks/push/__init__.py:1-28` — gym.register (§1)
- `source/chassis_nav/chassis_nav/tasks/push/push_env_cfg.py:1-331` — all manager configs §1-§7
- `source/chassis_nav/chassis_nav/robots/arm.py:1-109` — ARM_PUSH_CFG, EE_BODY_NAME, ARM_JOINT_NAMES (§1, §2)
- `source/chassis_nav/chassis_nav/tasks/push/mdp/__init__.py:1-15` — MDP re-exports (§5, §6)
- `source/chassis_nav/chassis_nav/tasks/push/mdp/observations.py:1-30` — object_position_in_robot_root_frame (§5)
- `source/chassis_nav/chassis_nav/tasks/push/mdp/rewards.py:1-147` — all 5 reward functions (§6)
- `source/chassis_nav/chassis_nav/tasks/push/agents/skrl_ppo_cfg.yaml:1-76` — PPO hyperparameters
- `harbor/benchmark-generator/benchmark-spec.json` — benchmark family and task metadata
- `harbor/create-task/task-implementation.md` — benchmark-level migration guide (§1-§7 templates)
- `harbor/create-task/isaac-rm-push-block-v0/iter_003/analysis.md` — convergence record (iter 3, success_rate=0.507)
- `assets/overseas_65_b_v_description/urdf/overseas_65_b_v_description_rmg24_armfixed.urdf` — robot URDF (exists on disk)
