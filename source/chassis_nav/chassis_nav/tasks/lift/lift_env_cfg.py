# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""ManagerBasedRL env config for the RM right-arm "lift block" task.

The chassis is fixed (fix_base=True), lift-pole locked at 0.534 m. The right arm
(r_joint1~r_joint6) accepts absolute joint-position commands and the gripper
(r_Joint_finger1/2) is driven by a binary open/close command. Goal: grasp the
5 cm DexCube on the table and lift it above 0.15 m (world z).

Workspace notes (mirrors the push task):
  Table centre (-0.5, -0.2, 0.0); cube init (-0.2, -0.1, 0.055) in env-local frame.
  Robot root sits at world z=-0.805 (lift-pole base); the table surface is world z~0.

NOTE: §6 reward is a constant-zero PLACEHOLDER (real reward authored by reward-generator).
§7 DR is intentionally empty.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

from isaaclab.assets import ArticulationCfg

from chassis_nav.robots.arm import (
    ARM_GRASP_CFG,
    ARM_JOINT_NAMES,
    EE_BODY_NAME,
    GRIPPER_JOINT_NAMES,
    GRIPPER_OPEN,
    GRIPPER_CLOSED,
)

from . import mdp

# Right-arm joint filter (used by obs + reset events)
_ARM_CFG = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)
# Gripper joint filter (obs)
_GRIPPER_CFG = SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES)
# Arm + gripper filter (reset: re-centre arm on default, re-open gripper)
_ARM_GRIPPER_CFG = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES + GRIPPER_JOINT_NAMES)


##
# Scene
##
@configclass
class RMLiftSceneCfg(InteractiveSceneCfg):
    """Ground + table + fixed-chassis robot (8 DOF) + cube + dome light."""

    # Robot moved +0.2 m in X vs ARM_GRASP_CFG default (0.3 → 0.5) to create a clear
    # physical gap between the chassis and the table (table stays at X=-0.5).
    robot: ArticulationCfg = ARM_GRASP_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.5, -0.3, -0.805),
            rot=(0.7071, 0.0, 0.0, -0.7071),
            joint_pos={
                "r_joint1": 1.5708,
                "r_joint2": 0.5236,
                "r_joint3": 1.2217,
                "r_joint4": 0.0,
                "r_joint5": 1.2217,
                "r_joint6": 0.0,
                "r_Joint_finger1": GRIPPER_OPEN,
                "r_Joint_finger2": GRIPPER_OPEN,
            },
            joint_vel={".*": 0.0},
        ),
    )

    # DexCube: spawned on the table surface (-x side, within arm reach).
    # Block shifted +0.1 m in X vs push task to compensate for the robot's backward move,
    # keeping arm reach ≈ 0.63 m (vs 0.54 m before — within the arm's workspace).
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[-0.1, -0.1, 0.055], rot=[1.0, 0.0, 0.0, 0.0]
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

    # SeattleLabTable: surface ~z=0, base sinks to z=-1.05 (ground plane)
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


##
# MDP
##
@configclass
class ActionsCfg:
    # 6 arm joints: absolute joint-position control (scaled deltas off default pose)
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
    )
    # gripper: 1 binary open/close command -> both finger joints
    # iter_005: SWAPPED open/close so action≥0 (PPO default near-zero) → physically CLOSED.
    # IsaacLab mapping: action<0 → close_command; action≥0 → open_command.
    # Previously open_command=OPEN meant near-zero init kept gripper open forever.
    # Now open_command=CLOSED: the rest state is fingers together; policy outputs
    # negative to open the gripper for approach, then back to ≥0 to close and grasp.
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=GRIPPER_JOINT_NAMES,
        open_command_expr={"r_Joint_finger.*": GRIPPER_CLOSED},
        close_command_expr={"r_Joint_finger.*": GRIPPER_OPEN},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # arm joint positions (relative to default) x 6
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _ARM_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # arm joint velocities x 6
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _ARM_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # gripper finger positions (absolute, interpretable: ~0.0325 open / 0.0 closed) x 2
        gripper_pos = ObsTerm(
            func=mdp.joint_pos,
            params={"asset_cfg": _GRIPPER_CFG},
        )
        # cube position in robot root frame x 3
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # end-effector (r_link8) position in robot root frame x 3
        # NOTE: pass robot_cfg EXPLICITLY so the ObservationManager resolves body_ids
        # (a SceneEntityCfg left as a function default is NOT auto-resolved -> body_ids
        # stays slice(None) and the body_pos_w index raises 'slice not subscriptable').
        ee_position = ObsTerm(
            func=mdp.ee_position_in_robot_root_frame,
            params={"robot_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME])},
        )
        # last action x 7
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """§6 lift reward. Composer = sum (RewardManager sums weight_i * dt * func_i per step).

    Adapted from Isaac-Lift-Cube-Franka-v0: reach -> lift indicator -> height-tracking
    ladder, with the dynamic 3D goal-tracking pair replaced by a hard-coded world-height
    pull (cube_height) + a contact gate (grasping), since this task has NO CommandsCfg.
    dt scaling is left in place to match the proven push/Franka repo convention (the
    success weight 1e-6 is calibrated for it). Magnitude budget (nominal, pre-dt) strictly
    increases reach(1) -> lift(15) -> height(16), later terms gated on lifted.
    """

    # Reaching: pull EE (r_link8) toward cube CoM (tanh, std=0.1).
    reaching_block = RewTerm(
        func=mdp.object_ee_distance_body,
        params={
            "std": 0.1,
            "robot_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
        },
        weight=2.0,
    )
    # Dense lift reward: tanh(height_gain/0.05) × near × gripper_closed ∈ [0,1].
    # iter_009: tanh归一化替换原始height_gain。
    # iter_010: 保持不变（核心信号）。
    lift_with_grip = RewTerm(
        func=mdp.cube_height_with_grip,
        params={
            "initial_z": 0.055,
            "std_h": 0.05,
            "proximity_threshold": 0.08,
            "gripper_closed_threshold": 0.025,
            "robot_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES),
        },
        weight=20.0,
    )
    # iter_010: 夹爪闭合塑形（无高度门控）。
    # iter_009视频显示侧推行为：策略靠近方块但未闭爪。lift_with_grip 需要 near×closed×height
    # 三者同时成立，但"closed"没有独立梯度信号 → 策略学不到闭爪。
    # gripper_near_cube_shaping = near × (1-tanh(finger_pos/0.02)) 提供独立闭爪梯度。
    # weight=0.5: hover上限≈3.3/episode << reaching_block 13.3/episode，不会成为主策略。
    gripper_closure = RewTerm(
        func=mdp.gripper_near_cube_shaping,
        params={
            "proximity_threshold": 0.08,
            "std": 0.02,
            "robot_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES),
        },
        weight=0.5,
    )
    # iter_010: 方块横向速度惩罚。
    # iter_009视频：f010-f013 显示方块向右横向滑动（侧推）。
    # 直接惩罚 ‖v_xy‖ 封堵侧推策略；唯一零惩罚路径是垂直抬升。
    # weight=-2.0 = reaching_block 量级，侧推完全抵消接近奖励。
    cube_push_penalty = RewTerm(
        func=mdp.cube_lateral_velocity_penalty,
        weight=-2.0,
    )
    # Success: cube z > 0.15 m AND EE within 8 cm (finger-contact proxy). Logging-only:
    # weight 1e-6 forces RewardManager to run the func so env._reach_success is set
    # (the manager SKIPS a term whose weight is exactly 0.0).
    success = RewTerm(
        func=mdp.block_lifted_and_grasped,
        params={
            "lift_height": 0.15,
            "contact_threshold": 0.08,
            "robot_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
        },
        weight=1e-6,
    )
    # Motion smoothness: penalize joint velocity (arm joints only).
    # Weight -5e-4 (5× vs push-task default) to encourage slow compliant motion.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5e-4,
        params={"asset_cfg": _ARM_CFG},
    )
    # Jerk penalty: penalize sudden velocity changes (mean abs joint acceleration).
    # Weight -5e-4 matches joint_vel magnitude so jerk ≈ velocity in penalty scale.
    joint_jerk = RewTerm(
        func=mdp.joint_jerk_l2,
        weight=-5e-4,
        params={"asset_cfg": _ARM_CFG},
    )
    # Table avoidance: soft penalty when structural arm links are near the table.
    # Weight reduced -0.5 → -0.2 (iter_003): in iter_002 the -0.5 penalty was firing
    # as the arm legitimately reached toward the cube (table_avoidance=-0.053 episodic,
    # 100× higher than iter_001), discouraging the arm from entering the grasping pose.
    table_avoidance = RewTerm(
        func=mdp.table_avoidance_penalty,
        params={
            "table_height": 0.02,
            "std": 0.02,
            "robot_cfg": SceneEntityCfg(
                "robot",
                body_names=["r_link1", "r_link2", "r_link3", "r_link4", "r_link5", "r_link6"],
            ),
        },
        weight=-0.2,
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # cube drops off the table
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class EventCfg:
    # §3 reset: arm joints re-centred on default with uniform offset; gripper re-opened to default
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.3, 0.3),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": _ARM_CFG,
        },
    )
    # §3 reset: re-open the gripper to its default (OPEN) position each episode (no random offset)
    reset_gripper_open = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": _GRIPPER_CFG,
        },
    )
    # §3 reset: cube within ±5 cm of table-centre init pose
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )


##
# Env top-level config
##
@configclass
class RMLiftBlockEnvCfg(ManagerBasedRLEnvCfg):
    scene: RMLiftSceneCfg = RMLiftSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # 30 Hz control, 1/60 s physics step, 2 physics substeps per control step
        self.decimation = 2
        self.episode_length_s = 200.0 / 30.0   # ~6.67 s, 200 control steps
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        # Camera: robot at X=0.5, block at X=-0.1 — widen shot to show both
        self.viewer.eye = (1.7, 0.9, 1.3)
        self.viewer.lookat = (-0.1, -0.1, 0.3)


@configclass
class RMLiftBlockEnvCfg_PLAY(RMLiftBlockEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
