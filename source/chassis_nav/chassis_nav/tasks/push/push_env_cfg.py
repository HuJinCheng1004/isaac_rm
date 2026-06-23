# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""RM 机器人右臂"推方块"任务的 ManagerBasedRL 环境配置。

机器人底盘固定（fix_base=True），升降杆锁定在 0.5 m 处，只有
r_joint1~r_joint6 接受绝对关节位置指令。任务目标：用掌面将桌面
上的 DexCube 方块推至随机 2D 目标点（±5 cm 以内算成功）。

工作空间说明（smoke 实测）：
  r_base_joint1 rpy=(0,-45°,0) 使臂在零位时朝机器人后方 (-x) 伸展。
  零位 EE (r_link8) ≈ (-0.615, -0.224, 0.879)；方块/桌子因此放在 -x 侧。
  桌面中心 (-0.5, -0.2, 0.0)，方块初始 (-0.5, -0.2, 0.055)。
  目标区域（机器人根系下）: x∈[-0.75,-0.25], y∈[-0.40,0.00]。
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
from isaaclab.assets import RigidObjectCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

from chassis_nav.robots.arm import ARM_PUSH_CFG, ARM_JOINT_NAMES, EE_BODY_NAME

from . import mdp

# 右臂关节过滤器（用于 obs 和 reset 事件）
_ARM_CFG = SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES)


##
# 场景
##
@configclass
class RMPushSceneCfg(InteractiveSceneCfg):
    """地面 + 桌子 + 固定底盘机器人 + 方块 + 穹形灯。"""

    robot: ArticulationCfg = ARM_PUSH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # DexCube 方块：桌面中心附近生成（-x 侧，手臂可达区域）
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

    # SeattleLabTable：表面约在 z=0，底部沉至 z=-1.05（接地平面）
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
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ARM_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 右臂关节位置（相对默认位）× 6
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _ARM_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # 右臂关节速度 × 6
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _ARM_CFG},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        # 方块在机器人根系下的位置 × 3
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # 目标指令（pos+quat）× 7
        target_object_position = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "object_pose"}
        )
        # 上一步动作 × 6
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    """目标：桌面上一个 2D 随机位置，每 4 s 重采样。"""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=EE_BODY_NAME,
        resampling_time_range=(4.0, 4.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.3, -0.1),     # 方块初始 x=-0.5，向 +x 推 0~0.2 m
            pos_y=(-0.3, -0.1),     # 方块初始 y=-0.2，±0.1 m
            pos_z=(0.04, 0.04),     # 目标高度 0.04 m
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class RewardsCfg:
    """Reward terms (IsaacLab RewardManager sums weight_i * dt * func_i per step)."""

    # Reaching: pull EE toward block CoM (tanh kernel).
    # iter 3: weight 1.0 -> 4.0. iter-1/2 evidence: reaching collapsed to ~0 (EE ends
    # ~31 cm from block) because reaching (1.0) was dwarfed by the tracking total
    # (16+8+5=29) and a single ballistic shove maximized tracking. Raising reaching to
    # 4.0 makes SUSTAINED EE-block proximity worth keeping (income only while the hand
    # stays on the block), without dominating the goal signal (still << coarse 16).
    reaching_block = RewTerm(
        func=mdp.object_ee_distance_body,
        params={
            "std": 0.1,
            "robot_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
        },
        weight=4.0,
    )
    # Tracking coarse: pull block toward goal (tanh kernel, std=0.3). UNGATED -- keeps
    # the long-range shove-initiation gradient intact (reward-experience #9). Unchanged.
    block_to_goal_tracking = RewTerm(
        func=mdp.block_to_goal_distance,
        params={"std": 0.3, "command_name": "object_pose"},
        weight=16.0,
    )
    # Tracking MID-BAND -- iter 3: CONTACT-GATED (was ungated in iter 2).
    # iter-2 root cause: the open-loop swat farmed the ungated mid/fine precision bands
    # with the arm already withdrawn (block coasts goal-ward, policy banks reward with no
    # hand on the block). Multiplying the std=0.12 kernel by the soft contact gate
    # g=1-tanh(d_ee_block/0.05) zeroes this income unless the EE is touching the block
    # (g~0 at the iter-2 31 cm abandon distance), so closed-loop pushing beats the swat.
    # std=0.12 keeps the proven peak-gradient band (3-15 cm); weight 8 unchanged.
    block_to_goal_tracking_mid_band = RewTerm(
        func=mdp.block_to_goal_distance_contact_gated,
        params={
            "std": 0.12,
            "std_gate": 0.10,
            "command_name": "object_pose",
            # pass robot_cfg EXPLICITLY so RewardManager resolves body_ids (defaults
            # are not auto-resolved; the unresolved slice would break body_pos_w idx).
            "robot_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
        },
        weight=8.0,
    )
    # Tracking fine -- iter 3: CONTACT-GATED (was ungated in iter 2).
    # Same gate as mid_band. std=0.08 (proven, iter 2) for the last cm; weight 5
    # unchanged. The policy can only collect the sharp last-cm pull while in contact,
    # forcing it to ride the block in instead of leaving it to coast.
    block_to_goal_tracking_fine_grained = RewTerm(
        func=mdp.block_to_goal_distance_contact_gated,
        params={
            "std": 0.08,
            "std_gate": 0.10,
            "command_name": "object_pose",
            "robot_cfg": SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
        },
        weight=5.0,
    )
    # Success: binary indicator. EFFECTIVELY LOGGING-ONLY (matches the proven library
    # base, which uses weight=0.0). MECHANICAL DEVIATION: this repo's block_at_goal sets
    # env._reach_success (read by harbor eval success_rate) as a side-effect in its body,
    # but RewardManager SKIPS a term's func entirely when weight == 0.0 (see
    # reward_manager.py:146). A negligible positive weight (1e-6 -> ~3e-8/step after
    # dt-scale) forces the func to run so the side-effect fires, while contributing no
    # meaningful optimization signal -- preserving the proven "logging-only" intent.
    success = RewTerm(
        func=mdp.block_at_goal,
        params={"threshold": 0.05, "command_name": "object_pose"},
        weight=1e-6,
    )
    # Regularizers
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": _ARM_CFG},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # 方块掉落桌面
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class EventCfg:
    # §3 复位：右臂关节在默认零位附近加均匀随机偏移
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.3, 0.3),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": _ARM_CFG,
        },
    )
    # §3 复位：方块在桌面中心 ±5 cm 内随机
    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object"),
        },
    )
    # §7 DR（startup，每 env 实例固定）：右手指摩擦
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
    # §7 DR（startup）：方块质量 ±20%
    block_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )


##
# 环境主配置
##
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
        # 30 Hz 控制，1/60 s 物理步，每步 2 次物理子步
        self.decimation = 2
        self.episode_length_s = 200.0 / 30.0   # ≈ 6.67 s，200 控制步
        self.sim.dt = 1.0 / 60.0
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        # 从 +x/+y 侧平视桌面，清晰看到手臂末端与桌面/方块的相对高度
        self.viewer.eye = (1.5, 0.8, 0.8)
        self.viewer.lookat = (-0.5, -0.2, 0.3)


@configclass
class RMPushBlockEnvCfg_PLAY(RMPushBlockEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
