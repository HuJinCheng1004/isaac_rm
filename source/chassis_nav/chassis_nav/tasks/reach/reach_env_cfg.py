# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""底盘"到达目标点"任务的基于管理器的 RL 环境配置（简化版，无相机）。

与 ``approach`` 任务的区别：目标 3D 框位置**全局已知**（直接读目标刚体世界位姿），
没有相机/检测/噪声/延迟/升降杆。动作是纯 2-D 差速 (v, ω)。任务目标：机器人接近
目标、对准（目标落在机体正前方）、并在目标前方 ``D_STANDOFF`` 处低速停住。

所有任务级旋钮集中在下面的 ``# --- 可调参数 ---`` 块。
"""

from __future__ import annotations

import math

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
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from chassis_nav.robots.chassis import CHASSIS_CFG, CHASSIS_PARAMS

from . import mdp

# --------------------------------------------------------------------------- #
# --- 可调参数 --------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
CONTROL_HZ = 10.0           # 策略/控制速率
PHYSICS_HZ = 100.0          # 物理求解器速率
EPISODE_S = 20.0            # 情节长度 [s]（够导航 + 停靠）

V_RANGE = (-0.2, 0.5)       # 前向速度映射 [m/s]
W_RANGE = (-1.0, 1.0)       # 偏航速率映射 [rad/s]

TARGET_SIZE = (0.25, 0.25, 0.40)   # 目标立方体尺寸 [m]
TARGET_HEIGHT = 0.5         # 目标中心世界高度 [m]
SPAWN_RADIUS = (1.5, 4.0)   # 目标相对底盘起点的距离区间 [m]

D_STANDOFF = 0.8            # 停靠距离：底盘<->目标平面距离 [m]
DIST_TOL = 0.2             # 成功距离容差 [m]（成功环 [0.6, 1.0] m）
BEARING_TOL = 0.25         # 成功方位容差 [rad]（≈14°，目标需基本正前方）
SPEED_TOL = 0.1            # 成功速度容差：sqrt(v^2+w^2) [m/s & rad/s]
SUCCESS_DWELL = 5          # 成功条件需持续的步数（0.5 s @ 10 Hz）

APPROACH_LAMBDA = 1.5      # approach 指数势能的衰减尺度 [m]：exp(-dist_xy/lambda) - baseline。
                            # iter 5：取代 progress + near_ring，平滑单调势能，越近梯度越陡。
APPROACH_SPAWN_AVG = 2.75  # approach 锚定基线对应的平均 spawn 距离 [m]（spawn_radius 1.5–4.0 中点）。
                            # baseline = exp(-spawn_avg/lambda)≈0.160，使 approach 在 spawn 处≈0：
                            # 杜绝"零动作静止收奖"局部最优，且无刷分门控边界。

DT = 1.0 / CONTROL_HZ
# 终端奖励权重除以 dt（IsaacLab 奖励 = func * weight * dt）。
SUCCESS_WEIGHT = 100.0 / DT  # 有效终端成功奖励 +100（高于 dense steady-state 一个量级）


##
# 场景
##
@configclass
class ChassisSceneCfg(InteractiveSceneCfg):
    """地面 + 灯 + 底盘 + 每个环境一个刚性目标立方体。"""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9)),
    )

    robot: ArticulationCfg = CHASSIS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    target: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.CuboidCfg(
            size=TARGET_SIZE,
            # 运动学刚体：悬浮静止，仅由重置事件直接放置；关闭碰撞（纯几何参考点，
            # 机器人可逼近/穿过，便于学习停靠环两侧的梯度）。
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, TARGET_HEIGHT)),
    )


##
# MDP
##
@configclass
class ActionsCfg:
    drive = mdp.DifferentialDriveActionCfg(
        asset_name="robot",
        left_wheel_joint=CHASSIS_PARAMS["left_wheel_joint"],
        right_wheel_joint=CHASSIS_PARAMS["right_wheel_joint"],
        wheel_radius=CHASSIS_PARAMS["wheel_radius"],
        wheel_base=CHASSIS_PARAMS["wheel_base"],
        v_range=V_RANGE,
        w_range=W_RANGE,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 1) 全局已知目标在基座系的平面位置 [dx_b, dy_b, dist_xy]（无相机、无损坏）。
        target = ObsTerm(
            func=mdp.target_in_robot_frame,
            params={"robot_name": "robot", "target_name": "target"},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        # 2) 本体感觉：前向线速度 + 偏航角速度。
        chassis_vel = ObsTerm(
            func=mdp.chassis_velocity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        # 3) 动作历史 a_{t-1}, a_{t-2}（抑制底盘抖动）。
        last_actions = ObsTerm(
            func=mdp.last_action,
            params={"action_name": "drive"},
            history_length=2,
            flatten_history_dim=True,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    # --- startup 物理域随机化 ---
    base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=CHASSIS_PARAMS["base_body"]),
            "mass_distribution_params": (0.8, 1.2),  # +/-20%
            "operation": "scale",
        },
    )
    wheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["link_left_wheel", "link_right_wheel"]),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # --- per-reset 域随机化 ---
    motor_strength = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_left_wheel", "joint_right_wheel"]),
            "stiffness_distribution_params": (1.0, 1.0),
            "damping_distribution_params": (0.8, 1.2),  # ~电机强度 +/-20%
            "operation": "scale",
        },
    )
    # 每个环境独立：随机底盘朝向 + 把目标放到周围可达环内（全局已知位置）。
    reset_scene = EventTerm(
        func=mdp.ResetChassisReach,
        mode="reset",
        params={
            "robot_name": "robot",
            "target_name": "target",
            "yaw_range": (-math.pi, math.pi),
            "xy_jitter": 0.2,
            "spawn_radius": SPAWN_RADIUS,
            "target_height": TARGET_HEIGHT,
        },
    )


@configclass
class RewardsCfg:
    # 密集前进/精停靠（单项）：锚定 spawn 基线的平滑指数势能 exp(-dist/lambda)-baseline。
    # iter 5 策略改变：取代 iter 3/4 的 progress + near_ring。
    #   * spawn 处≈0（无静态吸引子、无刷分带）、靠近>0、远离<0、处处可微、越近梯度越陡；
    #   * 非逐步差分 → 单步无物理抖动注入（修掉 progress 的高噪声梯度，PPO 难提梯度问题）；
    #   * 单调势能在停靠环附近自然加密 → 同时承担 near_ring 的精对环作用，无门控边界。
    approach = RewTerm(
        func=mdp.approach,
        weight=3.0,
        params={"lambda_dist": APPROACH_LAMBDA, "spawn_avg": APPROACH_SPAWN_AVG},
    )
    # 密集"对着开"：v_fwd * clamp(cos_bearing,0,1)，静止=0、面向且前进>0。
    # iter 3：耦合速度，机器人必须真的朝目标移动才拿奖励，直接塑造导航原语。
    # iter 5：weight 1.0→1.5，配合平滑 approach 势能强化"对着目标开"。
    heading = RewTerm(func=mdp.heading, weight=1.5)
    # 停住：接近停靠环时惩罚前向线速度（允许原地对准，不惩罚偏航）。
    stop = RewTerm(
        func=mdp.stop,
        weight=0.6,
        params={"d_standoff": D_STANDOFF, "dist_tol": DIST_TOL},
    )
    # 平滑：抑制动作突变（内置项）。
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # 稀疏终端成功。
    success = RewTerm(func=mdp.success_reward, weight=SUCCESS_WEIGHT)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    reach_outcome = DoneTerm(
        func=mdp.ReachOutcome,
        params={
            "d_standoff": D_STANDOFF,
            "dist_tol": DIST_TOL,
            "bearing_tol": BEARING_TOL,
            "speed_tol": SPEED_TOL,
            "success_dwell": SUCCESS_DWELL,
        },
    )


##
# Environment
##
@configclass
class ChassisReachEnvCfg(ManagerBasedRLEnvCfg):
    scene: ChassisSceneCfg = ChassisSceneCfg(num_envs=4096, env_spacing=6.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = int(round(PHYSICS_HZ / CONTROL_HZ))  # 10 物理步 / 控制步
        self.episode_length_s = EPISODE_S
        self.sim.dt = 1.0 / PHYSICS_HZ                          # 0.01 s
        self.sim.render_interval = self.decimation
        self.viewer.eye = (6.0, 6.0, 4.0)
        self.viewer.lookat = (2.0, 0.0, 0.5)


@configclass
class ChassisReachEnvCfg_PLAY(ChassisReachEnvCfg):
    """回放/评估用的轻量变体（少量环境，满难度）。"""

    scene: ChassisSceneCfg = ChassisSceneCfg(num_envs=16, env_spacing=6.0)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 6.0
