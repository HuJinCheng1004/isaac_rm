# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""底盘靠近/对齐任务的基于管理器的 RL 环境配置。

无渲染：目标边界框从姿态中解析计算，因此
数千个环境在没有任何摄像头光栅化的情况下运行。

所有任务级旋钮都位于下面的 ``# --- 可调参数 ---`` 块中。
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from chassis_nav.robots.chassis import CHASSIS_CFG, CHASSIS_PARAMS

from . import mdp

# --------------------------------------------------------------------------- #
# --- 可调参数 --------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
CONTROL_HZ = 10.0           # 策略/控制速率（匹配 10 Hz 视觉管道）
PHYSICS_HZ = 100.0          # 物理求解器速率
EPISODE_S = 15.0            # 情节长度 [s]

V_RANGE = (-0.2, 0.5)       # 前向速度映射 [m/s]
W_RANGE = (-1.0, 1.0)       # 偏航速率映射 [rad/s]

CAM_HFOV = 1.204            # 水平 FOV [rad] (~69 度)
CAM_VFOV = 0.75             # 垂直 FOV [rad] (~43 度)
TARGET_SIZE = (0.25, 0.25, 0.40)            # 目标立方体完整大小 [m]（= 3D 框边长）

# 头部相机固定在真机 home 位姿：HEAD_HOME_PITCH=400 / HEAD_HOME_YAW=500（舵机原始值）。
# 舵机标定（XRTeleop/run_realman_pose_teleop）：home servo 500 = 关节 0 rad；
# pitch 220 units/rad、yaw 120 units/rad。换算得：
#   head_joint2(pitch) = (400-500)/220 = -0.4545 rad ≈ -26°（向下）
#   head_joint1(yaw)   = (500-500)/120 = 0 rad
# 本模型相机光轴定义为 +x、而 head_joint2 轴沿 +x（仅绕光轴滚转、改不了视线方向），
# 因此把 26° 下倾角直接“烤进”相机光轴（同一组轴同时用于 BBox3DObservation 与目标放置，
# 保证一致），物理头部关节保持默认 home（yaw=0）。
HEAD_PITCH_DOWN = (500.0 - 400.0) / 220.0   # ≈ 0.4545 rad，相机光轴下倾角
_CP, _SP = math.cos(HEAD_PITCH_DOWN), math.sin(HEAD_PITCH_DOWN)
CAM_AXES = {                                # 相机局部光轴（camera_link 系，+x前/+y左/+z上）
    "forward": (_CP, 0.0, -_SP),            # 前向：水平前方下压 26°
    "right": (0.0, -1.0, 0.0),              # 右：图像 +u
    "up": (_SP, 0.0, _CP),                  # 上：图像 +v
}

D_TARGET = 0.4             # 理想 3D 距离（相机<->目标中心，"可操作距离"）[m]
CENTER_K = 2.0              # 居中奖励锐度
NEAR_PENALTY = 2.0          # approach 近端惩罚斜率：distance < D_TARGET 时的负惩罚强度
                           # （由 4.0 降低：成功区下界已到 0.15 m，过陡惩罚会把策略推离成功区）
# 成功条件：配合“按高度放置 + 容差放宽”，让 [0.4,1.6] m 高度区间内的目标几何可达
# （蒙特卡洛验证：cam0≈0.6 时约 96% 目标“居中+距离”成功区可达）。
CENTER_TOL = 0.30           # 居中成功容差：|中心投影| (ndc)  —— 由 0.25 放宽
DIST_TOL = 0.25            # 距离成功容差：|distance - D_target| [m]（成功区 [0.15, 0.65] m）
SUCCESS_DWELL = 5           # 成功条件必须保持的步数（0.5 s @ 10 Hz）—— 由 10 放宽
# 丢失视野连续 LOST_DWELL 步才判失败。由 1->3->5（0.5 s 宽限）：失败仍是主导终止
# （旧 run 健康期 success≈0、回合几乎全靠"丢失"结束），再放宽一点宽限给策略找回
# 视野的机会，缓解"成功难、失败易"的不对称。避免观测丢帧/噪声导致 visible 抖一帧
# 就触发"假失败"，逼策略不敢靠近。
LOST_DWELL = 5

# Sim-to-real 损坏（BBox3DObservation）：观察 (x,y,z,distance,sx,sy,sz,u_ndc,v_ndc)，
# 前 7 维米制标准差（深度/距离通道更大，模拟 RealSense 深度噪声），后 2 维为图像
# 归一化坐标噪声（2D 检测框中心，较准）。
BBOX_NOISE_STD = (0.02, 0.02, 0.04, 0.04, 0.02, 0.02, 0.03, 0.02, 0.02)
DROPOUT_PROB = 0.05                          # 每步假阴性概率
LATENCY_STEPS = (1, 2)                       # 观察延迟范围 [t-1, t-2]

# 终端奖励权重除以 dt（奖励 = func * weight * dt）。
# 对于 dt = 0.1 s：权重 1000 -> +100 有效；权重 -200 -> -20 有效。
# 失败有效惩罚由 -50 下调到 -20，配合 LOST_DWELL 放宽与 approach 正向化，缓解
# "成功难、失败易"的不对称，让策略敢于冒险靠近。
DT = 1.0 / CONTROL_HZ
SUCCESS_WEIGHT = 100.0 / DT
FAILURE_WEIGHT = -15.0 / DT  # 由 -20 上调（更不负）：配合可达放置，减小“靠近=丢失”的过度恐惧

# 目标中心**世界高度**区间 [m]（用户设定 0.4~1.2）。该上限 1.2 m 远低于几何可成功上限
# (~1.5 m，受“升降杆行程 1 m + 固定 26° 下倾”所限)，故区间内目标全部“居中+距离”成功区
# 几何可达，reset 不会触发高度夹断。要支持更高目标需更长升降行程（见 chassis URDF 限位）。
HEIGHT_RANGE = (0.4, 1.2)

# --- 课程（curriculum）：随训练把重置难度从"近、正前方、舒适高度"线性扩展到完整范围 ---
CURRICULUM_STEPS = 30000    # 在前 N 个训练步内线性退火到满难度（与 trainer.timesteps 同单位）
YAW_HALF_START = 0.4        # 初始偏航半幅 [rad]（≈±23°，目标更贴正前方，更易保持在窄 FOV 内）
HEIGHT_START = (0.6, 1.0)   # 初始目标高度区间 [m]（集中在相机舒适带，先制造成功经验）


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
            # 目标是**运动学刚体**：悬浮在 3D 空间任意位置，仅由重置事件直接放置，
            # 在情节内永远静止不动（不受重力、碰撞或接触力影响），保证 bbox 稳定。
            # 关闭碰撞：感知是解析的（不依赖物理接触），且碰撞终止已移除——目标设为
            # 纯感知参考，机器人可逼近/穿过它，从而能探索并学习 <D_TARGET 的软惩罚梯度。
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True, kinematic_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.1, 0.1)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.0, 0.0, 1.0)),
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
        # 升降杆行程与 reset 放置/几何一致（满行程，相机最高 ≈ 1.6 m）：策略需边靠近边
        # 降低相机以在 26° 下倾 FOV 内保持低目标居中。
        lift_range=(0.0, 1.0),
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # 1) analytic target obs (x, y, z, distance, sx, sy, sz, u_ndc, v_ndc) -- corrupted.
        #    相机系 3D 框（镜像 run_realtime.py 的 Box3D）+ 图像归一化中心 (u_ndc, v_ndc)，
        #    后者与 centering 奖励/成功判据同口径，直接给策略居中角度信号。
        bbox = ObsTerm(
            func=mdp.BBox3DObservation,
            params={
                "robot_name": "robot",
                "target_name": "target",
                "camera_body": CHASSIS_PARAMS["camera_body"],
                "hfov": CAM_HFOV,
                "vfov": CAM_VFOV,
                "camera_axes": CAM_AXES,
                "target_size": TARGET_SIZE,
                "noise_std": BBOX_NOISE_STD,
                "dropout_prob": DROPOUT_PROB,
                "latency_steps": LATENCY_STEPS,
            },
        )
        # 2) proprioception: forward speed + yaw rate.
        chassis_vel = ObsTerm(
            func=mdp.chassis_velocity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        # 2b) proprioception: 升降杆当前高度（策略用第 3 维动作调节它来控制 cy）。
        lift_height = ObsTerm(
            func=mdp.lift_height,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["platform_joint"])},
            noise=Unoise(n_min=-0.005, n_max=0.005),
        )
        # 3) action history a_{t-1}, a_{t-2} (greatly reduces base "twitching").
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
    # --- startup physics DR ---
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

    # --- per-reset DR ---
    motor_strength = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint_left_wheel", "joint_right_wheel"]),
            "stiffness_distribution_params": (1.0, 1.0),
            "damping_distribution_params": (0.8, 1.2),  # ~motor strength +/-20%
            "operation": "scale",
        },
    )
    # 每个环境独立随机化机器人状态（底盘朝向 + 升降杆高度），并把目标方块放进
    # 相机视野，保证初始帧目标 bbox 落在图像内。机械臂保持默认（不随机）。
    reset_in_view = EventTerm(
        func=mdp.ResetRobotTargetInView,
        mode="reset",
        params={
            "robot_name": "robot",
            "target_name": "target",
            "camera_body": CHASSIS_PARAMS["camera_body"],
            "lift_joint": "platform_joint",
            # 底盘：全向偏航 + 小幅平移抖动
            "yaw_range": (-math.pi, math.pi),
            "xy_jitter": 0.2,
            # 目标放置：直接采样**世界高度**（而非沿下倾视线采深度），保证“居中+距离”
            # 成功区几何可达；并自动把相机抬到目标上方 Δ 处保证初始可见。
            "height_range": HEIGHT_RANGE,
            "delta_range": (0.35, 0.75),       # 相机高于目标的余量 Δ（决定初始俯视角/距离）
            # 相机针孔模型（必须与 BBox3DObservation 一致，含 26° 下倾光轴）
            "hfov": CAM_HFOV,
            "vfov": CAM_VFOV,
            "camera_axes": CAM_AXES,
            # 画面裕度：目标初始更靠画面中心（远离 FOV 边缘），起手不在"快丢失"边缘。
            "margin_h": 0.5,
            "margin_v": 0.4,
            "lift_limits": (0.0, 1.0),         # platform_joint 行程（与动作 lift_range 一致）
            "frame_margin": 0.15,              # 相机需高于目标的最小余量（保证落在下倾 FOV 内）
        },
    )


@configclass
class RewardsCfg:
    centering = RewTerm(func=mdp.centering, weight=1.0, params={"k": CENTER_K})
    # approach 现为单峰正向塑形（峰值 +1 @ d_target）。提高权重到 2.0，强化"靠近"密集
    # 信号——诊断显示旧策略只学会居中、不肯靠近。
    approach = RewTerm(
        func=mdp.approach,
        weight=2.0,
        params={"d_target": D_TARGET, "near_penalty": NEAR_PENALTY},
    )
    # 抑制底盘(前向/偏航)与升降杆速度指令的加速度（急加减速/抖动）。
    smoothness = RewTerm(
        func=mdp.smoothness_penalty,
        weight=-0.05,
        params={"base_weight": 1.0, "lift_weight": 1.0},
    )
    success = RewTerm(func=mdp.success_reward, weight=SUCCESS_WEIGHT)
    failure = RewTerm(func=mdp.failure_penalty, weight=FAILURE_WEIGHT)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    task_outcome = DoneTerm(
        func=mdp.TaskOutcome,
        params={
            "robot_name": "robot",
            "target_name": "target",
            "d_target": D_TARGET,
            "center_tol": CENTER_TOL,
            "dist_tol": DIST_TOL,
            "success_dwell": SUCCESS_DWELL,
            "lost_dwell": LOST_DWELL,
        },
    )


@configclass
class CurriculumCfg:
    """随训练线性扩展 ``reset_in_view`` 的初始难度（偏航半幅 + 目标深度上限）。

    早期目标近、基本在正前方 -> 易制造成功经验；``CURRICULUM_STEPS`` 步后退火到
    ``reset_in_view`` 配置的完整 ``yaw_range`` / ``d_range``。课程项直接修改重置事件
    实例缓存的激活范围（见 :func:`mdp.approach_difficulty`）。
    """

    approach_difficulty = CurrTerm(
        func=mdp.approach_difficulty,
        params={
            "num_steps": CURRICULUM_STEPS,
            "yaw_start": YAW_HALF_START,
            "height_start": HEIGHT_START,
        },
    )


##
# Environment
##
@configclass
class ChassisApproachEnvCfg(ManagerBasedRLEnvCfg):
    scene: ChassisSceneCfg = ChassisSceneCfg(num_envs=4096, env_spacing=6.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = int(round(PHYSICS_HZ / CONTROL_HZ))  # 10 physics steps / control step
        self.episode_length_s = EPISODE_S
        self.sim.dt = 1.0 / PHYSICS_HZ                          # 0.01 s
        self.sim.render_interval = self.decimation
        # camera looking down the +x axis; nice default debug viewpoint
        self.viewer.eye = (6.0, 6.0, 4.0)
        self.viewer.lookat = (2.0, 0.0, 0.5)


@configclass
class ChassisSceneCfg_PLAY(ChassisSceneCfg):
    """PLAY 场景：在 camera_link 上额外挂一个第一视角调试相机。

    仅用于可视化（在 Isaac Sim 视角下拉里可选到它，看机器人"眼睛"看到的画面）；
    它**不进观测**——策略的输入仍是解析 bbox。其光轴下倾 26°、约定为 ``world``
    (forward=+X, up=+Z)，与 :data:`CAM_AXES` 完全一致，因此画面中心即 bbox 的
    ``(cx, cy)`` 参考。正式训练用的 :class:`ChassisSceneCfg` 不含相机（无渲染开销）。
    """

    front_cam: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/camera_link/front_cam",
        update_period=1.0 / CONTROL_HZ,     # 10 Hz
        height=368,
        width=640,                           # 宽高比 ≈ tan(HFOV/2)/tan(VFOV/2)，匹配解析 FOV
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            horizontal_aperture=32.8,        # -> HFOV ≈ 69°（VFOV 随宽高比 ≈ 43°）
            clipping_range=(0.05, 30.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(0.9742, 0.0, 0.2253, 0.0),  # 绕 +Y 下倾 26°（= HEAD_PITCH_DOWN），对齐 CAM_AXES
            convention="world",
        ),
    )


@configclass
class ChassisApproachEnvCfg_PLAY(ChassisApproachEnvCfg):
    """Lightweight variant for visualization / policy playback."""

    # 用带调试相机的 PLAY 场景（覆盖父类的无相机场景）。
    scene: ChassisSceneCfg_PLAY = ChassisSceneCfg_PLAY(num_envs=16, env_spacing=6.0)

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 6.0
        # 回放/评估按满难度进行（关闭课程，否则 common_step_counter=0 会一直停在最易档）。
        self.curriculum = None
        # keep corruption on so playback reflects the real (noisy/delayed) pipeline.
        # 观测维度与训练保持一致（不再覆盖 last_actions.history_length）：
        #   bbox(9) + chassis_vel(2) + lift(1) + last_actions(2*3=6) = 18 维。
