# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""差分驱动底盘的关节配置。

机器人是用户的 ``overseas_65_b_v_description`` 移动操纵臂。对于这个导航任务，
我们只*控制*两个驱动轮；脚轮自由滚动，每个上身关节（升降机/头部/双臂/夹子）
由一个刚性隐式执行器保持在其默认姿态，使平台表现为轮子上的刚体。

关键关节结构（来自 URDF）：
  * ``joint_left_wheel`` / ``joint_right_wheel``  : 连续驱动轮。
  * ``joint_swivel_wheel_[1-4]_[1-2]``            : 被动脚轮（转向 + 滚动）。
  * ``platform_joint``                            : 棱柱升降机。
  * ``head_joint1`` / ``head_joint2``             : 平移/倾斜头部。
  * ``camera_link``                               : RGB-D 摄像头（固定在头部）。
  * ``l_joint[1-7]`` / ``r_joint[1-7]``           : 双臂。
  * ``*_Joint_finger[1-2]``                       : 夹子。
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# --------------------------------------------------------------------------- #
# 动作项和边界框观察重用的物理常数。
# 在部署到硬件前，根据真实机器人/网格验证这些。
# --------------------------------------------------------------------------- #

# URDF 相对于本文件：../../../../assets/overseas_65_b_v_description/urdf/...
# 路径：robots/ -> chassis_nav/ -> chassis_nav/ -> source/ -> isaac_rm/ -> assets/
_ASSETS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "assets")
)

CHASSIS_PARAMS = {
    # URDF 路径（URDF 内的网格使用绝对 file:// 路径，因此
    # 完全离线解析）。
    "urdf_path": os.path.join(
        _ASSETS_DIR,
        "overseas_65_b_v_description",
        "urdf",
        "overseas_65_b_v_description_rmg24.urdf",
    ),
    # 驱动几何。轮距直接来自 URDF（轮子在
    # x = +/-0.148 m）。轮半径是估计 -- 从网格/
    # 硬件确认。错误的半径只会重新缩放动作映射，PPO
    # 会适应，因为策略也观察到*测量的*底盘速度。
    "wheel_base": 0.296,      # [m] 左<->右轮距离
    "wheel_radius": 0.075,    # [m] 驱动轮滚动半径（估计）
    # 其世界姿态定义虚拟摄像头的主体。使用
    # merge_fixed_joints=False，此链接在导入时幸存。如果你的导入器
    # 仍然合并它，将其设置为 "head_link2" 并在
    # bbox 观察配置中添加固定偏移。
    "camera_body": "camera_link",
    # 根/基体名称（URDF 根链接）。
    "base_body": "base_link_underpan",
    # 驱动轮关节名称。
    "left_wheel_joint": "joint_left_wheel",
    "right_wheel_joint": "joint_right_wheel",
}


CHASSIS_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=CHASSIS_PARAMS["urdf_path"],
        fix_base=False,
        # 将固定关节子级（特别是 ``camera_link``）保持为单独的主体
        # 以便观察可以直接读取摄像头姿态。
        merge_fixed_joints=False,
        # 从视觉网格建立碰撞（URDF 没有单独的
        # <collision> 原语可用于所有链接）。
        collision_from_visuals=True,
        self_collision=False,
        # 全局默认驱动；由下面的 ``actuators`` 按组覆盖。
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=10.0),
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_linear_velocity=10.0,
            max_angular_velocity=100.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),
        rot=(1.0, 0.0, 0.0, 0.0),  # identity, (w, x, y, z) convention in Isaac Lab 2.3.2
        joint_pos={
            "joint_left_wheel": 0.0,
            "joint_right_wheel": 0.0,
            "joint_swivel_wheel_.*": 0.0,
            "platform_joint": 0.0,
            "head_joint1": 0.0,
            "head_joint2": 0.0,
            "l_joint.*": 0.0,
            "r_joint.*": 0.0,
            ".*_Joint_finger.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 驱动轮：纯速度控制（刚度=0，阻尼是
        # 速度增益）。``effort_limit`` 是电机转矩上限，是一个
        # 域随机化目标（见 EventsCfg）。
        "drive_wheels": ImplicitActuatorCfg(
            joint_names_expr=["joint_left_wheel", "joint_right_wheel"],
            stiffness=0.0,
            damping=50.0,
            effort_limit_sim=20.0,
            velocity_limit_sim=50.0,
        ),
        # 被动脚轮：自由滚动/转向（无驱动）。
        "casters": ImplicitActuatorCfg(
            joint_names_expr=["joint_swivel_wheel_.*"],
            stiffness=0.0,
            damping=0.0,
            effort_limit_sim=0.0,
        ),
        # 基础以上的所有内容都在其默认姿态被严格保持。
        "upper_body": ImplicitActuatorCfg(
            joint_names_expr=[
                "platform_joint",
                "head_joint.*",
                "l_joint.*",
                "r_joint.*",
                ".*_Joint_finger.*",
            ],
            stiffness=2000.0,
            damping=100.0,
            effort_limit_sim=1000.0,
        ),
    },
)
"""差分驱动底盘关节（轮子驱动，上身锁定）。"""
