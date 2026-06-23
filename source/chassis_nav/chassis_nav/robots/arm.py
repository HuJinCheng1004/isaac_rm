# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""单右臂操控配置（底盘固定）。

两种配置：
  ARM_PUSH_CFG   — 6 DOF（r_joint1~r_joint6），夹爪 fixed，用于推方块等无抓取任务。
  ARM_GRASP_CFG  — 8 DOF（r_joint1~r_joint6 + r_Joint_finger1/2），用于抓取/提升任务。

底盘 fix_base=True。除上述可动关节外，所有关节（驱动轮/脚轮/升降杆/头部/左臂/左夹爪/
被动腕 r_joint7）在 URDF 变体中已被改为 type="fixed"。升降杆高度（0.534）已直接 bake
进 platform_joint 的 fixed origin（z: 0.271 → 0.805），不再依赖执行器顶住重力。

注意：未启用 merge_fixed_joints（保持 False），以保留 r_link8（末端执行器）与
r_Link_finger1/2 等被奖励/观测引用的具名刚体。

关节简表（来自 URDF）：
  r_joint1        : revolute   ±3.11 rad    shoulder yaw
  r_joint2        : revolute   ±2.27 rad    shoulder pitch
  r_joint3        : revolute   ±2.36 rad    elbow yaw
  r_joint4        : revolute   ±3.11 rad    elbow pitch
  r_joint5        : revolute   ±2.23 rad    wrist pitch
  r_joint6        : revolute   ±6.28 rad    wrist roll
  r_joint7        : revolute    0~0.79 rad  passive coupling (locked / fixed in URDF)
  r_Joint_finger1 : prismatic   0~0.0325 m  left finger  (axis -x, active in ARM_GRASP_CFG)
  r_Joint_finger2 : prismatic   0~0.0325 m  right finger (axis +x, active in ARM_GRASP_CFG)
末端执行器体: r_link8 (r_joint7 child, palm)
夹爪张开: finger pos = 0.0325 m；夹爪闭合: finger pos = 0.0 m
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

_ASSETS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "assets")
)

_URDF_PATH = os.path.join(
    _ASSETS_DIR,
    "overseas_65_b_v_description",
    "urdf",
    # 6 DOF only — 夹爪 fixed，用于推方块等无抓取任务
    "overseas_65_b_v_description_rmg24_armfixed.urdf",
)

_URDF_PATH_GRIPPER = os.path.join(
    _ASSETS_DIR,
    "overseas_65_b_v_description",
    "urdf",
    # 8 DOF — 夹爪 prismatic，用于抓取/提升任务
    "overseas_65_b_v_description_rmg24_armgripper_fixed.urdf",
)

# 末端执行器 body 名（r_joint7 → r_link8 为最后一个可动关节后的 palm 体）
EE_BODY_NAME = "r_link8"

# 活动关节正则
ARM_JOINT_NAMES = ["r_joint1", "r_joint2", "r_joint3", "r_joint4", "r_joint5", "r_joint6"]
GRIPPER_JOINT_NAMES = ["r_Joint_finger1", "r_Joint_finger2"]

# 夹爪位置常量（单位：m）
GRIPPER_OPEN = 0.0325
GRIPPER_CLOSED = 0.0

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
        rot=(0.7071, 0.0, 0.0, -0.7071),  # 绕 Z 轴顺时针 90°，使右臂朝向 -x（方块方向）
        # rot=(0.0, 0.0, 0.0, -0.7071),  # 绕 Z 轴顺时针 90°，使右臂朝向 -x（方块方向）
        # 仅右臂 6 DOF 仍是活动关节，其余已在 URDF 中 fixed（含升降杆，高度已
        # bake 进 origin）。来自 XRTeleop RIGHT_ARM_HOME_DEG = [90, 30, 70, 0, 70, 0]。
        joint_pos={
            "r_joint1": 1.5708,
            "r_joint2": 0.5236,
            "r_joint3": 1.2217,
            "r_joint4": 0.0,
            "r_joint5": 1.2217,
            "r_joint6": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 主动右臂（6 DOF），PD 位置控制
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["r_joint[1-6]"],
            stiffness=120.0,
            damping=6.0,
            effort_limit_sim=60.0,
            velocity_limit_sim=3.925,
        ),
        # 其余关节已在 URDF 中 type="fixed"（0 DOF），无需执行器锁定。
    },
    soft_joint_pos_limit_factor=0.95,
)
"""右臂单臂操控配置：底盘固定，仅 r_joint1~r_joint6 可动（6 DOF），其余 URDF 中已 fixed。"""


ARM_GRASP_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=_URDF_PATH_GRIPPER,
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
    actuators={
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=["r_joint[1-6]"],
            stiffness=120.0,
            damping=6.0,
            effort_limit_sim=60.0,
            velocity_limit_sim=3.925,
        ),
        "right_gripper": ImplicitActuatorCfg(
            joint_names_expr=["r_Joint_finger[12]"],
            stiffness=200.0,
            damping=10.0,
            effort_limit_sim=10.0,
            velocity_limit_sim=0.1,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
"""右臂抓取配置：底盘固定，r_joint1~r_joint6 + r_Joint_finger1/2 可动（8 DOF）。
夹爪默认张开（GRIPPER_OPEN = 0.0325 m），闭合时 pos = GRIPPER_CLOSED = 0.0 m。"""
