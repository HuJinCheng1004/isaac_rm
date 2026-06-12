# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""差分驱动动作项。

将 2-D 归一化动作 ``a = (a_v, a_w) in [-1, 1]^2`` 映射到底盘的两个驱动
轮关节速度目标：

    v_cmd = affine_map(a_v, v_range)        # 前向速度   [m/s]
    w_cmd = affine_map(a_w, w_range)        # 偏航速率   [rad/s]

    v_left_contact  = v_cmd - w_cmd * wheel_base / 2
    v_right_contact = v_cmd + w_cmd * wheel_base / 2

    q_dot_left  = left_sign  * v_left_contact  / wheel_radius
    q_dot_right = right_sign * v_right_contact / wheel_radius

``*_sign`` 因子吸收 URDF 关节轴约定（左轮
轴是 ``-x``，而右轮是 ``+x``）。默认值假设正命令
使机器人向前驱动；如果你的机器人旋转或向后驱动，翻转
相关的符号（见 README "调整"部分）。

结构镜像 :class:`isaaclab.envs.mdp.actions.NonHolonomicAction`。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets.articulation import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DifferentialDriveAction(ActionTerm):
    """2-D（前向速度，偏航速率）动作映射到两个驱动轮速度。"""

    cfg: "DifferentialDriveActionCfg"
    _asset: Articulation

    def __init__(self, cfg: "DifferentialDriveActionCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

        # 解析两个驱动轮关节（每个恰好一个）
        left_ids, left_names = self._asset.find_joints(self.cfg.left_wheel_joint)
        right_ids, right_names = self._asset.find_joints(self.cfg.right_wheel_joint)
        if len(left_ids) != 1:
            raise ValueError(f"Expected exactly one left wheel joint, got {left_names}")
        if len(right_ids) != 1:
            raise ValueError(f"Expected exactly one right wheel joint, got {right_names}")
        self._wheel_ids = [left_ids[0], right_ids[0]]
        self._wheel_names = [left_names[0], right_names[0]]

        # 解析升降杆（棱柱）关节——第 3 维动作控制它的高度。
        lift_ids, lift_names = self._asset.find_joints(self.cfg.lift_joint)
        if len(lift_ids) != 1:
            raise ValueError(f"Expected exactly one lift joint, got {lift_names}")
        self._lift_id = lift_ids[0]
        self._lift_min = float(self.cfg.lift_range[0])
        self._lift_max = float(self.cfg.lift_range[1])
        self._lift_speed = float(self.cfg.lift_speed)
        self._dt = float(self._env.step_dt)  # 控制步长 [s]（decimation * 物理 dt）

        # 缓冲区
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._wheel_vel_target = torch.zeros(self.num_envs, 2, device=self.device)
        # 升降杆位置目标（每控制步按速度积分得到，初值随后在 reset 中同步到实际高度）。
        self._lift_target = torch.zeros(self.num_envs, device=self.device)

        # 缓存的标量
        self._r = float(self.cfg.wheel_radius)
        self._half_base = 0.5 * float(self.cfg.wheel_base)
        self._left_sign = float(self.cfg.left_sign)
        self._right_sign = float(self.cfg.right_sign)
        # 仿射映射 [-1, 1] -> 范围：y = center + half_span * a
        self._v_center = 0.5 * (self.cfg.v_range[1] + self.cfg.v_range[0])
        self._v_span = 0.5 * (self.cfg.v_range[1] - self.cfg.v_range[0])
        self._w_center = 0.5 * (self.cfg.w_range[1] + self.cfg.w_range[0])
        self._w_span = 0.5 * (self.cfg.w_range[1] - self.cfg.w_range[0])

    # -- 属性 ---------------------------------------------------------- #
    @property
    def action_dim(self) -> int:
        return 3

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """映射后的物理命令 ``(v_cmd [m/s], w_cmd [rad/s])``。"""
        return self._processed_actions

    # -- 操作 ---------------------------------------------------------- #
    def process_actions(self, actions: torch.Tensor):
        # 存储原始，夹到有效策略范围，然后仿射映射到物理
        self._raw_actions[:] = actions
        a = torch.clamp(actions, -1.0, 1.0)
        self._processed_actions[:, 0] = self._v_center + self._v_span * a[:, 0]
        self._processed_actions[:, 1] = self._w_center + self._w_span * a[:, 1]
        # 第 3 维：升降杆速度命令 [m/s]。按控制步长积分到位置目标（速率受限，
        # 模拟真实升降机），并裁剪到行程范围。从*实际*当前高度积分，因此重置后
        # 自然接续，无需显式同步。
        lift_vel = a[:, 2] * self._lift_speed
        self._processed_actions[:, 2] = lift_vel
        current_lift = self._asset.data.joint_pos[:, self._lift_id]
        self._lift_target = torch.clamp(
            current_lift + lift_vel * self._dt, self._lift_min, self._lift_max
        )

    def apply_actions(self):
        v = self._processed_actions[:, 0]
        w = self._processed_actions[:, 1]
        v_left = v - w * self._half_base
        v_right = v + w * self._half_base
        self._wheel_vel_target[:, 0] = self._left_sign * v_left / self._r
        self._wheel_vel_target[:, 1] = self._right_sign * v_right / self._r
        self._asset.set_joint_velocity_target(self._wheel_vel_target, joint_ids=self._wheel_ids)
        # 升降杆位置目标（由上身高刚度执行器跟踪）。
        self._asset.set_joint_position_target(self._lift_target.unsqueeze(-1), joint_ids=[self._lift_id])
        # 平面约束：底盘只能在 xy 平面平移 + 绕 z 偏航；强制清除横滚/俯仰，
        # 防止高重心上身在急转/加速时把底盘掀翻或来回摇晃。
        if self.cfg.planar_lock:
            self._lock_to_plane()

    def _lock_to_plane(self):
        """把底盘根姿态投影到 SE(2)：仅保留偏航，归零横滚/俯仰及其角速度。

        每个物理子步执行一次（``apply_actions`` 在 decimation 循环内被调用），
        因此倾倒无法累积。xy 平移、偏航、以及竖直方向的自然落地动力学不受影响。
        """
        data = self._asset.data
        # 姿态：丢弃横滚/俯仰，只留偏航（位置原样保留，含自然落地高度）。
        upright = math_utils.yaw_quat(data.root_quat_w)
        self._asset.write_root_pose_to_sim(torch.cat([data.root_pos_w, upright], dim=-1))
        # 速度：清零横滚/俯仰角速度（线速度与偏航角速度保留）。
        lin = data.root_lin_vel_w
        ang = data.root_ang_vel_w.clone()
        ang[:, 0:2] = 0.0
        self._asset.write_root_velocity_to_sim(torch.cat([lin, ang], dim=-1))

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        # 把升降杆位置目标同步到重置事件刚设定的实际高度（事件重置先于动作重置执行），
        # 避免重置后第一步把升降杆拽回旧目标。
        idx = slice(None) if env_ids is None else env_ids
        self._raw_actions[idx] = 0.0
        self._processed_actions[idx] = 0.0
        self._lift_target[idx] = self._asset.data.joint_pos[idx, self._lift_id]


@configclass
class DifferentialDriveActionCfg(ActionTermCfg):
    """:class:`DifferentialDriveAction` 的配置。"""

    class_type: type[ActionTerm] = DifferentialDriveAction

    left_wheel_joint: str = "joint_left_wheel"
    right_wheel_joint: str = "joint_right_wheel"
    lift_joint: str = "platform_joint"
    """升降杆棱柱关节名（第 3 维动作控制其高度）。"""

    wheel_radius: float = 0.075
    """驱动轮滚动半径 [m]。见 ``CHASSIS_PARAMS``。"""
    wheel_base: float = 0.296
    """左<->右轮距离 [m]。"""

    v_range: tuple[float, float] = (-0.2, 0.5)
    """a_v in [-1, 1] 映射到的物理前向速度范围 [m/s]。"""
    w_range: tuple[float, float] = (-1.0, 1.0)
    """a_w in [-1, 1] 映射到的物理偏航速率范围 [rad/s]。"""

    lift_range: tuple[float, float] = (0.0, 1.0)
    """升降杆高度行程 [m]（动作积分后裁剪到此范围；URDF 限位 0~1）。"""
    lift_speed: float = 0.15
    """a_lift in [-1, 1] 映射到的升降杆最大速度 [m/s]（匹配 URDF 速度限位）。"""

    left_sign: float = -1.0
    """补偿左轮 URDF 轴 (-x) 的符号。如需要请翻转。"""
    right_sign: float = 1.0
    """补偿右轮 URDF 轴 (+x) 的符号。如需要请翻转。"""

    planar_lock: bool = True
    """每个物理子步把底盘约束在 xy 平面（清除横滚/俯仰），防止高重心上身倾倒。"""
