# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""2-D 差分驱动动作项（仅前向速度 + 偏航速率，无升降杆）。

``reach`` 任务无相机，不需要升降杆调节相机高度，因此动作降为纯 2-D：

    a = (a_v, a_w) in [-1, 1]^2
    v_cmd = affine(a_v, v_range)        # 前向速度 [m/s]
    w_cmd = affine(a_w, w_range)        # 偏航速率 [rad/s]
    v_left  = v_cmd - w_cmd * wheel_base / 2
    v_right = v_cmd + w_cmd * wheel_base / 2
    q_dot_left  = left_sign  * v_left  / wheel_radius
    q_dot_right = right_sign * v_right / wheel_radius

升降杆与上身关节由各自的隐式执行器保持在默认姿态（见 chassis.py），本动作项只
驱动两个驱动轮。``planar_lock`` 每物理子步把底盘约束在 SE(2)，防止高重心上身倾倒。
逻辑镜像 ``approach`` 任务的 3-D 版本，去掉第 3 维升降杆。
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

        left_ids, left_names = self._asset.find_joints(self.cfg.left_wheel_joint)
        right_ids, right_names = self._asset.find_joints(self.cfg.right_wheel_joint)
        if len(left_ids) != 1:
            raise ValueError(f"Expected exactly one left wheel joint, got {left_names}")
        if len(right_ids) != 1:
            raise ValueError(f"Expected exactly one right wheel joint, got {right_names}")
        self._wheel_ids = [left_ids[0], right_ids[0]]
        self._wheel_names = [left_names[0], right_names[0]]

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._wheel_vel_target = torch.zeros(self.num_envs, 2, device=self.device)

        self._r = float(self.cfg.wheel_radius)
        self._half_base = 0.5 * float(self.cfg.wheel_base)
        self._left_sign = float(self.cfg.left_sign)
        self._right_sign = float(self.cfg.right_sign)
        self._v_center = 0.5 * (self.cfg.v_range[1] + self.cfg.v_range[0])
        self._v_span = 0.5 * (self.cfg.v_range[1] - self.cfg.v_range[0])
        self._w_center = 0.5 * (self.cfg.w_range[1] + self.cfg.w_range[0])
        self._w_span = 0.5 * (self.cfg.w_range[1] - self.cfg.w_range[0])

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        a = torch.clamp(actions, -1.0, 1.0)
        self._processed_actions[:, 0] = self._v_center + self._v_span * a[:, 0]
        self._processed_actions[:, 1] = self._w_center + self._w_span * a[:, 1]

    def apply_actions(self):
        v = self._processed_actions[:, 0]
        w = self._processed_actions[:, 1]
        v_left = v - w * self._half_base
        v_right = v + w * self._half_base
        self._wheel_vel_target[:, 0] = self._left_sign * v_left / self._r
        self._wheel_vel_target[:, 1] = self._right_sign * v_right / self._r
        self._asset.set_joint_velocity_target(self._wheel_vel_target, joint_ids=self._wheel_ids)
        if self.cfg.planar_lock:
            self._lock_to_plane()

    def _lock_to_plane(self):
        """把底盘根姿态投影到 SE(2)：仅保留偏航，归零横滚/俯仰及其角速度。"""
        data = self._asset.data
        upright = math_utils.yaw_quat(data.root_quat_w)
        pose = torch.cat([data.root_pos_w, upright], dim=-1)
        lin = data.root_lin_vel_w
        ang = data.root_ang_vel_w.clone()
        ang[:, 0:2] = 0.0
        vel = torch.cat([lin, ang], dim=-1)
        pose = torch.nan_to_num(pose, nan=0.0, posinf=0.0, neginf=0.0)
        vel = torch.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
        self._asset.write_root_pose_to_sim(pose)
        self._asset.write_root_velocity_to_sim(vel)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        idx = slice(None) if env_ids is None else env_ids
        self._raw_actions[idx] = 0.0
        self._processed_actions[idx] = 0.0


@configclass
class DifferentialDriveActionCfg(ActionTermCfg):
    """:class:`DifferentialDriveAction`（2-D）的配置。"""

    class_type: type[ActionTerm] = DifferentialDriveAction

    left_wheel_joint: str = "joint_left_wheel"
    right_wheel_joint: str = "joint_right_wheel"

    wheel_radius: float = 0.075
    wheel_base: float = 0.296

    v_range: tuple[float, float] = (-0.2, 0.5)
    w_range: tuple[float, float] = (-1.0, 1.0)

    left_sign: float = -1.0
    right_sign: float = 1.0

    planar_lock: bool = True
