# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""底盘"到达目标点"任务的重置事件。

:class:`ResetChassisReach` 每次情节重置时，*每个环境独立地*：

* **底盘**：放回环境原点附近（小幅 xy 抖动）、随机偏航、速度清零。
* **目标方块**：放到环境原点周围、距离 ``spawn_radius`` 区间、随机方位角、固定世界
  高度 ``target_height`` 处（运动学刚体，悬浮静止）。

目标位置对策略是"全局已知"的（观测直接读其世界位姿，无相机/检测），因此无需像
``approach`` 任务那样保证落在相机视野内。机械臂/升降杆/头部保持默认（由各自执行器
锁定）。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ResetChassisReach(ManagerTermBase):
    """随机化底盘朝向并把目标放到底盘周围的可达环内（全局已知位置）。"""

    def __init__(self, cfg: EventTermCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        p = cfg.params
        self.robot_name: str = p.get("robot_name", "robot")
        self.target_name: str = p.get("target_name", "target")
        self._yaw_range = tuple(p.get("yaw_range", (-math.pi, math.pi)))
        self._xy_jitter = float(p.get("xy_jitter", 0.2))
        self._spawn_radius = tuple(p.get("spawn_radius", (1.5, 4.0)))
        self._target_height = float(p.get("target_height", 0.5))

    def __call__(
        self,
        env: "ManagerBasedEnv",
        env_ids: torch.Tensor,
        robot_name: str = "robot",
        target_name: str = "target",
        yaw_range: tuple = (-math.pi, math.pi),
        xy_jitter: float = 0.2,
        spawn_radius: tuple = (1.5, 4.0),
        target_height: float = 0.5,
    ) -> None:
        # 参数在 __init__ 中消费（存到 self）；此处列出仅为 2.3.2 的签名校验。
        robot = env.scene[self.robot_name]
        target = env.scene[self.target_name]
        n = len(env_ids)
        dev = self.device
        origins = env.scene.env_origins[env_ids]   # (n, 3)

        # --- 1) 底盘：原点附近 + 小幅 xy 抖动 + 随机偏航，速度清零 ---
        root = robot.data.default_root_state[env_ids].clone()  # (n, 13)
        pos = root[:, 0:3] + origins
        pos[:, 0] += math_utils.sample_uniform(-self._xy_jitter, self._xy_jitter, (n,), dev)
        pos[:, 1] += math_utils.sample_uniform(-self._xy_jitter, self._xy_jitter, (n,), dev)

        yaw = math_utils.sample_uniform(self._yaw_range[0], self._yaw_range[1], (n,), dev)
        zeros = torch.zeros(n, device=dev)
        dquat = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
        quat = math_utils.quat_mul(root[:, 3:7], dquat)

        robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
        robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)

        # --- 2) 目标：环境原点周围的可达环内，随机方位角，固定世界高度 ---
        r = math_utils.sample_uniform(self._spawn_radius[0], self._spawn_radius[1], (n,), dev)
        theta = math_utils.sample_uniform(-math.pi, math.pi, (n,), dev)
        tgt_pos = torch.zeros(n, 3, device=dev)
        tgt_pos[:, 0] = origins[:, 0] + r * torch.cos(theta)
        tgt_pos[:, 1] = origins[:, 1] + r * torch.sin(theta)
        tgt_pos[:, 2] = origins[:, 2] + self._target_height

        # 目标朝向：直立单位四元数（位置全局已知即可，朝向不参与任务）。
        tgt_quat = torch.zeros(n, 4, device=dev)
        tgt_quat[:, 0] = 1.0

        target.write_root_pose_to_sim(torch.cat([tgt_pos, tgt_quat], dim=-1), env_ids=env_ids)
        target.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)
