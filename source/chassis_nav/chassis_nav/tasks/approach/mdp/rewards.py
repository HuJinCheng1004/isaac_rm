# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""底盘靠近/对齐任务的奖励项。

所有塑形奖励读取*干净的*地真边界框（不是噪声/
延迟的策略观察），并由目标可见性门控，因此瞬间
假阴性检测永远不会错误地奖励"丢失"行为。

    R = w1 * R_center + w2 * R_approach + w3 * R_smooth + R_terminal

* ``centering``       : exp(-k * sqrt(u_ndc^2 + v_ndc^2))     （保持目标居中）
* ``approach``        : -|distance - d_target|                （达到可操作距离 [m]）
* smoothness          : 使用内置 ``mdp.action_rate_l2`` (-||a_t - a_{t-1}||^2)
* ``success_reward``  : +1 指示器 -> 通过权重缩放到 ~ +100
* ``failure_penalty`` : +1 指示器 -> 通过权重缩放到 ~ -50

注意：奖励管理器将每个项乘以 ``weight * dt``。对于 10 Hz
控制速率（dt = 0.1 s），终端权重因此除以 dt
（例如权重 1000 -> 有效 +100）。见环境配置 / README。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .observations import get_bbox3d_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def centering(env: "ManagerBasedRLEnv", k: float = 2.0) -> torch.Tensor:
    """exp(-k * 径向投影偏移），当目标不可见时为零。

    径向偏移由 3D 中心投影到图像的 ``(u_ndc, v_ndc)`` 计算，与原 2D 居中等价
    （u/v_ndc 即归一化方位角），驱动底盘偏航对正水平、升降杆对正纵向。
    """
    s = get_bbox3d_state(env)
    radial = torch.sqrt(s.u_ndc ** 2 + s.v_ndc ** 2)
    return torch.exp(-k * radial) * s.visible.float()


def approach(env: "ManagerBasedRLEnv", d_target: float = 0.8) -> torch.Tensor:
    """-|distance - d_target|，当目标不可见时为零。

    ``distance`` 是目标 3D 中心到相机原点的米制距离（run_realtime 的
    ``Box3D.distance``），驱动底盘把目标靠近到可操作距离 ``d_target`` [m]。
    """
    s = get_bbox3d_state(env)
    return -(s.distance - d_target).abs() * s.visible.float()


def success_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """+1 在满足成功条件的步骤上（由 TaskOutcome 项设置）。"""
    flag = getattr(env, "_chassis_success", None)
    if flag is None:
        return torch.zeros(env.num_envs, device=env.device)
    return flag.float()


def failure_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """+1 在发生碰撞或丢失目标失败的步骤上（使用负权重）。"""
    flag = getattr(env, "_chassis_failure", None)
    if flag is None:
        return torch.zeros(env.num_envs, device=env.device)
    return flag.float()
