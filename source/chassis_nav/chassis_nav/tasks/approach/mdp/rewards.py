# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""底盘靠近/对齐任务的奖励项。

所有塑形奖励读取*干净的*地真边界框（不是噪声/延迟的策略观察）。密集奖励
**不再**按目标可见性门控——丢失视野由终止项判为失败（立即结束情节），因此
不需要靠"密集奖励归零"来抑制丢失行为。

    R = w1 * R_center + w2 * R_approach + w3 * R_smooth + R_terminal

* ``centering``         : exp(-k * sqrt(u_ndc^2 + v_ndc^2))   （保持目标居中）
* ``approach``          : 越靠近 d_target 奖励越高；近于 d_target 则转为惩罚
* ``smoothness_penalty``: 惩罚底盘(v,w)与升降杆速度指令的逐步变化（≈加速度）
* ``success_reward``    : +1 指示器 -> 通过权重缩放到 ~ +100
* ``failure_penalty``   : +1 指示器 -> 通过权重缩放到 ~ -50

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
    """exp(-k * 径向投影偏移）。不再按可见性门控。

    径向偏移由 3D 中心投影到图像的 ``(u_ndc, v_ndc)`` 计算，与原 2D 居中等价
    （u/v_ndc 即归一化方位角），驱动底盘偏航对正水平、升降杆对正纵向。
    """
    s = get_bbox3d_state(env)
    radial = torch.sqrt(s.u_ndc ** 2 + s.v_ndc ** 2)
    return torch.exp(-k * radial)


def approach(env: "ManagerBasedRLEnv", d_target: float = 0.4, near_penalty: float = 4.0) -> torch.Tensor:
    """靠近塑形：在 ``d_target`` 处取得正向峰值（+1）；过近则转为惩罚。

    ``distance`` 是目标 3D 中心到相机原点的米制距离（run_realtime 的
    ``Box3D.distance``）。原先远端为纯负值（最高仅趋于 0），缺乏"靠近"的正向
    动机，导致策略宁可保持距离也不冒丢失视野的失败风险。改为单峰正向塑形：

    * ``distance >= d_target``：奖励 = ``exp(-(distance - d_target))`` ∈ (0, 1]，
      随接近 ``d_target`` 单调升高至 +1，给出明确的"靠近"正梯度。
    * ``distance <  d_target``：奖励 = ``1 - near_penalty * (d_target - distance)``，
      越靠近负惩罚越深，防止底盘撞上目标——替代已移除的硬碰撞终止。

    在 ``d_target`` 处连续（两段均为 +1）。不再按可见性门控。
    """
    s = get_bbox3d_state(env)
    over = s.distance - d_target
    far = torch.exp(-over.clamp(min=0.0))   # 远端：越近越高（(0,1]，趋于 +1）
    near = (-over).clamp(min=0.0)           # 近端：d_target - distance（>=0）
    return far - near_penalty * near


def smoothness_penalty(
    env: "ManagerBasedRLEnv",
    base_weight: float = 1.0,
    lift_weight: float = 1.0,
) -> torch.Tensor:
    """惩罚底盘(前向/偏航)与升降杆速度指令的逐步变化（≈加速度），抑制急加减速/抖动。

    三维动作均为*速度*指令 ``(v, w, lift_vel)``，相邻控制步的差分即对应的加速度。
    底盘两维 ``(v, w)`` 与升降杆维各自可调权重，便于分别约束底盘和升降杆的加速度。
    返回非负的加权平方和（在配置中以负权重使用）。
    """
    diff = env.action_manager.action - env.action_manager.prev_action
    base = torch.sum(torch.square(diff[:, 0:2]), dim=-1)   # v, w 加速度
    lift = torch.square(diff[:, 2])                        # 升降杆加速度
    return base_weight * base + lift_weight * lift


def success_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """+1 在满足成功条件的步骤上（由 TaskOutcome 项设置）。"""
    flag = getattr(env, "_chassis_success", None)
    if flag is None:
        return torch.zeros(env.num_envs, device=env.device)
    return flag.float()


def failure_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """+1 在丢失目标（视野丢失）失败的步骤上（使用负权重）。"""
    flag = getattr(env, "_chassis_failure", None)
    if flag is None:
        return torch.zeros(env.num_envs, device=env.device)
    return flag.float()
