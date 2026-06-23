# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""底盘"到达目标点"任务的观测（全状态、无相机）。

与 ``approach`` 任务不同：这里目标 3D 框的位置是**全局已知**的（直接读目标刚体
世界位姿），不经过相机/检测/噪声/延迟。策略观测目标在**机器人基座坐标系**下的
平面相对位置 + 自身速度 + 历史动作。

共享状态 :class:`ReachState` 由观测/奖励/终止三方共用，并按 ``common_step_counter``
缓存一次，保证同一物理步内三者读到完全一致的几何量（无论调用顺序）。
"""

from __future__ import annotations

from typing import NamedTuple, TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ReachState(NamedTuple):
    """机器人基座系下的目标几何 + 底盘本体感觉（每物理步缓存一次）。"""

    dx_b: torch.Tensor      # (N,) 目标在基座系的前向偏移 [m]
    dy_b: torch.Tensor      # (N,) 目标在基座系的左向偏移 [m]
    dist_xy: torch.Tensor   # (N,) 基座<->目标的平面距离 [m]
    cos_bearing: torch.Tensor  # (N,) dx_b / dist_xy ∈ [-1,1]，=1 表示目标正前方
    v_fwd: torch.Tensor     # (N,) 底盘前向线速度 [m/s]
    w_yaw: torch.Tensor     # (N,) 底盘偏航角速度 [rad/s]


_CACHE_ATTR = "_chassis_reach_state"


def get_reach_state(
    env: "ManagerBasedEnv", robot_name: str = "robot", target_name: str = "target"
) -> ReachState:
    """计算（并按步缓存）机器人基座系下的目标几何 + 底盘速度。"""
    step = int(env.common_step_counter)
    cached = getattr(env, _CACHE_ATTR, None)
    if cached is not None and cached[0] == step:
        return cached[1]

    robot = env.scene[robot_name]
    target = env.scene[target_name]

    robot_pos = robot.data.root_pos_w        # (N,3) 世界
    robot_quat = robot.data.root_quat_w      # (N,4) wxyz
    tgt_pos = target.data.root_pos_w         # (N,3) 世界

    rel_w = tgt_pos - robot_pos                                   # (N,3) 世界
    rel_b = math_utils.quat_apply_inverse(robot_quat, rel_w)      # (N,3) 基座系
    dx_b = rel_b[:, 0]
    dy_b = rel_b[:, 1]
    dist_xy = torch.sqrt(dx_b * dx_b + dy_b * dy_b).clamp(min=1e-4)
    cos_bearing = (dx_b / dist_xy).clamp(-1.0, 1.0)

    v_fwd = robot.data.root_lin_vel_b[:, 0]
    w_yaw = robot.data.root_ang_vel_b[:, 2]

    # NaN/Inf 兜底：物理求解器偶发爆炸时，不让 inf 经 RunningStandardScaler 污染整批
    # PPO 更新（approach 任务踩过的坑）。
    def _clean(x, lo=-1e4, hi=1e4):
        return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp(lo, hi)

    state = ReachState(
        dx_b=_clean(dx_b),
        dy_b=_clean(dy_b),
        dist_xy=_clean(dist_xy, lo=1e-4, hi=1e4),
        cos_bearing=_clean(cos_bearing, lo=-1.0, hi=1.0),
        v_fwd=_clean(v_fwd, lo=-10.0, hi=10.0),
        w_yaw=_clean(w_yaw, lo=-10.0, hi=10.0),
    )
    setattr(env, _CACHE_ATTR, (step, state))
    return state


def target_in_robot_frame(
    env: "ManagerBasedEnv",
    robot_name: str = "robot",
    target_name: str = "target",
) -> torch.Tensor:
    """策略观测：目标在基座系的平面位置，形状 (N, 3) = [dx_b, dy_b, dist_xy]。"""
    s = get_reach_state(env, robot_name, target_name)
    return torch.stack([s.dx_b, s.dy_b, s.dist_xy], dim=-1)


def chassis_velocity(
    env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """本体感觉：基座系前向线速度 + 偏航角速度，形状 (N, 2)。"""
    asset = env.scene[asset_cfg.name]
    v_fwd = asset.data.root_lin_vel_b[:, 0:1]
    w_yaw = asset.data.root_ang_vel_b[:, 2:3]
    out = torch.cat([v_fwd, w_yaw], dim=-1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).clamp(-10.0, 10.0)
