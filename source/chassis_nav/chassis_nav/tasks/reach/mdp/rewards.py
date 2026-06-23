# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""底盘"到达目标点"任务的奖励项。

任务语义：机器人接近目标 → 对准（目标落在机体正前方）→ 在目标前方 ``d_standoff``
处低速停住。奖励按"前进 / 对准 / 精停靠 / 停住 / 成功"拆分，便于后续用 harbor 的
reward-tune 自动迭代。所有项读取 :func:`observations.get_reach_state` 的共享几何。

----------------------------------------------------------------------------
每阶段量级预算（仓库约定：dense 项有效 per-step = func * weight * dt，dt=0.1；
success 权重已手工除以 dt 故其有效 landmark = 名义 100；权重见 reach_env_cfg.RewardsCfg）：

  阶段      项                 saturate 行为                          名义 per-step (func*weight)
  ------    ----------------   ------------------------------------   --------------------------
  导航/精停 approach (3.0)     exp(-dist/1.5) - baseline(0.160)：      spawn≈0；1.5m≈+0.62；
                                 spawn≈0、靠近>0、远离<0；平滑单调       0.8m 环≈+1.28；0m≈+2.52
                                 （越近梯度越陡，自然加密停靠环附近）     （episode 累积主导项）
  对准前进  heading  (1.5)     v_fwd * clamp(cos_bearing,0,1)：         +0.75 @ 0.5m/s 正对前进
                                 面向且前进→+、静止 v_fwd=0→0             （驱动"对着开"原语）
  停住      stop     (0.6)     进环后 -v_fwd^2，停下 → 0               小负 → 0
  平滑      action_rate(-0.01) 动作突变惩罚（正则，|w|≤0.1）            小负
  终端      success  (100/dt→100 名义)  稀疏一次性 landmark             +100（高于 dense sum 一个量级）

设计演进（消灭"零动作静止"局部最优 + 平滑可学梯度）：
  * iter 0 approach = -|dist-d0| 恒负 → 全程静止局部最优 (success=0)。
  * iter 1 approach = 1-tanh(dist-d0)：恒正但梯度太平，机器人几乎不动 (total +1.6)。
  * iter 2 approach = 高斯 exp(-0.5((dist-d0)/1)^2)：在 spawn(1.5m) 处≈0.78，是个**大正值**，
    机器人不动就能拿到 approach=+8.87 → 与 iter 0 同类的静止局部最优 (success=0)。
  * iter 3/4 progress = dist_prev - dist_curr（势能差）：静止=0 杜绝静态吸引子，
    但 1 步差分受物理抖动 / planar_lock 噪声主导，PPO 在 20M 步无法稳定提取方向梯度
    （progress 收敛为负，机器人退化静止）；配套 near_ring 高斯造成 1.0–1.5m 刷分带。
  * iter 5 修复（本次）：改用**锚定 spawn 基线的平滑指数势能** approach：
      r = exp(-dist/lambda) - exp(-spawn_avg/lambda)。
      - spawn 平均处 ≈ 0（无静态吸引子、无刷分带），仅真正靠近才转正；
      - 处处可微、随距离单调，越靠近目标梯度越陡 → 取代 near_ring 的精对环作用，无门控边界；
      - 非逐步差分，单步内无物理抖动注入（修掉 progress 的高噪声梯度问题）；
      - 同时取代 progress + near_ring 两项，奖励结构更简单。
  * heading 改为 v_fwd*clamp(cos_bearing,0,1)：直接奖励"面向目标且前进"原语；
    静止 (v_fwd=0) 时为 0，不面向 (cos<0 被 clamp 到 0) 时为 0，不会奖励空转。
    iter 5：weight 1.0→1.5，配合平滑 approach 势能强化"对着目标开"。
  * stop 只惩罚前向线速度 v_fwd（不惩罚 w_yaw）：进环后对准需原地旋转。
  * success 名义 +100，高于 dense episode sum 一个量级（experience #2）。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from .observations import get_reach_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def approach(
    env: "ManagerBasedRLEnv",
    lambda_dist: float = 1.5,
    spawn_avg: float = 2.75,
) -> torch.Tensor:
    """密集前进塑形：锚定到 spawn 基线的平滑指数势能。

    r = exp(-dist_xy / lambda_dist) - exp(-spawn_avg / lambda_dist)
        * spawn 平均处 (dist≈spawn_avg) → r ≈ 0   （杜绝"零动作静止收奖"局部最优）
        * 朝目标靠近 (dist 减小)        → r > 0    （平滑、随靠近单调增大）
        * 远离目标 (dist 增大)          → r < 0    （劝阻后退）

    iter 5 策略改变（替换 iter 3/4 的 progress + near_ring）：
      * iter 0/1/2 的各种 approach(dist) 在 spawn 处恒为大正值 → 静止局部最优；
      * iter 3/4 的 progress = dist_prev - dist_curr 是 1 步势能差，物理抖动 / planar_lock
        噪声太大，PPO 在 20M 步也无法稳定提取方向梯度（progress 收敛为负，机器人退化静止）；
      * near_ring 是有门控的局部高斯，造成 1.0–1.5m 刷分带 / 边界绕圈。
    本次改用**平滑指数势能** exp(-dist/λ)，并减去 spawn 平均距离处的基线值，使其：
      - 在 spawn 处 ≈ 0（无静态吸引子、无刷分带），仅当真正靠近目标才转正；
      - 处处可微、随距离单调，越靠近目标梯度越陡（自然在停靠环附近加密），
        取代 near_ring 的局部精对环作用，且不存在门控边界；
      - 不是逐步差分，单步内无物理抖动注入（避免 progress 的高噪声梯度问题）。

    数值兜底：物理爆炸时不让 inf 污染整批 PPO 更新（baseline 用 Python ``math.exp``，
    逐 env 项用 ``torch.exp``）。
    """
    s = get_reach_state(env)
    baseline = math.exp(-spawn_avg / max(lambda_dist, 1e-3))
    val = torch.exp(-s.dist_xy / max(lambda_dist, 1e-3)) - baseline
    return torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0).clamp(-2.0, 2.0)


def heading(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """密集"对着开"塑形：前向线速度 × 朝向目标的方位余弦（仅取正向分量）。

    r = v_fwd * clamp(cos_bearing, 0, 1)
        * 面向目标且前进 (cos>0, v_fwd>0)  → r > 0    （奖励"对着目标开"原语）
        * 静止 (v_fwd=0)                   → r = 0    （不动不给奖励）
        * 背对目标 (cos<0)                 → clamp→0  （不惩罚，但也不奖励空转/倒车）

    iter 3：iter 0/1/2 的全局 cos_bearing 在静止时仍给正奖励（面向即可收奖），
    与"静止局部最优"共谋。改为耦合速度后，机器人必须**真的朝目标移动**才拿到奖励，
    直接塑造导航原语。iter 5：weight 1.0→1.5，强化"对着目标开"原语以配合平滑 approach 势能。
    """
    s = get_reach_state(env)
    return s.v_fwd * torch.clamp(s.cos_bearing, 0.0, 1.0)


def stop(env: "ManagerBasedRLEnv", d_standoff: float = 0.8, dist_tol: float = 0.2) -> torch.Tensor:
    """停住塑形：仅当已接近停靠环时，惩罚**前向线速度**（鼓励"到了就停"）。

    r = -(v_fwd^2) * in_ring，``in_ring`` = |dist_xy - d_standoff| < dist_tol。
    环外不惩罚速度（导航阶段需要移动）。

    移除 w_yaw 惩罚：进环后对准目标需原地旋转 (w_yaw≠0)，只要求线速度归零，
    允许底盘在环上原地精对准；偏航的最终静止由成功判据 (speed_tol) 把关。
    """
    s = get_reach_state(env)
    in_ring = ((s.dist_xy - d_standoff).abs() < dist_tol).float()
    return -(s.v_fwd * s.v_fwd) * in_ring


def success_reward(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """稀疏终端成功奖励（读取 :class:`terminations.ReachOutcome` 缓存的成功标志）。"""
    flag = getattr(env, "_reach_success", None)
    if flag is None:
        return torch.zeros(env.num_envs, device=env.device)
    return flag.float()
