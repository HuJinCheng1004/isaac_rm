# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""底盘"到达目标点"任务的终止项。

:class:`ReachOutcome` 是有状态终止项：当机器人位于目标前方停靠环上
(``|dist_xy - d_standoff| < dist_tol``)、目标基本正前方 (``cos_bearing > cos(bearing_tol)``)、
且低速 (``v^2 + w^2 < speed_tol^2``)，并持续 ``success_dwell`` 步时判成功。

终止管理器在奖励管理器之前运行，因此这里缓存的 ``env._reach_success`` 由
:func:`rewards.success_reward` 在同一步读取。``time_out`` 由内置项单独处理，
以便 PPO 在时间限制时正确 bootstrap 值函数。
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

from .observations import get_reach_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ReachOutcome(ManagerTermBase):
    """评估"到达并停住"成功条件并缓存奖励标志。"""

    def __init__(self, cfg: TerminationTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        p = cfg.params
        self.d_standoff: float = float(p.get("d_standoff", 0.8))
        self.dist_tol: float = float(p.get("dist_tol", 0.2))
        self.bearing_tol: float = float(p.get("bearing_tol", 0.25))
        self.speed_tol: float = float(p.get("speed_tol", 0.1))
        self.success_dwell: int = int(p.get("success_dwell", 5))

        self._cos_tol = math.cos(self.bearing_tol)
        self._success_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        d_standoff: float = 0.8,
        dist_tol: float = 0.2,
        bearing_tol: float = 0.25,
        speed_tol: float = 0.1,
        success_dwell: int = 5,
    ) -> torch.Tensor:
        # 参数在 __init__ 中消费（存到 self）；此处列出仅为 2.3.2 的签名校验。
        s = get_reach_state(env)

        dist_ok = (s.dist_xy - self.d_standoff).abs() < self.dist_tol
        bearing_ok = s.cos_bearing > self._cos_tol
        speed_ok = (s.v_fwd * s.v_fwd + s.w_yaw * s.w_yaw) < (self.speed_tol * self.speed_tol)
        good = dist_ok & bearing_ok & speed_ok

        self._success_count = torch.where(
            good, self._success_count + 1, torch.zeros_like(self._success_count)
        )
        success = self._success_count >= self.success_dwell

        env._reach_success = success
        return success

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._success_count[:] = 0
        else:
            idx = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self._success_count[idx] = 0
