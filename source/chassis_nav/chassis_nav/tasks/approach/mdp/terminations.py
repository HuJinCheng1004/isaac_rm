# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""底盘靠近/对齐任务的终止项。

:class:`TaskOutcome` 是单个有状态（基于类）的终止项，一起
评估两个情节结束条件，因为它们共享
停驻计数器并馈送终端奖励项：

* **success**   : 目标居中且 3D 距离 ~ d_target，保持 ``success_dwell`` 步。
* **lost**      : 目标超出视野（几何）或框中心投影超出图像，
                  保持 ``lost_dwell`` 步（默认 1 步，即一旦丢失视野立即失败）。

碰撞终止已移除——改由 :func:`rewards.approach` 在距离近于 ``d_target`` 时
施加软惩罚来防止底盘撞上目标。

终止管理器在每一步的奖励管理器*之前*运行，因此
缓存在此处的 ``env._chassis_success`` / ``env._chassis_failure`` 标志由
:func:`rewards.success_reward` / :func:`rewards.failure_penalty` 在同一步读取。
``time_out`` 由内置项单独处理，以便 PPO 在时间限制时正确启动
值函数。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, TerminationTermCfg

from .observations import get_bbox3d_state

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class TaskOutcome(ManagerTermBase):
    """一起评估成功/碰撞/丢失并缓存奖励标志。"""

    def __init__(self, cfg: TerminationTermCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        p = cfg.params
        self.robot_name: str = p.get("robot_name", "robot")
        self.target_name: str = p.get("target_name", "target")
        self.d_target: float = float(p.get("d_target", 0.4))
        self.center_tol: float = float(p.get("center_tol", 0.15))
        self.dist_tol: float = float(p.get("dist_tol", 0.2))
        self.success_dwell: int = int(p.get("success_dwell", 10))
        self.lost_dwell: int = int(p.get("lost_dwell", 1))

        self._success_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._lost_count = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

    def __call__(
        self,
        env: "ManagerBasedRLEnv",
        robot_name: str = "robot",
        target_name: str = "target",
        d_target: float = 0.4,
        center_tol: float = 0.15,
        dist_tol: float = 0.2,
        success_dwell: int = 10,
        lost_dwell: int = 1,
    ) -> torch.Tensor:
        # 参数在 __init__ 中被使用；在此处列出以进行 2.3.2 签名验证。
        s = get_bbox3d_state(env)
        u_ndc, v_ndc, distance, visible = s.u_ndc, s.v_ndc, s.distance, s.visible

        # --- 成功：居中 + 到达目标距离，持续 ---
        centred = torch.sqrt(u_ndc**2 + v_ndc**2) < self.center_tol
        dist_ok = (distance - self.d_target).abs() < self.dist_tol
        good = visible & centred & dist_ok
        self._success_count = torch.where(good, self._success_count + 1, torch.zeros_like(self._success_count))
        success = self._success_count >= self.success_dwell

        # --- 丢失：不可见，或中心投影超出图像，持续 lost_dwell 步（默认 1 = 立即失败）---
        out_of_frame = (~visible) | (u_ndc.abs() > 1.0) | (v_ndc.abs() > 1.0)
        self._lost_count = torch.where(out_of_frame, self._lost_count + 1, torch.zeros_like(self._lost_count))
        lost = self._lost_count >= self.lost_dwell

        # 为奖励项缓存标志（丢失视野判为失败；失败优先于成功）
        failure = lost
        success = success & (~failure)
        env._chassis_success = success
        env._chassis_failure = failure

        return success | failure

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._success_count[:] = 0
            self._lost_count[:] = 0
        else:
            idx = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self._success_count[idx] = 0
            self._lost_count[idx] = 0
