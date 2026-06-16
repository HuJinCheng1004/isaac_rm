# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""底盘靠近/对齐任务的课程（curriculum）项。

:func:`approach_difficulty` 随训练推进，把 :class:`mdp.events.ResetRobotTargetInView`
的初始难度从"近、基本在正前方"线性扩展到该重置事件配置的完整范围：

* **偏航半幅** ``yaw_start -> yaw_full``（目标方位角范围逐步打开）。
* **目标深度上限** ``d_max_start -> d_full``（初始距离逐步拉远）。

早期更易把目标保持在窄 FOV 内并完成靠近，从而制造成功经验、打破"只居中不靠近"
的保守局部最优；后期恢复满难度以保证泛化。

课程管理器每步对*被重置*的环境调用本函数；这里直接改写重置事件实例缓存的激活
范围（``term._yaw_range`` / ``term._d_max``），下次重置即生效。函数返回当前进度
``p ∈ [0, 1]`` 供 TensorBoard 记录。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def approach_difficulty(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    num_steps: float,
    yaw_start: float = 0.4,
    height_start: tuple = (0.6, 1.0),
) -> float:
    """线性退火重置难度，返回进度 ``p`` 供日志。

    早期：目标基本在正前方（窄偏航）、高度集中在相机舒适带（易居中+靠近）-> 制造
    成功经验；``num_steps`` 步后退火到满难度（全向偏航 + 完整高度区间）。

    Args:
        num_steps: 退火到满难度所需的训练步数（与 ``trainer.timesteps`` 同单位，即
            ``env.common_step_counter``）。
        yaw_start: 初始偏航半幅 [rad]，退火到 ``term._yaw_full``。
        height_start: 初始目标高度区间 [m]，退火到 ``term._height_full``。
    """
    term = getattr(env, "_reset_in_view_term", None)
    if term is None:
        return 0.0

    p = min(1.0, max(0.0, env.common_step_counter / max(1.0, float(num_steps))))

    yaw_half = yaw_start + p * (term._yaw_full - yaw_start)
    term._yaw_range = (-yaw_half, yaw_half)

    h_lo = height_start[0] + p * (term._height_full[0] - height_start[0])
    h_hi = height_start[1] + p * (term._height_full[1] - height_start[1])
    term._height_range = (h_lo, h_hi)

    return p
