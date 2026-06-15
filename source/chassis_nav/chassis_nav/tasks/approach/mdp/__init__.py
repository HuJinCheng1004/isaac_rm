# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""底盘靠近任务的 MDP 项。

重新导出完整的内置 Isaac Lab MDP 库（``last_action``，
``action_rate_l2``，``time_out``，``randomize_*``，``reset_root_state_uniform``，
...）加上此任务的自定义动作/观察/奖励/终止项。
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .actions import DifferentialDriveAction, DifferentialDriveActionCfg  # noqa: F401
from .events import ResetRobotTargetInView  # noqa: F401
from .observations import BBox3DObservation, chassis_velocity, get_bbox3d_state, lift_height  # noqa: F401
from .rewards import approach, centering, failure_penalty, smoothness_penalty, success_reward  # noqa: F401
from .terminations import TaskOutcome  # noqa: F401
