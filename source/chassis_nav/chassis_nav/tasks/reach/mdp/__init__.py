# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""底盘"到达目标点"任务的 MDP 项。

重新导出完整的内置 Isaac Lab MDP 库（``last_action``、``action_rate_l2``、
``time_out``、``randomize_*`` ...）加上本任务的自定义动作/观测/奖励/终止/事件项。
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from .actions import DifferentialDriveAction, DifferentialDriveActionCfg  # noqa: F401
from .events import ResetChassisReach  # noqa: F401
from .observations import chassis_velocity, get_reach_state, target_in_robot_frame  # noqa: F401
from .rewards import approach, heading, stop, success_reward  # noqa: F401
from .terminations import ReachOutcome  # noqa: F401
