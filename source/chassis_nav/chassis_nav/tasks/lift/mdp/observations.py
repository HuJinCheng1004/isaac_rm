# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from chassis_nav.robots.arm import EE_BODY_NAME

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Cube position expressed in the robot root frame (3-D)."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w
    )
    return object_pos_b


def ee_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
) -> torch.Tensor:
    """End-effector (r_link8) position expressed in the robot root frame (3-D).

    Uses body_pos_w + subtract_frame_transforms, mirroring object_position_in_robot_root_frame,
    so the agent senses where its hand is relative to its fixed base (and hence the lift height).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    ee_pos_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0], :3]
    ee_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w
    )
    return ee_pos_b
