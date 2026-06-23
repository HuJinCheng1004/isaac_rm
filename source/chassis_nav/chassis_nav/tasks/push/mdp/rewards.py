# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_ee_distance_body(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["r_link8"]),
) -> torch.Tensor:
    """tanh 形 reaching 奖励：末端执行器 r_link8 → 方块 CoM 距离。"""
    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    cube_pos_w = object.data.root_pos_w
    ee_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    distance = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(distance / std)


def ee_z_to_block(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["r_link8"]),
) -> torch.Tensor:
    """exp 形 EE 高度对齐奖励：拉末端下降到方块 z 高度。

    iter 3: 解决"悬停失败"——3D reaching 梯度方向无关，手臂从正上方接近方块
    （同 x,y，z 略高），自上而下接触→竖直力→方块不产生水平位移。本项强制末端
    降到方块高度（z≈0.055 m），使侧向接触→水平力→方块可被推动。

    用 exp 核而非 tanh：home 位 EE 在方块上方 ~0.82 m。tanh std=0.1 时
    1-tanh(8.24)≈0（零梯度）；exp std=0.5 时 exp(-1.648)=0.193，从 home 到接触
    全程都有有意义梯度。
    """
    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    ee_z = robot.data.body_pos_w[:, robot_cfg.body_ids[0], 2]
    block_z = object.data.root_pos_w[:, 2]
    return torch.exp(-(ee_z - block_z).abs() / std)


def block_to_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """tanh-shaped tracking reward: block -> goal distance (planar push, no height gate).

    GOAL FRAME (iter 1 fix): the command pos is resolved in the per-env origin / table
    frame (identity rotation, z=0 origin), NOT the robot articulation root. The fixed
    chassis root sits at world z=-0.805 (lift-pole base, ~0.8 m below the table); resolving
    the cmd z=0.04 against the root put the goal ~0.8 m underground and ~1.0 m from the
    block, making success geometrically impossible. env.scene.env_origins is the table /
    block reference frame, so goal_w = env_origin + cmd lands at table height (z~0.04) in
    the block's reachable region.
    """
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w = env.scene.env_origins + des_pos_b
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def block_to_goal_distance_contact_gated(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    std_gate: float = 0.05,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["r_link8"]),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Contact-gated block->goal tracking (iter 3): precision tracking that the policy
    can only collect WHILE the EE is in contact with the block.

    iter 1-2 evidence: the policy learned an OPEN-LOOP "swat" -- one ballistic shove,
    then the EE retracts (reaching collapsed to ~0, EE ends ~31 cm from block) while the
    block coasts to a stop ~10 cm short of the goal. The ungated precision tracking
    (mid/fine bands) was farmable by a single shove: once the block drifts goal-ward the
    policy keeps banking tracking reward with the arm already withdrawn, so there is no
    incentive to STAY in contact for the closed-loop final centimetres.

    Mechanism: multiply the tracking kernel `1 - tanh(d_block_goal / std)` by a soft
    contact indicator `g = 1 - tanh(d_ee_block / std_gate)`. With std_gate=0.05:
      g(d_ee_block=0)    = 1.00
      g(d_ee_block=0.05) = 0.24
      g(d_ee_block=0.10) = 0.04
      g(d_ee_block=0.31) ~ 0   (the iter-2 abandoned-arm distance)
    So precision reward is paid ONLY while the EE is touching the block; abandoning the
    block after a shove zeroes this term, making closed-loop pushing strictly beat the
    swat exploit. The COARSE attractor (std=0.3, w=16) stays UNGATED so long-range shove
    initiation keeps its gradient (per the iter-3 brief and reward-experience #9).

    Goal frame matches block_to_goal_distance (env-origin / table frame, iter-1 fix).
    """
    object: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w = env.scene.env_origins + des_pos_b
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    tracking = 1 - torch.tanh(distance / std)
    # soft contact gate: EE (r_link8) -> block CoM distance, world frame
    ee_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    d_ee_block = torch.norm(object.data.root_pos_w - ee_w, dim=1)
    gate = 1 - torch.tanh(d_ee_block / std_gate)
    return tracking * gate


def block_at_goal(
    env: ManagerBasedRLEnv,
    threshold: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Binary success indicator; sets env._reach_success for harbor eval success_rate.

    GOAL FRAME (iter 1 fix): goal resolved in env-origin / table frame (see
    block_to_goal_distance). SUCCESS METRIC (iter 1): XY distance only, per the task spec
    "block-to-goal XY distance < 5 cm" -- the z component is dropped so a small table-height
    offset cannot block success.
    """
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w = env.scene.env_origins + des_pos_b
    distance = torch.norm(des_pos_w[:, :2] - object.data.root_pos_w[:, :2], dim=1)
    result = (distance < threshold).float()
    env._reach_success = result
    return result
