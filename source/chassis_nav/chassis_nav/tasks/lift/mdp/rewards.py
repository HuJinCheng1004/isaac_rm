# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Reward MDP terms for the RM right-arm "lift block" task (§6).

Adapted from the proven Isaac-Lift-Cube-Franka-v0 base (reach -> lift indicator ->
goal-tracking ladder) and from the sibling push task's repo idioms (no FrameTransformer:
the EE position is read directly from ``robot.data.body_pos_w[:, body_id]``; success term
uses a negligible positive weight so the RewardManager runs it and the
``env._reach_success`` side-effect fires).

KEY DIFFERENCE vs the Franka base: this task has NO CommandsCfg / object_pose command.
The lift target is a HARD-CODED world height (z = 0.15 m above the table surface), so
``cube_height_reward`` / ``block_lifted_and_grasped`` compute height directly from the
cube's world pose and never call ``env.command_manager`` (that would crash).

Composer = sum (IsaacLab RewardManager sums weight_i * dt * func_i per step). The
RewardManager's dt scaling is LEFT IN PLACE to match the proven push/Franka repo
convention (the success term's weight=1e-6 is calibrated for that convention: weight==0
is skipped by the manager, a tiny positive weight survives so the side-effect runs).

Per-step saturated magnitude budget (nominal weights, before the ~dt=1/30 scaling),
strictly increasing reach -> lift -> height:
  reaching_block : [0, 1]   * 1.0  -> ~1/step
  lifting_object : {0, 1}   * 15.0 -> 15 when cube z > 0.04 m
  cube_height    : [0, 1]   * 16.0 -> up to 16 (gated on lifted), pulls cube to 0.20 m
  grasping       : [0, 1]   * 5.0  -> up to 5 (gated on lifted), EE-cube contact
  success        : {0, 1}   * 1e-6 -> logging-only side-effect (sets env._reach_success)
  action_rate    : <= 0,    * -1e-4 regularizer
  joint_vel      : <= 0,    * -1e-4 regularizer
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from chassis_nav.robots.arm import EE_BODY_NAME, GRIPPER_JOINT_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def zero_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Constant-zero placeholder reward (kept for back-compat; not wired in §6)."""
    return torch.zeros(env.num_envs, device=env.device)


def joint_jerk_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Mean absolute joint acceleration over the controlled joints (jerk proxy).

    Penalizes sudden velocity changes between physics steps. Uses
    ``robot.data.joint_acc[:, joint_ids]`` which PhysX computes per body.
    Mean-abs (not sum-of-squares) avoids the 6-joint amplification that L2-squared
    introduces with high-variance random-policy actions; comparable in magnitude to
    ``joint_vel_l2`` when weighted the same.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    return torch.mean(torch.abs(robot.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def table_avoidance_penalty(
    env: ManagerBasedRLEnv,
    table_height: float = 0.02,
    std: float = 0.02,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Soft penalty when structural arm links approach the table surface.

    Uses a tanh gate over world-z of the specified bodies: gate ≈ 1 when body z is
    near or below ``table_height``, near 0 when well above. Returns the sum of per-body
    gate values (positive), multiplied by a negative weight in the cfg.

    ``table_height = 0.02`` (20 mm above table surface at z ≈ 0) keeps the penalty
    near-zero at cube-grasp height (z ≈ 0.055) while firing strongly when links
    scrape the table (z < 0.02). Structural links only (r_link1–r_link6) are included;
    the end-effector (r_link8) and fingers are excluded because they must approach the cube.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    body_z = robot.data.body_pos_w[:, robot_cfg.body_ids, 2]  # (N, num_bodies)
    gate = 1.0 - torch.tanh((body_z - table_height) / std)
    return gate.sum(dim=-1)


def pre_grasp_reward(
    env: ManagerBasedRLEnv,
    proximity_threshold: float = 0.08,
    gripper_closed_threshold: float = 0.005,
    minimal_height: float = 0.04,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES),
) -> torch.Tensor:
    """Binary indicator: cube lifted AND EE near AND gripper closed.

    Gated on cube_z > minimal_height to prevent the static-hover shortcut discovered
    in iter_003 (policy kept gripper closed near the cube ON the table for 93% of
    each episode, earning 62.5/68.6 total reward without ever lifting).
    Only fires when all three conditions hold simultaneously: the cube is off the table,
    the EE is within proximity_threshold, and the gripper fingers are closed.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    cube_pos_w = obj.data.root_pos_w
    ee_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    d = torch.norm(cube_pos_w - ee_w, dim=1)
    near = (d < proximity_threshold).float()
    finger_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids].mean(dim=1)
    gripper_closed = (finger_pos < gripper_closed_threshold).float()
    lifted = (cube_pos_w[:, 2] > minimal_height).float()
    return lifted * near * gripper_closed


def cube_height_with_grip(
    env: ManagerBasedRLEnv,
    initial_z: float = 0.055,
    std_h: float = 0.05,
    proximity_threshold: float = 0.08,
    gripper_closed_threshold: float = 0.005,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES),
) -> torch.Tensor:
    """Dense lifting reward: tanh(height_gain/std_h) × (EE near) × (gripper closed).

    Avoids all prior shortcuts:
    - hover (cube on table, gripper closed): height_gain=0 → zero reward
    - open-gripper-push (iter_005): gripper_closed=False → zero reward
    tanh normalization maps height_gain to [0,1] so the signal is dense even for
    small lifts: std_h=0.05m → 0.5cm lift gives tanh(0.1)≈0.10, 5cm gives tanh(1)≈0.76.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    cube_pos_w = obj.data.root_pos_w
    cube_z = cube_pos_w[:, 2]
    ee_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    d = torch.norm(cube_pos_w - ee_w, dim=1)
    near = (d < proximity_threshold).float()
    finger_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids].mean(dim=1)
    gripper_closed = (finger_pos < gripper_closed_threshold).float()
    height_gain = torch.clamp(cube_z - initial_z, min=0.0)
    return near * gripper_closed * torch.tanh(height_gain / std_h)


def cube_lateral_velocity_penalty(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalty for cube moving horizontally — blocks the side-push shortcut.

    Returns ‖v_xy‖ of the cube (positive, multiplied by a negative weight in cfg).
    Pushing the cube sideways earns negative reward; the only zero-penalty strategy
    is to either not contact the cube or contact it vertically (grasp-and-lift).
    """
    obj: RigidObject = env.scene[object_cfg.name]
    lateral_vel = obj.data.root_lin_vel_w[:, :2]
    return torch.norm(lateral_vel, dim=1)


def gripper_near_cube_shaping(
    env: ManagerBasedRLEnv,
    proximity_threshold: float = 0.08,
    std: float = 0.02,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=GRIPPER_JOINT_NAMES),
) -> torch.Tensor:
    """Soft reward for gripper closure when EE is near the cube (no lifted gate).

    Provides gradient signal to push the policy from open-gripper-push toward
    closed-gripper-grasp. std=0.02m is chosen so the gradient is non-zero at the
    current open state (finger_pos=0.0325m): tanh(0.0325/0.02)=0.85, gradient≈-7.
    Weight kept small (0.5) to cap max hover episodic at ~3.3/episode, well below
    reaching_block (~6.8), so the hover shortcut is not a competitive strategy.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    cube_pos_w = obj.data.root_pos_w
    ee_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    d = torch.norm(cube_pos_w - ee_w, dim=1)
    near = (d < proximity_threshold).float()
    finger_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids].mean(dim=1)
    closure = 1.0 - torch.tanh(finger_pos / std)
    return near * closure


def object_ee_distance_body(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
) -> torch.Tensor:
    """tanh reaching reward: end-effector (r_link8) -> cube CoM proximity.

    Reads the EE position directly from ``body_pos_w`` (this robot has NO FrameTransformer),
    mirroring the sibling push task. Returns ``1 - tanh(d / std)`` in [0, 1].
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    cube_pos_w = obj.data.root_pos_w
    ee_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    distance = torch.norm(cube_pos_w - ee_w, dim=1)
    return 1 - torch.tanh(distance / std)


def object_is_lifted(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Binary +1 indicator while the cube's world z exceeds ``minimal_height``.

    From the Franka lift base; encourages any upward movement off the table surface.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.where(obj.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_is_lifted_grasped(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    proximity_threshold: float = 0.12,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
) -> torch.Tensor:
    """Binary +1 when cube is lifted above ``minimal_height`` AND EE is within ``proximity_threshold``.

    Prevents the arm-body-push shortcut: the lifting indicator only fires when the EE
    is actually near the cube, forcing the policy to approach and grasp before lifting.
    Threshold 0.12 m gives slack for the wrist offset — tighter than the 0.08 m success
    gate so the policy has room to develop before success starts firing.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    cube_pos_w = obj.data.root_pos_w
    ee_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    d_ee_cube = torch.norm(cube_pos_w - ee_w, dim=1)
    lifted = cube_pos_w[:, 2] > minimal_height
    near = d_ee_cube < proximity_threshold
    return (lifted & near).float()


def cube_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    minimal_height: float,
    proximity_threshold: float = 0.12,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
) -> torch.Tensor:
    """tanh height-tracking reward pulling the cube toward ``target_height`` (world z).

    Gated on (a) cube lifted above ``minimal_height`` AND (b) EE within ``proximity_threshold``
    of the cube. The proximity gate prevents the arm-body-push shortcut from collecting height
    reward — only a proper grasp-and-lift earns it.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    cube_pos_w = obj.data.root_pos_w
    cube_z = cube_pos_w[:, 2]
    ee_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    d_ee_cube = torch.norm(cube_pos_w - ee_w, dim=1)
    lifted = (cube_z > minimal_height).float()
    near = (d_ee_cube < proximity_threshold).float()
    height_error = (target_height - cube_z).abs()
    return lifted * near * (1 - torch.tanh(height_error / std))


def grasping_reward(
    env: ManagerBasedRLEnv,
    std_gate: float,
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
) -> torch.Tensor:
    """Soft finger-contact reward, gated on the cube being lifted.

    Finger contact is approximated by EE (r_link8) -> cube CoM distance (same soft-gate
    approach as the push task's contact gate): ``g = 1 - tanh(d_ee_cube / std_gate)`` is ~1
    when the hand is on the cube and ~0 when far. Multiplying by the lifted indicator pays
    this term only while the cube is held off the table, rewarding STAYING in contact during
    the lift instead of bumping the cube and withdrawing.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    cube_pos_w = obj.data.root_pos_w
    ee_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    d_ee_cube = torch.norm(cube_pos_w - ee_w, dim=1)
    lifted = (cube_pos_w[:, 2] > minimal_height).float()
    gate = 1 - torch.tanh(d_ee_cube / std_gate)
    return lifted * gate


def block_lifted_and_grasped(
    env: ManagerBasedRLEnv,
    lift_height: float,
    contact_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[EE_BODY_NAME]),
) -> torch.Tensor:
    """Binary success indicator; sets ``env._reach_success`` for harbor eval success_rate.

    Success = cube world z > ``lift_height`` (15 cm above the table) AND the EE is within
    ``contact_threshold`` of the cube CoM (finger-contact proxy, since this robot has no
    contact sensor wired). EFFECTIVELY LOGGING-ONLY (weight 1e-6 in the cfg): the negligible
    positive weight forces RewardManager to run the func so the side-effect fires (the
    manager SKIPS a term whose weight is exactly 0.0), while contributing no meaningful
    optimization signal. Harbor train.py reads ``_episode_sums["success"]`` for success_rate.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]
    cube_pos_w = obj.data.root_pos_w
    ee_w = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    d_ee_cube = torch.norm(cube_pos_w - ee_w, dim=1)
    lifted = (cube_pos_w[:, 2] > lift_height).float()
    contact = (d_ee_cube < contact_threshold).float()
    result = lifted * contact
    env._reach_success = result
    return result
