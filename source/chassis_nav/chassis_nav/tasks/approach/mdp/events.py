# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""底盘靠近/对齐任务的重置事件。

核心是 :class:`ResetRobotTargetInView`：在每次情节重置时，
*每个环境独立地*随机初始化机器人状态，并把目标方块放到相机视野内，
**保证初始帧的目标 bbox 落在图像内**，且目标的大小/位置充分随机。

随机化的机器人自由度：

* **底盘朝向** —— 根姿态偏航角（外加小幅 x/y 抖动）。
* **升降杆高度** —— ``platform_joint`` 棱柱关节，决定相机高度。

机械臂保持默认位姿（不随机）。头部相机固定在真机 home 位姿，其 26° 下倾
已“烤进”相机光轴（``camera_axes``，见 ``approach_env_cfg``），因此这里无需
驱动头部关节——物理头部留在默认 home。

升降杆由高刚度隐式执行器保持，因此除了写入关节状态外，还要调用
``set_joint_position_target`` 把目标位置设到新采样值，否则执行器会在几个
物理步内把它拉回默认高度。

目标是**悬浮在 3D 空间**的（关闭重力），可出现在相机视锥内任意位置，而不再
贴地。

**视野保证的原理。** 先写入机器人的新根姿态与升降杆高度，再读取
``body_pos_w`` —— 该属性会触发一次前向运动学更新，于是返回的相机世界位姿
已经反映了刚写入的新构型。随后我们直接在相机视锥内采样目标：

1. 在画面裕度内均匀采样图像坐标 ``(u0, v0) ∈ [-margin_h, margin_h] ×
   [-margin_v, margin_v]``（左右/上下位置随机）；
2. 在 ``[d_min, d_max]`` 内采样前向深度 ``d``（决定 bbox 大小，越近越大）；
3. 用*真实*相机四元数把视线变换到世界系，目标点 = ``cam_pos + d · dir``。
   由于视线的前向分量恒为 1，该点的相机前向深度即 ``d``，投影恒等于
   ``(u0, v0)`` —— 故只要 ``(u0, v0)`` 在裕度内，目标中心必然落在画面内。

朝下的视线会限制 ``d`` 上限以避免目标穿到地面以下（``z ≥ z_floor``）。目标
朝向随机取全 3D 欧拉角。对任意升降杆高度/底盘朝向，上述保证均成立。
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class ResetRobotTargetInView(ManagerTermBase):
    """随机化机器人状态并把目标放进相机视野（保证初始可见、大小/位置随机）。"""

    def __init__(self, cfg: EventTermCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        p = cfg.params
        self.robot_name: str = p.get("robot_name", "robot")
        self.target_name: str = p.get("target_name", "target")
        self.camera_body: str = p.get("camera_body", "camera_link")

        robot = env.scene[self.robot_name]

        # --- 解析相机主体与升降杆关节索引（仅一次） ---
        cam_ids, cam_names = robot.find_bodies(self.camera_body)
        if len(cam_ids) != 1:
            raise ValueError(
                f"期望恰好匹配一个相机主体 '{self.camera_body}'，实际得到 {cam_names}。"
            )
        self._cam_idx = cam_ids[0]
        self._plat_ids, _ = robot.find_joints(p.get("lift_joint", "platform_joint"))

        # --- 随机化范围 ---
        self._yaw_range = tuple(p.get("yaw_range", (-math.pi, math.pi)))
        self._xy_jitter = float(p.get("xy_jitter", 0.2))
        self._lift_range = tuple(p.get("lift_range", (0.2, 0.5)))

        # --- 相机针孔模型（必须与 BBox3DObservation 完全一致，含 26° 下倾光轴） ---
        hfov = float(p.get("hfov", 1.204))
        vfov = float(p.get("vfov", 0.75))
        self._tan_h = float(math.tan(0.5 * hfov))
        self._tan_v = float(math.tan(0.5 * vfov))
        axes = p.get(
            "camera_axes",
            {"forward": (1.0, 0.0, 0.0), "right": (0.0, -1.0, 0.0), "up": (0.0, 0.0, 1.0)},
        )
        self._fwd = torch.tensor(axes["forward"], dtype=torch.float32, device=self.device)
        self._right = torch.tensor(axes["right"], dtype=torch.float32, device=self.device)
        self._up = torch.tensor(axes["up"], dtype=torch.float32, device=self.device)

        # --- 目标放置参数（相机视锥内 3D 采样） ---
        d_range = tuple(p.get("d_range", (1.2, 4.0)))       # 相机前向深度范围 [m]（决定 bbox 大小）
        self._d_min, self._d_max = float(d_range[0]), float(d_range[1])
        self._margin_h = float(p.get("margin_h", 0.8))      # 水平视野裕度 (|u0| 上界)
        self._margin_v = float(p.get("margin_v", 0.7))      # 垂直视野裕度 (|v0| 上界)
        self._z_floor = float(p.get("z_floor", 0.1))        # 目标中心最低离地高度 [m]（防止穿地）

    def __call__(
        self,
        env: "ManagerBasedEnv",
        env_ids: torch.Tensor,
        robot_name: str = "robot",
        target_name: str = "target",
        camera_body: str = "camera_link",
        lift_joint: str = "platform_joint",
        yaw_range: tuple = (-math.pi, math.pi),
        xy_jitter: float = 0.2,
        lift_range: tuple = (0.2, 0.5),
        hfov: float = 1.204,
        vfov: float = 0.75,
        camera_axes: dict | None = None,
        d_range: tuple = (1.2, 4.0),
        margin_h: float = 0.8,
        margin_v: float = 0.7,
        z_floor: float = 0.1,
    ) -> None:
        # 参数在 __init__ 中被消费（存到 self）；此处列出仅为 2.3.2 的签名校验。
        robot = env.scene[self.robot_name]
        target = env.scene[self.target_name]
        n = len(env_ids)
        dev = self.device

        # --------------------------------------------------------------- #
        # 1) 底盘朝向：随机偏航 + 小幅 x/y 抖动（z 用默认、velocity 置零）。
        # --------------------------------------------------------------- #
        root = robot.data.default_root_state[env_ids].clone()  # (n, 13)
        pos = root[:, 0:3] + env.scene.env_origins[env_ids]
        pos[:, 0] += math_utils.sample_uniform(-self._xy_jitter, self._xy_jitter, (n,), dev)
        pos[:, 1] += math_utils.sample_uniform(-self._xy_jitter, self._xy_jitter, (n,), dev)

        yaw = math_utils.sample_uniform(self._yaw_range[0], self._yaw_range[1], (n,), dev)
        zeros = torch.zeros(n, device=dev)
        delta = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
        quat = math_utils.quat_mul(root[:, 3:7], delta)

        robot.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1), env_ids=env_ids)
        robot.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)

        # --------------------------------------------------------------- #
        # 2) 升降杆高度：写入关节状态并设定位置目标（由高刚度执行器保持住）。
        # --------------------------------------------------------------- #
        lift = math_utils.sample_uniform(
            self._lift_range[0], self._lift_range[1], (n, len(self._plat_ids)), dev
        )
        limits = robot.data.soft_joint_pos_limits[env_ids][:, self._plat_ids]  # (n, J, 2)
        lift = torch.max(torch.min(lift, limits[..., 1]), limits[..., 0])
        robot.write_joint_state_to_sim(lift, torch.zeros_like(lift), joint_ids=self._plat_ids, env_ids=env_ids)
        robot.set_joint_position_target(lift, joint_ids=self._plat_ids, env_ids=env_ids)

        # --------------------------------------------------------------- #
        # 3) 读取新构型下的相机世界位姿（触发前向运动学更新）。
        # --------------------------------------------------------------- #
        cam_pos = robot.data.body_pos_w[env_ids, self._cam_idx]    # (n, 3) 世界
        cam_quat = robot.data.body_quat_w[env_ids, self._cam_idx]  # (n, 4) wxyz

        # 世界系下的相机光轴基（前/右/上）。dir_local 的前向分量恒为 1，故
        # 视线 dir = fwd + u0*tan_h*right + v0*tan_v*up 上任一点的"前向深度"= d。
        fwd_w = math_utils.quat_apply(cam_quat, self._fwd.expand(n, 3))    # (n, 3)
        right_w = math_utils.quat_apply(cam_quat, self._right.expand(n, 3))
        up_w = math_utils.quat_apply(cam_quat, self._up.expand(n, 3))

        # --------------------------------------------------------------- #
        # 4) 在相机视锥内 3D 采样目标点（投影到画面 (u0,v0) 内 -> 保证可见；
        #    前向深度 d -> bbox 大小随机；目标随机全 3D 朝向）。
        # --------------------------------------------------------------- #
        u0 = math_utils.sample_uniform(-self._margin_h, self._margin_h, (n,), dev)  # 左右位置
        v0 = math_utils.sample_uniform(-self._margin_v, self._margin_v, (n,), dev)  # 上下位置
        # 视线方向（前向分量=1），目标 = cam_pos + d * dir，投影恒为 (u0, v0)。
        dir_world = fwd_w + (u0 * self._tan_h)[:, None] * right_w + (v0 * self._tan_v)[:, None] * up_w

        # 朝下的视线限制最大深度，避免目标穿到地面以下 (z >= z_floor)；
        # 然后在 [d_min, 不穿地的最大深度] 内均匀取深度，让目标沿视线散开（高度有变化），
        # 而不是全部被压到地板上。
        dz = dir_world[:, 2]
        d_cap = torch.where(
            dz < -1e-4, (cam_pos[:, 2] - self._z_floor) / (-dz), torch.full_like(dz, self._d_max)
        )
        d_hi = torch.minimum(torch.full_like(dz, self._d_max), d_cap)
        d_lo = torch.minimum(torch.full_like(dz, self._d_min), d_hi)  # 极陡视线时退化为贴地
        t = math_utils.sample_uniform(0.0, 1.0, (n,), dev)
        d = d_lo + t * (d_hi - d_lo)                                  # 前向深度 [m]（决定 bbox 大小）
        tgt_pos = cam_pos + d[:, None] * dir_world

        # 随机全 3D 朝向（悬浮目标，"上下左右旋转"任意）。
        rpy = math_utils.sample_uniform(-math.pi, math.pi, (n, 3), dev)
        tgt_quat = math_utils.quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])

        target.write_root_pose_to_sim(torch.cat([tgt_pos, tgt_quat], dim=-1), env_ids=env_ids)
        target.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)
