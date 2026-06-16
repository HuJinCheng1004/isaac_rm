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

目标是**悬浮在 3D 空间**的（关闭重力），出现在用户指定的世界高度区间内
（默认 0.4~1.6 m），而不再贴地或被推到远处。

**放置原理（以高度为第一性变量）。** 旧实现沿 26° 下倾视线采样“前向深度 d”，
导致大 d 时目标被推到地面附近 —— 那里“居中且距离 d_target”的成功区在几何上
不可达（是旧 run 0% 成功率的主因之一）。新实现改为**直接采样目标世界高度**：

1. 采样目标世界高度 ``z_t ∈ height_range``（夹到相机能俯视看到的上限）；
2. 采样相机余量 ``Δ ∈ delta_range``，令相机位于目标上方 ``Δ`` 处，反推升降杆
   高度并写入（相机高度 = ``cam0 + lift``，``cam0`` 由前向运动学探测一次后缓存）；
3. 读取*真实*相机位姿，在画面裕度内采样 ``(u0, v0)`` 得到视线方向，再解深度 ``d``
   使目标精确落在世界高度 ``z_t``：``cam_z + d·dir_z = z_t``。由于相机在目标上方
   （``dir_z < 0``）必有 ``d > 0``，且投影恒等于 ``(u0, v0)`` —— 故目标既精确处于
   指定高度，又必然落在画面内、且初始就处于“成功区可达”的几何配置。

目标朝向随机取全 3D 欧拉角。对任意底盘朝向，上述保证均成立。
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
        # 目标中心**世界高度**范围 [m]（地面 z=0）。这是放置目标的第一性变量：
        # 直接采样高度（而非沿 26° 下倾视线采样深度），避免目标被推到远处/地面附近导致
        # “居中且距离 d_target”的成功区在几何上不可达（旧实现 0% 成功率的主因之一）。
        self._height_range = tuple(p.get("height_range", (0.4, 1.6)))
        # 相机抬到目标上方的余量 Δ [m]：26° 下倾相机必须**位于目标上方**才能在 FOV 内
        # 看到它；Δ 同时决定初始俯视角与初始距离（d ≈ Δ/|dir_z| ≈ 0.6~2.5 m）。
        self._delta_range = tuple(p.get("delta_range", (0.35, 0.75)))

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

        # --- 目标放置参数 ---
        self._margin_h = float(p.get("margin_h", 0.5))      # 水平视野裕度 (|u0| 上界)
        self._margin_v = float(p.get("margin_v", 0.4))      # 垂直视野裕度 (|v0| 上界)
        self._lift_limits = tuple(p.get("lift_limits", (0.0, 1.0)))  # platform_joint 行程 [m]
        # 初始可见所需的“相机高于目标”的最小余量 [m]（低于它目标落在下倾 FOV 之上、看不到）
        self._frame_margin = float(p.get("frame_margin", 0.15))
        # lift=0 时相机世界高度（cam_z = _cam0 + lift）。首次 reset 由前向运动学探测后缓存。
        self._cam0: float | None = None

        # --- 课程支持 ---
        # 记录“满难度”范围；课程项（mdp.approach_difficulty）会随训练把激活的
        # ``self._yaw_range`` / ``self._height_range`` 从更易的初值线性退火到这些满值。
        self._yaw_full = float(self._yaw_range[1])
        self._height_full = tuple(self._height_range)
        env._reset_in_view_term = self

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
        height_range: tuple = (0.4, 1.6),
        delta_range: tuple = (0.35, 0.75),
        hfov: float = 1.204,
        vfov: float = 0.75,
        camera_axes: dict | None = None,
        margin_h: float = 0.5,
        margin_v: float = 0.4,
        lift_limits: tuple = (0.0, 1.0),
        frame_margin: float = 0.15,
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
        # 2) 探测 lift=0 时的相机世界高度 cam0（cam_z = cam0 + lift）。仅首次 reset。
        # --------------------------------------------------------------- #
        plat = self._plat_ids
        if self._cam0 is None:
            zero_lift = torch.zeros(n, len(plat), device=dev)
            robot.write_joint_state_to_sim(zero_lift, torch.zeros_like(zero_lift), joint_ids=plat, env_ids=env_ids)
            self._cam0 = float(robot.data.body_pos_w[env_ids, self._cam_idx, 2].mean().item())

        # --------------------------------------------------------------- #
        # 3) 采样目标世界高度 + 相机余量 Δ -> 反推升降杆高度并写入。
        #    目标高度夹到“相机俯视可见”的上限：相机最高 = cam0 + lift_max，需高于目标
        #    至少 frame_margin，否则目标落在 26° 下倾 FOV 之上而看不到。
        # --------------------------------------------------------------- #
        ground_z = env.scene.env_origins[env_ids, 2]                   # (n,) 地面高度
        h_lo, h_hi = float(self._height_range[0]), float(self._height_range[1])
        z_ceiling = (self._cam0 + self._lift_limits[1] - self._frame_margin) - ground_z  # 相对地面
        z_t = math_utils.sample_uniform(h_lo, h_hi, (n,), dev)
        z_t = torch.minimum(z_t, torch.clamp(z_ceiling, min=h_lo))     # 高目标夹到可达上限
        cam_margin = math_utils.sample_uniform(self._delta_range[0], self._delta_range[1], (n,), dev)
        cam_z_des = ground_z + z_t + cam_margin                        # 期望相机世界高度
        lift = torch.clamp(cam_z_des - self._cam0, self._lift_limits[0], self._lift_limits[1])
        lift = lift.unsqueeze(-1).expand(n, len(plat)).contiguous()
        limits = robot.data.soft_joint_pos_limits[env_ids][:, plat]    # (n, J, 2)
        lift = torch.max(torch.min(lift, limits[..., 1]), limits[..., 0])
        robot.write_joint_state_to_sim(lift, torch.zeros_like(lift), joint_ids=plat, env_ids=env_ids)
        robot.set_joint_position_target(lift, joint_ids=plat, env_ids=env_ids)

        # --------------------------------------------------------------- #
        # 4) 读最终相机位姿，在画面裕度内把目标放到**精确世界高度** z_t。
        #    相机已位于目标上方（dir_z<0），解深度 d 使 cam_z + d*dir_z = z_world。
        # --------------------------------------------------------------- #
        cam_pos = robot.data.body_pos_w[env_ids, self._cam_idx]    # (n, 3) 世界
        cam_quat = robot.data.body_quat_w[env_ids, self._cam_idx]  # (n, 4) wxyz
        fwd_w = math_utils.quat_apply(cam_quat, self._fwd.expand(n, 3))    # (n, 3)
        right_w = math_utils.quat_apply(cam_quat, self._right.expand(n, 3))
        up_w = math_utils.quat_apply(cam_quat, self._up.expand(n, 3))

        u0 = math_utils.sample_uniform(-self._margin_h, self._margin_h, (n,), dev)  # 左右位置
        v0 = math_utils.sample_uniform(-self._margin_v, self._margin_v, (n,), dev)  # 上下位置
        dir_world = fwd_w + (u0 * self._tan_h)[:, None] * right_w + (v0 * self._tan_v)[:, None] * up_w

        z_world = ground_z + z_t
        dz = torch.clamp(dir_world[:, 2], max=-1e-2)                   # 下视分量（防除零/朝上）
        d = ((z_world - cam_pos[:, 2]) / dz).clamp(min=0.2)            # 深度 [m]
        tgt_pos = cam_pos + d[:, None] * dir_world
        tgt_pos[:, 2] = z_world                                        # 强制精确高度

        # 随机全 3D 朝向（悬浮目标，"上下左右旋转"任意）。
        rpy = math_utils.sample_uniform(-math.pi, math.pi, (n, 3), dev)
        tgt_quat = math_utils.quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2])

        target.write_root_pose_to_sim(torch.cat([tgt_pos, tgt_quat], dim=-1), env_ids=env_ids)
        target.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)
