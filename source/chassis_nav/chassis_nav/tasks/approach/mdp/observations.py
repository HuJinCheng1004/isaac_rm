# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""底盘靠近/对齐任务的观察。

中心部分是 :class:`BBox3DObservation`，一个*无渲染*的分析模型，
镜像 ``third_party/dino_sam/run_realtime.py`` 里 GroundingDINO+SAM2+深度
管线输出的 3D 包围盒（:class:`perception.box3d.Box3D`）。给定摄像头主体
姿态和目标物体姿态，它在**相机光学坐标系**（x 右、y 下、z 前，米——与
run_realtime 完全一致）下解析地给出目标的 3D 框，并返回:

    ``(x, y, z, distance, sx, sy, sz, u_ndc, v_ndc)``

其中 ``(x, y, z)`` 是目标中心在相机系下的米制坐标，``distance`` 是中心到
相机原点的距离（米，即 run_realtime 的 ``Box3D.distance``），``(sx, sy, sz)``
是目标 3D 框**沿相机坐标轴**（右/下/前）的边长（米），``(u_ndc, v_ndc)`` 是中心
投影到图像的归一化坐标（FOV 边缘=±1）——与奖励/成功判据同口径的居中角度信号
（等价于 2D 检测框中心，真机直接可得）。

真机部署用 run_realtime 的 ``--no-oriented``（AABB，``R = I``）：其边长就是点云
在相机轴上的范围，**不做主轴旋转/排序**。因此本观察也把目标 OBB 的 8 角变换到
相机系后，沿相机轴取 AABB 范围（随目标朝向变化）来对齐 —— 而非按 PCA 主轴降序。
若改用 oriented（PCA-OBB），size 约定会变（按主轴范围降序），需相应调整或重训。

该观察存在两种视图:

* **清晰/地真** (:meth:`BBox3DObservation.get_clean_state`) — 由奖励和终止
  使用。缓存一次每个环境步（用 ``env.common_step_counter`` 键控）使得
  终止 → 奖励 → 观察都共享一个计算，不管调用顺序。同时附带把中心投影到
  图像得到的 ``(u_ndc, v_ndc)``（用于居中/丢失判定）与 ``visible`` 标志。
* **损坏的** (:meth:`BBox3DObservation.__call__`) — *策略*看到的。添加
  sim-to-real 差异: 高斯抖动（米制噪声）、观察延迟（返回 ``t-1`` / ``t-2``
  帧），以及 5% 假阴性丢弃（整帧置零，对应 run_realtime 没有检测到目标）。

``visible``（目标在 FOV 内/镜头前面几何）是地真标志；*丢弃*帧仅是观察
伪影，**不**翻转 ``visible`` —— 因此代理学会滑行通过错过的检测，而不是
猛踩刹车。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple, TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _to_torch(x) -> torch.Tensor:
    """恒等通过（Isaac Lab 2.3.2 资产数据已经是 torch 张量）。

    作为命名助手保持，以便导入它的其他模块保持不变。
    """
    return x


# 存储在环境上的提供程序键，以便奖励/终止可以获取清晰框。
_PROVIDER_ATTR = "_chassis_bbox_provider"


class BBox3DState(NamedTuple):
    """清晰（地真）3D 框状态，供奖励/终止读取。"""

    center: torch.Tensor    # (N, 3) 目标中心 (x右, y下, z前) [m]
    distance: torch.Tensor  # (N,)   中心到相机原点距离 [m]
    size: torch.Tensor      # (N, 3) 3D 框边长（降序）[m]
    u_ndc: torch.Tensor     # (N,)   中心水平投影 (FOV 边缘 = +/-1)
    v_ndc: torch.Tensor     # (N,)   中心垂直投影 (FOV 边缘 = +/-1)
    visible: torch.Tensor   # (N,)   目标中心在镜头前且落在画面内
    obs: torch.Tensor       # (N, 9) 策略可见的清晰观察 [x,y,z,dist,sx,sy,sz,u_ndc,v_ndc]


def get_bbox3d_state(env: "ManagerBasedEnv") -> BBox3DState:
    """获取清晰的 3D 框状态（:class:`BBox3DState`）。

    由奖励和终止项使用。要求 :class:`BBox3DObservation` 项存在于观察配置中
    （它在 ``env`` 上注册自己）。
    """
    provider: BBox3DObservation = getattr(env, _PROVIDER_ATTR, None)
    if provider is None:
        raise RuntimeError(
            "No BBox3DObservation registered on the env. Add the `bbox` observation"
            " term to the policy observation group."
        )
    return provider.get_clean_state(env)


class BBox3DObservation(ManagerTermBase):
    """解析目标 3D 边界框观察（相机系），具有 sim-to-real 损坏。"""

    def __init__(self, cfg: ObservationTermCfg, env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        p = cfg.params
        self.robot_name: str = p.get("robot_name", "robot")
        self.target_name: str = p.get("target_name", "target")
        self.camera_body: str = p.get("camera_body", "camera_link")

        # 针孔半角度（仅用于把中心投影成 (u_ndc, v_ndc) 做居中/丢失判定）
        hfov: float = float(p.get("hfov", 1.204))  # ~69 度（RealSense D435 彩色）
        vfov: float = float(p.get("vfov", 0.75))   # ~43 度
        self._tan_half_h = float(torch.tan(torch.tensor(0.5 * hfov)))
        self._tan_half_v = float(torch.tan(torch.tensor(0.5 * vfov)))

        # 目标 3D 框半范围（米）。真机部署用 run_realtime 的 --no-oriented（AABB，
        # R=I），其 size 是点云沿“相机坐标轴”(右,下,前)的范围，不做主轴旋转/排序。
        # 因此清晰 size 由目标 OBB 的 8 角变换到相机系后沿相机轴取 AABB 范围得到
        # （随目标朝向变化，见 get_clean_state）；损坏阶段再加米制噪声。
        size = p.get("target_size", (0.25, 0.25, 0.40))
        half = [0.5 * float(s) for s in size]
        self._half = torch.tensor(half, dtype=torch.float32, device=self.device)
        # 单位框 8 角符号组合 (8,3)
        self._corner_signs = torch.tensor(
            [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
            dtype=torch.float32, device=self.device,
        )

        # 摄像头-局部轴约定（前向 = 光学轴，右/上 = 图像）。
        # 默认：前向=+x，右=-y，上=+z（ROS 主体框架 -> 图像）。
        axes = p.get("camera_axes", {"forward": (1.0, 0.0, 0.0), "right": (0.0, -1.0, 0.0), "up": (0.0, 0.0, 1.0)})
        self._fwd_axis = torch.tensor(axes["forward"], dtype=torch.float32, device=self.device)
        self._right_axis = torch.tensor(axes["right"], dtype=torch.float32, device=self.device)
        self._up_axis = torch.tensor(axes["up"], dtype=torch.float32, device=self.device)

        # sim-to-real 损坏参数：噪声为 9 维标准差。前 7 维米制 (x,y,z,dist,sx,sy,sz)，
        # 后 2 维为图像归一化坐标 (u_ndc, v_ndc) —— 直接暴露居中角度信号，与奖励/成功
        # 判据（用 u_ndc/v_ndc）口径一致，避免策略只能从 x/z 反算（近距且目标在下倾光轴
        # 下方时 z→0、该比值病态）。
        self._noise_std = torch.tensor(
            p.get("noise_std", (0.02, 0.02, 0.04, 0.04, 0.02, 0.02, 0.03, 0.02, 0.02)),
            dtype=torch.float32, device=self.device,
        )
        self._dropout_prob = float(p.get("dropout_prob", 0.05))
        lat = p.get("latency_steps", (1, 2))
        self._min_lat, self._max_lat = int(lat[0]), int(lat[1])

        # 解析机器人关节上的摄像头主体索引
        robot = env.scene[self.robot_name]
        cam_ids, cam_names = robot.find_bodies(self.camera_body)
        if len(cam_ids) != 1:
            raise ValueError(
                f"Expected exactly one camera body matching '{self.camera_body}', got {cam_names}. "
                "If the URDF importer merged the fixed-jointed camera link, set merge_fixed_joints=False "
                "in the robot spawn config, or point `camera_body` at 'head_link2'."
            )
        self._cam_idx = cam_ids[0]

        # 观察维度（策略可见向量长度）：3D 框 7 维 + 图像归一化中心 (u_ndc, v_ndc)。
        self._obs_dim = 9

        # 延迟环形缓冲（索引 0 = 最新），每环延迟，步骤簿记
        self._buffer = torch.zeros(self.num_envs, self._max_lat + 1, self._obs_dim, device=self.device)
        self._latency = torch.randint(self._min_lat, self._max_lat + 1, (self.num_envs,), device=self.device)
        self._last_roll_step = -1

        # 每步清晰状态缓存
        self._cache: BBox3DState | None = None
        self._cache_step = -1

        # 注册提供程序，以便奖励/终止可以读取清晰的 3D 框
        setattr(env, _PROVIDER_ATTR, self)

    # ----------------------------------------------------------------------- #
    # 地真几何（每步缓存一次）。
    # ----------------------------------------------------------------------- #
    def get_clean_state(self, env: "ManagerBasedEnv") -> BBox3DState:
        step = int(env.common_step_counter)
        if self._cache is not None and self._cache_step == step:
            return self._cache

        n = self.num_envs
        robot = env.scene[self.robot_name]
        target = env.scene[self.target_name]

        cam_pos = robot.data.body_pos_w[:, self._cam_idx]    # (N,3) 世界
        cam_quat = robot.data.body_quat_w[:, self._cam_idx]  # (N,4) wxyz (Isaac Lab 2.3.2)
        tgt_pos = target.data.root_pos_w                     # (N,3) 世界
        tgt_quat = target.data.root_quat_w                   # (N,4) wxyz

        # 目标中心在相机主体系
        c_cam = math_utils.quat_apply_inverse(cam_quat, tgt_pos - cam_pos)  # (N,3)
        fwd = (c_cam * self._fwd_axis).sum(-1)      # 沿光学轴的深度 (z 前)
        right = (c_cam * self._right_axis).sum(-1)  # x 右
        up = (c_cam * self._up_axis).sum(-1)        # 图像上

        # 相机光学系下的 3D 中心（run_realtime 约定：x 右, y 下, z 前）
        x = right
        y = -up
        z = fwd
        center = torch.stack([x, y, z], dim=-1)             # (N,3) [m]
        distance = torch.linalg.norm(c_cam, dim=-1)         # (N,) 中心到相机原点 [m]

        # 把中心投影到归一化图像坐标（FOV 边缘 = +/-1），用于居中/丢失判定
        eps = 1e-3
        fwd_safe = torch.clamp(fwd, min=eps)
        u_ndc = (right / fwd_safe) / self._tan_half_h
        v_ndc = (up / fwd_safe) / self._tan_half_v

        # 可见 = 中心在镜头前 且 投影落在画面内（检测器能看到目标中心）
        visible = (fwd > eps) & (u_ndc.abs() <= 1.0) & (v_ndc.abs() <= 1.0)

        # 3D 框边长（米）：把目标 OBB 的 8 角变换到相机系，沿相机轴 (右/下/前) 取
        # AABB 范围 —— 对齐 run_realtime 的 --no-oriented（R=I，不旋转/排序主轴）。
        corners_local = (self._corner_signs * self._half).unsqueeze(0).expand(n, 8, 3).reshape(n * 8, 3)
        tq = tgt_quat.unsqueeze(1).expand(n, 8, 4).reshape(n * 8, 4)
        corners_world = math_utils.quat_apply(tq, corners_local).reshape(n, 8, 3) + tgt_pos.unsqueeze(1)
        rel_c = (corners_world - cam_pos.unsqueeze(1)).reshape(n * 8, 3)
        cq = cam_quat.unsqueeze(1).expand(n, 8, 4).reshape(n * 8, 4)
        pc = math_utils.quat_apply_inverse(cq, rel_c).reshape(n, 8, 3)  # 8 角在相机主体系
        right_c = (pc * self._right_axis).sum(-1)   # (N,8) 沿相机 x(右)
        up_c = (pc * self._up_axis).sum(-1)         # (N,8) 沿相机 y(上/下，范围与符号无关)
        fwd_c = (pc * self._fwd_axis).sum(-1)       # (N,8) 沿相机 z(前/深度)
        size_x = right_c.amax(dim=1) - right_c.amin(dim=1)
        size_y = up_c.amax(dim=1) - up_c.amin(dim=1)
        size_z = fwd_c.amax(dim=1) - fwd_c.amin(dim=1)
        size = torch.stack([size_x, size_y, size_z], dim=-1)  # (N,3) 相机轴 AABB 边长

        ndc = torch.stack([u_ndc, v_ndc], dim=-1)                       # (N,2) 图像归一化中心
        obs = torch.cat([center, distance.unsqueeze(-1), size, ndc], dim=-1)  # (N,9)
        obs = torch.where(visible.unsqueeze(-1), obs, torch.zeros_like(obs))

        # NaN/Inf 兜底：若某个环境物理求解器爆炸（body_pos_w 出现 nan/inf），不让它经
        # 观测/奖励污染整批 PPO 更新（旧 run 在 step7200 整网权重变 NaN 后 5 万步全废）。
        # 把坏环境视作"不可见"（触发丢失失败、自然重置），并把数值字段清零。
        finite = torch.isfinite(obs).all(dim=-1) & torch.isfinite(distance) \
            & torch.isfinite(u_ndc) & torch.isfinite(v_ndc)
        visible = visible & finite
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        u_ndc = torch.nan_to_num(u_ndc, nan=0.0, posinf=0.0, neginf=0.0)
        v_ndc = torch.nan_to_num(v_ndc, nan=0.0, posinf=0.0, neginf=0.0)
        distance = torch.nan_to_num(distance, nan=0.0, posinf=0.0, neginf=0.0)
        center = torch.nan_to_num(center, nan=0.0, posinf=0.0, neginf=0.0)
        size = torch.nan_to_num(size, nan=0.0, posinf=0.0, neginf=0.0)

        self._cache = BBox3DState(
            center=center, distance=distance, size=size,
            u_ndc=u_ndc, v_ndc=v_ndc, visible=visible, obs=obs,
        )
        self._cache_step = step
        return self._cache

    # ----------------------------------------------------------------------- #
    # 损坏的策略观察。
    # ----------------------------------------------------------------------- #
    def __call__(
        self,
        env: "ManagerBasedEnv",
        robot_name: str = "robot",
        target_name: str = "target",
        camera_body: str = "camera_link",
        hfov: float = 1.204,
        vfov: float = 0.75,
        camera_axes: dict | None = None,
        target_size: tuple = (0.25, 0.25, 0.40),
        noise_std: tuple = (0.02, 0.02, 0.04, 0.04, 0.02, 0.02, 0.03, 0.02, 0.02),
        dropout_prob: float = 0.05,
        latency_steps: tuple = (1, 2),
    ) -> torch.Tensor:
        # 注意：所有参数在 __init__ 中被使用（存储在自身）；它们
        # 在此处列出仅因为 Isaac Lab 2.3.2 验证 __call__ 签名。
        state = self.get_clean_state(env)
        clean, visible = state.obs, state.visible

        # 检测器抖动（仅当实际上有东西可见时）
        det = clean.clone()
        noise = torch.randn_like(det) * self._noise_std
        det = det + noise * visible.unsqueeze(-1).float()

        # 每个环境步骤恰好推进延迟环形缓冲一次
        step = int(env.common_step_counter)
        if step != self._last_roll_step:
            self._buffer[:, 1:] = self._buffer[:, :-1].clone()
            self._last_roll_step = step
        self._buffer[:, 0] = det  # newest frame

        # 按环境读取延迟的帧
        env_ids = torch.arange(self.num_envs, device=self.device)
        delayed = self._buffer[env_ids, self._latency].clone()

        # 5% 假阴性丢弃 -> 此帧"无检测"
        drop = torch.rand(self.num_envs, device=self.device) < self._dropout_prob
        delayed = torch.where(drop.unsqueeze(-1), torch.zeros_like(delayed), delayed)
        return delayed

    def reset(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._buffer[:] = 0.0
            self._latency[:] = torch.randint(
                self._min_lat, self._max_lat + 1, (self.num_envs,), device=self.device
            )
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self._buffer[env_ids_t] = 0.0
            self._latency[env_ids_t] = torch.randint(
                self._min_lat, self._max_lat + 1, (env_ids_t.numel(),), device=self.device
            )


def chassis_velocity(env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """本体感觉：基框架中的前向线速度和偏航速率，形状 (N, 2)。

    NaN/Inf 兜底 + 裁剪：若某环境物理求解器爆炸，``root_*_vel_b`` 可能变成 inf；
    本体感知不像 bbox 那样有兜底，inf 经 RunningStandardScaler 会把整个 PPO 更新
    污染成 NaN（旧 run 在 ~step45k 整网变 NaN 的根因）。这里清洗并裁到物理量级。
    """
    asset = env.scene[asset_cfg.name]
    v_fwd = _to_torch(asset.data.root_lin_vel_b)[:, 0:1]
    w_yaw = _to_torch(asset.data.root_ang_vel_b)[:, 2:3]
    out = torch.cat([v_fwd, w_yaw], dim=-1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).clamp(-10.0, 10.0)


def lift_height(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["platform_joint"]),
) -> torch.Tensor:
    """本体感觉：升降杆当前高度，形状 (N, 1)。

    策略用第 3 维动作控制升降杆来调节相机高度（从而调节目标在画面中的纵向位置，
    即 3D 中心的 y/距离），因此把当前高度暴露给策略以便闭环。

    同样做 NaN/Inf 兜底 + 裁剪（行程 0~1，留余量裁到 [-1, 2]），避免物理爆炸时污染网络。
    """
    asset = env.scene[asset_cfg.name]
    out = _to_torch(asset.data.joint_pos)[:, asset_cfg.joint_ids]
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 2.0)
