#!/usr/bin/env python3
# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""在 MuJoCo 里回放 Isaac/skrl 训练的底盘靠近策略（sim-to-sim）。

不依赖 Isaac Lab。直接加载 skrl checkpoint（``best_agent.pt``）里的策略 MLP 和
``RunningStandardScaler``，在 MuJoCo 中复现训练时的观测/动作契约：

  观测 (13)  = bbox(7) + chassis_vel(2) + lift(1) + last_action(3)
  动作 (3)   = [a_v, a_w, a_lift] ∈ [-1,1]，仿射映射到差速驱动 + 升降杆

bbox 是*解析*目标 3D 框（不渲染），从 MuJoCo 的 camera_link 站点位姿与目标位姿
按 ``mdp/observations.py:get_clean_state`` 的几何复算。每回合按
``mdp/events.py:ResetRobotTargetInView`` 把目标放进相机视锥保证初始可见。
成功/碰撞/丢失判定镜像 ``mdp/terminations.py:TaskOutcome``。

用法::

    conda run -n realman python sim2sim/play_mujoco.py            # 打开 viewer
    conda run -n realman python sim2sim/play_mujoco.py --headless --episodes 5
"""
from __future__ import annotations

import argparse
import os
import time

import mujoco
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SCENE = os.path.join(HERE, "chassis_scene.xml")
DEFAULT_CKPT = os.path.join(
    REPO, "logs", "skrl", "chassis_approach",
    "2026-06-15_17-49-01_ppo_torch", "checkpoints", "best_agent.pt",
)

# --------------------------------------------------------------------------- #
# 任务常量（镜像 source/.../approach/approach_env_cfg.py + mdp/actions.py）
# --------------------------------------------------------------------------- #
CONTROL_HZ = 10.0
PHYSICS_HZ = 100.0
DECIM = int(round(PHYSICS_HZ / CONTROL_HZ))   # 10 物理步 / 控制步
DT_CTRL = 1.0 / CONTROL_HZ
DT_PHYS = 1.0 / PHYSICS_HZ
EPISODE_S = 15.0
MAX_CTRL_STEPS = int(EPISODE_S * CONTROL_HZ)

# 底盘按 SE(2) 速度体驱动（复现 Isaac：速度控制轮 + 每物理子步 planar-lock，
# 等效于一个被速度命令直接驱动、保持水平的浮动底盘）。底盘以固定离地高度滑行，
# 升降杆/手臂仍走物理；驱动轮由执行器空转以提供正确的滚动视觉。
REST_Z = 0.245                                 # 底盘静置离地高度（脚轮贴地）

V_RANGE = (-0.2, 0.5)
W_RANGE = (-1.0, 1.0)
WHEEL_R = 0.075
WHEEL_BASE = 0.296
HALF_BASE = 0.5 * WHEEL_BASE
LEFT_SIGN = -1.0
RIGHT_SIGN = 1.0
# 底盘命令 -> 实际运动的符号（由本 URDF 的轮轴/驱动符号约定决定，经回放校准）：
# 该差速驱动里正的 v_cmd / w_cmd 实际产生 -X / 顺时针运动，故取负号把命令映射到
# 物理运动（同时 vel.fwd / vel.yaw 观测=实际，保持与训练一致）。
DRIVE_SIGN_V = -1.0
DRIVE_SIGN_W = -1.0
LIFT_RANGE_ACT = (0.0, 1.0)   # 动作裁剪范围（DifferentialDriveActionCfg 默认）
LIFT_SPEED = 0.15

CAM_HFOV = 1.204
CAM_VFOV = 0.75
TAN_H = float(np.tan(0.5 * CAM_HFOV))
TAN_V = float(np.tan(0.5 * CAM_VFOV))
HEAD_PITCH_DOWN = (500.0 - 400.0) / 220.0     # ≈0.4545 rad，相机光轴下倾
_CP, _SP = float(np.cos(HEAD_PITCH_DOWN)), float(np.sin(HEAD_PITCH_DOWN))
FWD_AXIS = np.array([_CP, 0.0, -_SP])         # camera_link 局部：前(下压26°)
RIGHT_AXIS = np.array([0.0, -1.0, 0.0])       # 右
UP_AXIS = np.array([_SP, 0.0, _CP])           # 上
TARGET_SIZE = np.array([0.25, 0.25, 0.40])    # 目标方块完整尺寸（与 MJCF 一致）
HALF = 0.5 * TARGET_SIZE
_CORNER_SIGNS = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                         dtype=float)

# 成功/碰撞/丢失（镜像 TerminationsCfg）
D_TARGET = 0.4
CENTER_TOL = 0.25
DIST_TOL = 0.2
COLLISION_DIST = 0.25
SUCCESS_DWELL = 5
LOST_DWELL = 10

# 重置（镜像 ResetRobotTargetInView 参数）
YAW_RANGE = (-np.pi, np.pi)
XY_JITTER = 0.2
LIFT_RESET = (0.2, 0.5)
D_RANGE = (1.2, 4.0)
MARGIN_H = 0.8
MARGIN_V = 0.7
Z_FLOOR = 0.1
EPS = 1e-3


# --------------------------------------------------------------------------- #
# 策略（skrl GaussianMixin 的确定性均值前向 + RunningStandardScaler）
# --------------------------------------------------------------------------- #
def _elu(x):
    return np.where(x > 0, x, np.exp(np.minimum(x, 0.0)) - 1.0)


class Policy:
    """从 checkpoint 复现 net_container(256,128,64 ELU) -> policy_layer 的均值。"""

    def __init__(self, ckpt_path: str, clip_threshold: float = 5.0):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        p = ck["policy"]
        self.W = [p[f"net_container.{i}.weight"].numpy() for i in (0, 2, 4)]
        self.b = [p[f"net_container.{i}.bias"].numpy() for i in (0, 2, 4)]
        self.Wout = p["policy_layer.weight"].numpy()
        self.bout = p["policy_layer.bias"].numpy()
        sp = ck["state_preprocessor"]
        self.mean = sp["running_mean"].numpy()
        self.std = np.sqrt(sp["running_variance"].numpy() + 1e-8)
        self.clip = clip_threshold
        assert self.W[0].shape[1] == 18, f"obs dim mismatch: {self.W[0].shape}"

    def act(self, obs: np.ndarray) -> np.ndarray:
        x = (obs - self.mean) / self.std
        x = np.clip(x, -self.clip, self.clip)
        for W, b in zip(self.W, self.b):
            x = _elu(x @ W.T + b)
        return x @ self.Wout.T + self.bout   # 均值动作（确定性回放）


# --------------------------------------------------------------------------- #
# 几何小工具
# --------------------------------------------------------------------------- #
def quat_to_mat(q: np.ndarray) -> np.ndarray:
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def yaw_to_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)])


def rand_quat(rng) -> np.ndarray:
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)


class Sim:
    def __init__(self, rng):
        self.m = mujoco.MjModel.from_xml_path(SCENE)
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = 1.0 / PHYSICS_HZ
        self.rng = rng
        # 索引
        self.cam_sid = self.m.site("camera_link").id
        self.base_bid = self.m.body("base_link_underpan").id
        self.free_jid = self.m.joint("base_free").id
        self.free_qadr = self.m.jnt_qposadr[self.free_jid]
        self.free_vadr = self.m.jnt_dofadr[self.free_jid]
        self.lift_qadr = self.m.jnt_qposadr[self.m.joint("platform_joint").id]
        self.act = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
                    for i in range(self.m.nu)}
        self.aL = self.act["joint_left_wheel_v"]
        self.aR = self.act["joint_right_wheel_v"]
        self.aLift = self.act["platform_joint_p"]
        self.a_prev1 = np.zeros(3)   # a_{t-1}
        self.a_prev2 = np.zeros(3)   # a_{t-2}
        self.success_count = 0
        self.lost_count = 0
        # SE(2) 底盘状态（运动学驱动）
        self.bx = 0.0
        self.by = 0.0
        self.byaw = 0.0
        self.v_cmd = 0.0
        self.w_cmd = 0.0
        self.vfx = 0.0   # 实际前向速度 = DRIVE_SIGN_V * v_cmd
        self.wz = 0.0    # 实际偏航角速度 = DRIVE_SIGN_W * w_cmd

    # -- 观测 ------------------------------------------------------------- #
    def _bbox_clean(self):
        """解析目标 3D 框观测（9 维，相机系；不可见则置零）。

        镜像重训配置（env.yaml）：[x, y, z, distance, sx, sy, sz, u_ndc, v_ndc]
        —— 3D 中心 + 距离 + 沿相机轴的框边长 + 图像归一化中心。
        """
        cam_pos = self.d.site(self.cam_sid).xpos.copy()
        R_cam = self.d.site(self.cam_sid).xmat.reshape(3, 3).copy()
        tgt_pos = self.d.mocap_pos[0].copy()
        tgt_quat = self.d.mocap_quat[0].copy()

        c_cam = R_cam.T @ (tgt_pos - cam_pos)        # 目标中心于相机局部系
        fwd = float(c_cam @ FWD_AXIS)
        right = float(c_cam @ RIGHT_AXIS)
        up = float(c_cam @ UP_AXIS)
        x, y, z = right, -up, fwd
        distance = float(np.linalg.norm(c_cam))
        fwd_safe = max(fwd, EPS)
        u_ndc = (right / fwd_safe) / TAN_H
        v_ndc = (up / fwd_safe) / TAN_V
        visible = (fwd > EPS) and (abs(u_ndc) <= 1.0) and (abs(v_ndc) <= 1.0)

        # 8 角 -> 相机轴 AABB 边长
        R_t = quat_to_mat(tgt_quat)
        corners_w = (R_t @ (_CORNER_SIGNS * HALF).T).T + tgt_pos      # (8,3)
        pc = (corners_w - cam_pos) @ R_cam                            # 相机局部 (8,3)
        rc = pc @ RIGHT_AXIS
        uc = pc @ UP_AXIS
        fc = pc @ FWD_AXIS
        size = [rc.max() - rc.min(), uc.max() - uc.min(), fc.max() - fc.min()]

        obs9 = (np.array([x, y, z, distance, *size, u_ndc, v_ndc])
                if visible else np.zeros(9))
        return obs9, u_ndc, v_ndc, distance, visible

    def observe(self):
        bbox, u, v, dist, vis = self._bbox_clean()
        # chassis_vel：基框架前向线速度 + 偏航角速度
        v_world = self.d.qvel[self.free_vadr:self.free_vadr + 3].copy()
        w_local = self.d.qvel[self.free_vadr + 3:self.free_vadr + 6].copy()
        R_base = self.d.body(self.base_bid).xmat.reshape(3, 3)
        v_fwd = float((R_base.T @ v_world)[0])
        w_yaw = float(w_local[2])
        lift = float(self.d.qpos[self.lift_qadr])
        # last_actions：history_length=2 -> [a_{t-1}, a_{t-2}]（flatten，最近在前）
        obs = np.concatenate([bbox, [v_fwd, w_yaw], [lift],
                              self.a_prev1, self.a_prev2])
        return obs.astype(np.float64), (u, v, dist, vis)

    # -- 动作 ------------------------------------------------------------- #
    def apply_action(self, a_raw: np.ndarray):
        self.a_prev2 = self.a_prev1
        self.a_prev1 = a_raw.copy()
        a = np.clip(a_raw, -1.0, 1.0)
        self.v_cmd = 0.5 * (V_RANGE[1] + V_RANGE[0]) + 0.5 * (V_RANGE[1] - V_RANGE[0]) * a[0]
        self.w_cmd = 0.5 * (W_RANGE[1] + W_RANGE[0]) + 0.5 * (W_RANGE[1] - W_RANGE[0]) * a[1]
        self.vfx = DRIVE_SIGN_V * self.v_cmd     # 实际前向速度
        self.wz = DRIVE_SIGN_W * self.w_cmd      # 实际偏航角速度
        # 驱动轮执行器：仅用于滚动视觉（底盘由 SE(2) 运动学推进）。
        v_left = self.v_cmd - self.w_cmd * HALF_BASE
        v_right = self.v_cmd + self.w_cmd * HALF_BASE
        self.d.ctrl[self.aL] = LEFT_SIGN * v_left / WHEEL_R
        self.d.ctrl[self.aR] = RIGHT_SIGN * v_right / WHEEL_R
        # 升降杆：位置目标（走物理）。
        lift_vel = a[2] * LIFT_SPEED
        cur = float(self.d.qpos[self.lift_qadr])
        self.d.ctrl[self.aLift] = float(np.clip(cur + lift_vel * DT_CTRL,
                                                LIFT_RANGE_ACT[0], LIFT_RANGE_ACT[1]))
        # 其余上身执行器（hold @ 0）保持默认 ctrl=0

    def _write_base(self):
        """把 SE(2) 状态写进 freejoint：固定离地高度，水平，速度=命令值。"""
        q = self.free_qadr
        self.d.qpos[q + 0] = self.bx
        self.d.qpos[q + 1] = self.by
        self.d.qpos[q + 2] = REST_Z
        self.d.qpos[q + 3:q + 7] = yaw_to_quat(self.byaw)
        v = self.free_vadr
        self.d.qvel[v + 0] = self.vfx * np.cos(self.byaw)     # 世界系线速度
        self.d.qvel[v + 1] = self.vfx * np.sin(self.byaw)
        self.d.qvel[v + 2] = 0.0
        self.d.qvel[v + 3] = 0.0
        self.d.qvel[v + 4] = 0.0
        self.d.qvel[v + 5] = self.wz                          # 偏航角速度（局部）

    def step_control(self):
        for _ in range(DECIM):
            # SE(2) 积分（unicycle）：先转后平移
            self.byaw += self.wz * DT_PHYS
            self.bx += self.vfx * np.cos(self.byaw) * DT_PHYS
            self.by += self.vfx * np.sin(self.byaw) * DT_PHYS
            mujoco.mj_step(self.m, self.d)
            self._write_base()                   # 覆盖底盘积分，保持运动学驱动
        mujoco.mj_forward(self.m, self.d)        # 刷新覆盖后的运动学供观测

    # -- 重置 ------------------------------------------------------------- #
    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        rng = self.rng
        # 底盘：随机偏航 + xy 抖动（SE(2) 运动学状态）
        self.v_cmd = self.w_cmd = self.vfx = self.wz = 0.0
        self.bx = rng.uniform(-XY_JITTER, XY_JITTER)
        self.by = rng.uniform(-XY_JITTER, XY_JITTER)
        self.byaw = rng.uniform(*YAW_RANGE)
        self._write_base()
        # 升降杆
        lift0 = rng.uniform(*LIFT_RESET)
        self.d.qpos[self.lift_qadr] = lift0
        mujoco.mj_forward(self.m, self.d)
        # 目标放进视锥（保证初始可见）
        cam_pos = self.d.site(self.cam_sid).xpos.copy()
        R_cam = self.d.site(self.cam_sid).xmat.reshape(3, 3).copy()
        fwd_w = R_cam @ FWD_AXIS
        right_w = R_cam @ RIGHT_AXIS
        up_w = R_cam @ UP_AXIS
        u0 = rng.uniform(-MARGIN_H, MARGIN_H)
        v0 = rng.uniform(-MARGIN_V, MARGIN_V)
        dir_w = fwd_w + u0 * TAN_H * right_w + v0 * TAN_V * up_w
        dz = dir_w[2]
        d_cap = (cam_pos[2] - Z_FLOOR) / (-dz) if dz < -1e-4 else D_RANGE[1]
        d_hi = min(D_RANGE[1], d_cap)
        d_lo = min(D_RANGE[0], d_hi)
        d = rng.uniform(d_lo, d_hi)
        tgt_pos = cam_pos + d * dir_w
        self.d.mocap_pos[0] = tgt_pos
        self.d.mocap_quat[0] = rand_quat(rng)
        # 升降杆位置目标对齐
        self.d.ctrl[self.aLift] = lift0
        self.a_prev1 = np.zeros(3)
        self.a_prev2 = np.zeros(3)
        self.success_count = 0
        self.lost_count = 0
        mujoco.mj_forward(self.m, self.d)

    # -- 终止判定（镜像 TaskOutcome）------------------------------------- #
    def outcome(self, u, v, dist, vis):
        centred = np.hypot(u, v) < CENTER_TOL
        dist_ok = abs(dist - D_TARGET) < DIST_TOL
        good = vis and centred and dist_ok
        self.success_count = self.success_count + 1 if good else 0
        success = self.success_count >= SUCCESS_DWELL
        base_xy = self.d.qpos[self.free_qadr:self.free_qadr + 2]
        tgt_xy = self.d.mocap_pos[0][:2]
        collision = float(np.linalg.norm(base_xy - tgt_xy)) < COLLISION_DIST
        out_of_frame = (not vis) or (abs(u) > 1.0) or (abs(v) > 1.0)
        self.lost_count = self.lost_count + 1 if out_of_frame else 0
        lost = self.lost_count >= LOST_DWELL
        failure = collision or lost
        return (success and not failure), failure, ("collision" if collision else
                                                    "lost" if lost else "")


def run(headless: bool, episodes: int, seed: int, fast: bool, ckpt: str):
    rng = np.random.default_rng(seed)
    sim = Sim(rng)
    policy = Policy(ckpt)
    print(f"[play] scene={SCENE}")
    print(f"[play] ckpt ={ckpt}")

    def run_episode(viewer=None):
        sim.reset()
        obs, info = sim.observe()
        for t in range(MAX_CTRL_STEPS):
            a = policy.act(obs)
            sim.apply_action(a)
            sim.step_control()
            obs, info = sim.observe()
            u, v, dist, vis = info
            success, failure, why = sim.outcome(u, v, dist, vis)
            if viewer is not None:
                viewer.sync()
                if not fast:
                    time.sleep(DT_CTRL)   # 默认按 10 Hz 实时回放
            if success or failure:
                tag = "SUCCESS" if success else f"FAIL({why})"
                print(f"  [t={t:3d}] {tag}  dist={dist:.3f} center=({u:+.2f},{v:+.2f})")
                return tag
        print(f"  [t={MAX_CTRL_STEPS}] TIMEOUT  dist={dist:.3f} center=({u:+.2f},{v:+.2f})")
        return "TIMEOUT"

    if headless:
        tally = {}
        for ep in range(episodes):
            print(f"episode {ep}")
            r = run_episode()
            tally[r.split("(")[0]] = tally.get(r.split("(")[0], 0) + 1
        print("summary:", tally)
    else:
        import mujoco.viewer
        with mujoco.viewer.launch_passive(sim.m, sim.d) as viewer:
            ep = 0
            while viewer.is_running() and (episodes <= 0 or ep < episodes):
                print(f"episode {ep}")
                run_episode(viewer)
                ep += 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="无窗口，跑统计")
    ap.add_argument("--episodes", type=int, default=0, help="回合数（viewer 下 0=无限）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true", help="viewer 下全速回放（默认 10Hz 实时）")
    ap.add_argument("--ckpt", type=str, default=DEFAULT_CKPT, help="策略 checkpoint 路径")
    args = ap.parse_args()
    if args.headless and args.episodes <= 0:
        args.episodes = 5
    run(args.headless, args.episodes, args.seed, args.fast, args.ckpt)
