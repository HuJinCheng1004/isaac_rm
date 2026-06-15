# sim2sim — 在 MuJoCo 里回放 Isaac/skrl 训练的底盘靠近策略

把 `logs/skrl/chassis_approach/2026-06-12_20-16-11_ppo_torch/checkpoints/best_agent.pt`
（skrl PPO）搬到 MuJoCo 上回放，不依赖 Isaac Lab，环境就在 conda `realman` 里。

```
sim2sim/
  build_scene.py     # 由原始 MJCF 生成可仿真场景 chassis_scene.xml（不改原文件）
  chassis_scene.xml  # 生成物（浮动底盘 + 执行器 + 地面 + 目标）
  play_mujoco.py     # 加载 checkpoint，复现观测/动作契约，driveMuJoCo + viewer 回放
```

## 依赖（conda realman）

`realman` 已有 `mujoco 3.8.1`。本工具额外需要 CPU 版 PyTorch：

```bash
conda run -n realman pip install torch --index-url https://download.pytorch.org/whl/cpu
```

不需要 Isaac Lab / skrl —— 策略是个小 MLP，直接从 checkpoint 读权重和
`RunningStandardScaler` 并手写前向。

## 用法

```bash
# 1) 生成场景（已生成；改了机器人/参数再跑）
conda run -n realman python sim2sim/build_scene.py

# 2) 打开 viewer 实时回放（默认无限回合，关窗即停）
conda run -n realman python sim2sim/play_mujoco.py

# 全速 / 指定回合 / 指定随机种子
conda run -n realman python sim2sim/play_mujoco.py --fast --episodes 5 --seed 7

# 无窗口跑统计（服务器/无显示）
conda run -n realman python sim2sim/play_mujoco.py --headless --episodes 50
```

无显示环境想出图，用 `MUJOCO_GL=egl` + `mujoco.Renderer` 离屏渲染（见 git 历史里的
检查脚本）。

## 复现的“契约”（与训练严格对齐）

- **观测 13 维**（顺序取自保存的 `params/env.yaml`，**与当前源码不同**）：
  `bbox(4)[x,y,z,distance] + chassis_vel(2)[v_fwd,w_yaw] + lift(1) + last_actions(6)[a_{t-1},a_{t-2}]`。
  - bbox 是**解析**目标框（不渲染），从 MuJoCo 的 `camera_link` 站点位姿与目标位姿
    按相机光轴（含 26° 下倾 `CAM_AXES`）几何复算；不可见则置零。
  - **`RunningStandardScaler` 必须套用**（mean/var 来自 checkpoint，clip ±5）。
- **动作 3 维** ∈[-1,1]：`a_v→[-0.2,0.5] m/s`、`a_w→[-1,1] rad/s`、`a_lift→`速度积分到
  `platform_joint` 位置。
- 控制 10 Hz / 物理 100 Hz（decimation 10），回合 15 s。

## 两个关键工程决定

1. **底盘按 SE(2) 速度体驱动**（运动学），而非靠轮地摩擦。原因：导出 MJCF 里 4 个
   脚轮比两个驱动轮低 ~17 mm，凸包碰撞下驱动轮悬空、无法推进；而 Isaac 训练时底盘
   本就是“速度控制轮 + 每物理子步 planar-lock”——等效于一个被速度命令直接驱动、保持
   水平的浮动底盘。这里以固定离地高度滑行底盘，升降杆/手臂仍走物理，驱动轮由执行器
   空转以提供正确滚动视觉。

2. **驱动符号 `DRIVE_SIGN_V=-1`、`DRIVE_SIGN_W=-1`**（在 `play_mujoco.py` 顶部）。
   该 URDF 的轮轴/驱动符号约定使正的 `v_cmd`/`w_cmd` 实际产生 -X/顺时针运动，经回放
   校准取负号把命令映射到物理运动（同时 `vel.fwd`/`vel.yaw` 观测=实际，保持自洽）。
   判据：正确符号下机器人**靠近**目标（distance 单调下降）且**朝正确方向居中**。

## 这个 checkpoint 的真实水平

训练日志（tfevents）显示该策略**较弱**：整段训练的平均回合总奖励始终为**负**
（best≈-22），平均回合长度 89/150，成功奖励很小。回放观察到的行为与之一致：
机器人能**靠近到 ~0.5–0.7 m** 并大致对正，但很少满足严格成功带
（角度居中 `‖(u,v)‖<0.25` + `|dist-0.4|<0.2` 连续 5 步）。

根因是**任务本身**：策略输入的是**米制** bbox（目标贴近且位于下倾光轴下方时，水平
分量 `x` 只有几厘米→看似已居中），而成功判据用的是**角度** `u_ndc`（此时因深度很小
而偏大）。这个“米制 vs 角度”的错配在 Isaac 里同样存在，正是训练成功率低的原因。
也就是说：**sim-to-sim 是可行且忠实的，瓶颈在 checkpoint 本身**——换个训练更充分、
或把成功判据/观测口径对齐的模型，同一套回放即可复现更好的效果。
