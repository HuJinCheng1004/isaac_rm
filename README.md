# chassis_nav — 底盘导航 RL 训练流水线 (Isaac Lab 2.3.2 + Isaac Sim 5.1 + skrl PPO)

一个**无渲染、基于抽象 BBox 特征**的差速底盘"靠近并对准目标"导航 RL 任务。
策略只看 ~10 维抽象特征（不输入图像），因此可以在单卡上开 4096 个并行环境
快速收敛，并通过域随机化做 sim-to-real。完整设计依据见 `require.txt`。

机器人使用你自己的差速底盘 URDF：
`/home/shihao/XRTeleop/assets/robot_discription/overseas_65_b_v_description/...`
（两个驱动轮 + 4 个万向轮 + 双臂/头部/相机；本任务只控制两个驱动轮，其余关节锁定）。

---

## 1. 设计如何对应 require.txt

| require.txt | 实现 |
|---|---|
| 状态空间 ~10 维 | `bbox`(4) + `chassis_vel`(2) + `last_actions`(4)，见 `mdp/observations.py` |
| 动作空间 2 维连续 (v, ω) | `DifferentialDriveAction`，映射到左右轮速度，见 `mdp/actions.py` |
| R_center / R_approach / R_smooth / R_terminal | `mdp/rewards.py` + 内置 `action_rate_l2` |
| BBox 噪声 / 5% 漏检 / 100ms 延迟 | `BBoxObservation` 内部 per-env 环形缓冲实现 |
| 物理 DR（质量/轮摩擦/电机±20%） | `EventsCfg`，复用内置 `randomize_rigid_body_mass/material/actuator_gains` |
| PPO + MLP[256,128,64]+ELU | `agents/skrl_ppo_cfg.yaml` |
| 4096 并行 / skrl 后端 | `approach_env_cfg.py` + `scripts/train.py` |

**关键点：** 奖励与终止读取**干净的几何 BBox**（`BBoxObservation.get_clean_state`），
策略观测读取**加噪/延迟/漏检后的 BBox**。漏检只污染策略观测、不改变"真值可见性"，
因此模型学会在漏检几帧时靠惯性滑行，而不是急刹车。

---

## 2. 环境（已装好，针对 IsaacLab 2.3.2 + Isaac Sim 5.1）

> 已安装并冒烟验证完毕，装在 **`/home/shihao/pistar/.venv`**（Python 3.11）：
> `isaacsim 5.1.0.0` + `isaaclab 0.54.2`(对应 **IsaacLab 仓库 tag v2.3.2**) + `skrl 1.4.3`。
> 该 venv 同时是 openpi/JAX 训练环境，二者已确认可共存（torch 2.7.0、jax GPU 正常）。

若需在新机器复现安装：
```bash
# (1) IsaacLab 切到与 Isaac Sim 5.x / py3.11 配套的 tag
cd /home/shihao/IsaacLab && git checkout v2.3.2

# (2) pip 装 Isaac Sim 5.1（py3.11 轮子）
/home/shihao/pistar/.venv/bin/python -m pip install 'isaacsim[all]==5.1.0.0' \
    --extra-index-url https://pypi.nvidia.com

# (3) 装 IsaacLab 扩展 + skrl（注意把 skrl 钉在 <2，2.3.2 与 skrl 2.x 不兼容）
cd /home/shihao/IsaacLab && source /home/shihao/pistar/.venv/bin/activate \
    && export OMNI_KIT_ACCEPT_EULA=YES && ./isaaclab.sh --install skrl
/home/shihao/pistar/.venv/bin/python -m pip install 'skrl>=1.4.3,<2'
# 若 flatdict 因 uv 构建隔离报 pkg_resources 缺失：
#   /home/shihao/pistar/.venv/bin/python -m pip install --no-build-isolation flatdict==4.0.1
```

> 首次 `import isaacsim` 需接受 EULA：所有命令前加 `OMNI_KIT_ACCEPT_EULA=YES`。
> 本任务包无需 `pip install -e`，`scripts/train.py|play.py` 已自带 sys.path 兜底。

---

## 3. 训练 / 验证 / 回放

> 用 venv 的 python 直接跑（pip 版 isaacsim 由 `AppLauncher` 启动 Kit），**不经 `./isaaclab.sh`**。
> 所有命令前置 `OMNI_KIT_ACCEPT_EULA=YES`。

```bash
cd /home/shihao/isaac_rm
PY=/home/shihao/pistar/.venv/bin/python

# 冒烟测试（已验证通过：32 env、2 迭代，~12s）
OMNI_KIT_ACCEPT_EULA=YES $PY scripts/train.py \
    --task Isaac-Chassis-Approach-v0 --num_envs 32 --headless --max_iterations 2

# 正式训练（4096 env，无头）
OMNI_KIT_ACCEPT_EULA=YES $PY scripts/train.py \
    --task Isaac-Chassis-Approach-v0 --num_envs 4096 --headless

# 回放（带界面，加载 checkpoint）
OMNI_KIT_ACCEPT_EULA=YES $PY scripts/play.py \
    --task Isaac-Chassis-Approach-Play-v0 --num_envs 16 \
    --checkpoint /abs/path/to/logs/skrl/chassis_approach/<run>/checkpoints/best_agent.pt

OMNI_KIT_ACCEPT_EULA=YES $PY scripts/play.py \
    --task Isaac-Chassis-Approach-Play-v0 --num_envs 16 \
    --enable_cameras --checkpoint <你的ckpt路径>

```

日志/Checkpoint 在运行目录下 `logs/skrl/chassis_approach/`；
TensorBoard：`tensorboard --logdir /home/shihao/isaac_rm/logs/skrl/chassis_approach`。

> 注：URDF 导入时双臂 `meshes/rm65/*.STL`、部分 `rmg24/*` 网格路径在该模型里不存在，会刷
> "Failed to resolve mesh" 错误——但底盘/轮子/相机网格都在、能正常导入,机械臂本就被锁定,
> 仅缺可视/碰撞几何,对底盘导航任务无影响。

---4.38s

## 4. 可调参数（集中在 `approach_env_cfg.py` 顶部 `# --- tunables ---`）

| 参数 | 默认 | 含义 |
|---|---|---|
| `CONTROL_HZ` / `PHYSICS_HZ` | 10 / 100 | 控制频率（对齐 10Hz 视觉）/ 物理频率 |
| `V_RANGE` / `W_RANGE` | (-0.2,0.5) / (-1,1) | 动作映射到的物理线/角速度范围 |
| `CAM_HFOV` / `CAM_VFOV` | 1.204 / 0.75 rad | 虚拟相机水平/垂直 FOV |
| `TARGET_SIZE` / `TARGET_HALF` | 0.25×0.25×0.40 | 目标立方体尺寸（half 必须 = size/2） |
| `A_TARGET` | 0.25 | 理想 BBox 面积占比（"可操作距离"） |
| `CENTER_K` | 2.0 | 对准奖励陡峭度 |
| `COLLISION_DIST` | 0.45 m | base↔target 距离小于此判为碰撞 |
| `SUCCESS_DWELL` / `LOST_DWELL` | 10 / 10 步 | 成功/丢失需持续的帧数 |
| `BBOX_NOISE_STD` | (.02,.02,.01,.02) | (cx,cy,area,aspect) 高斯噪声 std |
| `DROPOUT_PROB` | 0.05 | 每帧漏检概率 |
| `LATENCY_STEPS` | (1,2) | 观测延迟步数范围（10Hz 下 ≈100–200ms） |

底盘物理常量在 `robots/chassis.py` 的 `CHASSIS_PARAMS`：`wheel_base=0.296`（来自 URDF），
`wheel_radius=0.075`（**估计值，请按实物/网格核实**），`camera_body="camera_link"`。

---

## 5. 部署对接（sim → real）

观测向量顺序（共 10 维，必须与真实侧一致）：

```
[ cx, cy, area_ratio, w/h,      # 来自 DINO+SAM2 的 BBox（按图像归一化）
  v, ω,                         # 底盘实测线速度/角速度
  a_{t-1}^v, a_{t-1}^ω,         # 上一帧动作
  a_{t-2}^v, a_{t-2}^ω ]        # 上上帧动作
```

- `cx, cy ∈ [-1,1]`：BBox 中心相对图像中心的归一化偏差（0=正中心）。
- `area_ratio ∈ [0,1]`：BBox 面积 / 图像面积。
- 真实侧检测失败（漏检）时把 bbox 四维置 0，与训练一致。

动作输出 `(a_v, a_ω) ∈ [-1,1]`，按同一仿射映射还原为 `(v_cmd, ω_cmd)` 下发底盘
（`v = a_v` 映射到 `V_RANGE`，`ω` 同理），不要再经过轮速换算（轮速换算只是仿真内部用于驱动 PhysX 轮子）。

---

## 6. 风险与调参注记（已冒烟跑通；正式训练前请按此核对/排查）

- **前进/旋转方向**：若机器人原地打转或倒着走，翻转 `DifferentialDriveActionCfg`
  的 `left_sign` / `right_sign`（URDF 左轮 axis 为 -x，已默认 `left_sign=-1`）。
- **wheel_radius**：影响动作→轮速的缩放；因为观测里有实测速度反馈，PPO 能自适应，
  但部署前仍建议量准。
- **万向轮稳定性**：被动 caster 用自由 continuous 关节模拟，PhysX 偶有抖动。
  若不稳：调低 caster 摩擦 / 提高 `PHYSICS_HZ` / 或把 caster 换成低摩擦球。
- **相机光轴朝向**：`BBoxObservation` 默认 forward=+x, right=-y, up=+z（ROS body→image）。
  若投影出的 bbox 方向不对，改 `camera_axes` 参数（ROS 光学帧常为 z-forward）。
- **camera_link 被合并**：本配置 `merge_fixed_joints=False` 保留该 body；若你的导入器仍
  合并了它，把 `CHASSIS_PARAMS["camera_body"]` 改为 `"head_link2"`。
- **终止奖励量级**：reward = func × weight × dt。当前 `SUCCESS_WEIGHT=100/dt`、
  `FAILURE_WEIGHT=-50/dt`，即有效 +100/−50。改控制频率时此换算会自动跟随 `DT`。

---

## 7. 目录结构

```
isaac_rm/
├── README.md
├── require.txt
├── source/chassis_nav/
│   ├── pyproject.toml
│   └── chassis_nav/
│       ├── __init__.py
│       ├── robots/chassis.py            # ArticulationCfg(UrdfFileCfg) + 执行器分组
│       └── tasks/approach/
│           ├── __init__.py              # gym.register
│           ├── approach_env_cfg.py      # 场景 + 各 Manager + PLAY 变体（含 tunables）
│           ├── mdp/
│           │   ├── actions.py           # DifferentialDriveAction
│           │   ├── observations.py      # BBoxObservation（投影+延迟+漏检+噪声）
│           │   ├── rewards.py
│           │   └── terminations.py      # TaskOutcome（success/collision/lost）
│           └── agents/skrl_ppo_cfg.yaml
└── scripts/
    ├── train.py
    └── play.py
```
