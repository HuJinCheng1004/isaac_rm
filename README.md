# isaac_rm — RM Push-Block RL 训练流水线

Isaac Lab 2.3.2 + Isaac Sim 5.1 + skrl PPO，机械臂将方块推至目标位置。

**最高成功率：94.2%**（`ppo_Isaac-RM-Push-Block-v0_20260623-094029`，40M 步，4096 并行环境）

---

## 硬件要求

- NVIDIA GPU（建议 RTX 3090 及以上，显存 ≥ 16GB）
- CUDA 12.x
- Ubuntu 22.04 / 24.04
- RAM ≥ 32GB

---

## 1. 环境安装

```bash
git clone https://github.com/HuJinCheng1004/isaac_rm.git
cd isaac_rm

# 一键安装（约 10~20 分钟，Isaac Sim 约 8GB 下载）
bash setup_env.sh
```

安装完成后激活环境：

```bash
conda activate isaac_rm
```

安装本项目包：

```bash
pip install -e ./source/chassis_nav
```

> 国内网络建议挂代理后再执行，NVIDIA PyPI 源在国内访问较慢。

---

## 2. 训练（复现 94.2% 成功率）

所有关键超参（`std_gate=0.10`、`learning_rate=1.5e-4`）已写入源码，**直接运行即可**：

```bash
cd ~/isaac_rm

OMNI_KIT_ACCEPT_EULA=YES python harbor/scripts/rl/skrl_local/train.py \
    --config-name=ppo.parallel \
    task=Isaac-RM-Push-Block-v0 \
    seed=42 \
    total_timesteps=40000000
```

训练产物输出到 `harbor/outputs/ppo_Isaac-RM-Push-Block-v0_<时间戳>/`：

| 文件 | 说明 |
|---|---|
| `checkpoint.pth` | 最终权重（harbor 格式） |
| `metrics.jsonl` | 每步 reward 分项 + success_rate |
| `skrl/checkpoints/best_agent.pt` | skrl 最佳 checkpoint |

训练时间约 15~30 分钟（取决于 GPU）。

---

## 3. 查看结果

```bash
# 查看最新一次训练的成功率
tail -1 harbor/outputs/ppo_Isaac-RM-Push-Block-v0_*/metrics.jsonl | python3 -c "
import json, sys
d = json.loads(sys.stdin.read())
print(f'success_rate: {d[\"eval/success_rate\"]:.3f}')
"
```

---

## 4. 可视化回放

```bash
OMNI_KIT_ACCEPT_EULA=YES python scripts/play.py \
    --task Isaac-RM-Push-Block-Play-v0 \
    --num_envs 16 \
    --checkpoint harbor/outputs/ppo_Isaac-RM-Push-Block-v0_<时间戳>/skrl/checkpoints/best_agent.pt \
    --enable_cameras --real-time
```

---

## 5. 关键超参说明

| 参数 | 值 | 位置 |
|---|---|---|
| `std_gate` (contact gate) | 0.10 | `tasks/push/push_env_cfg.py` |
| `learning_rate` | 1.5e-4 | `tasks/push/agents/skrl_ppo_cfg.yaml` |
| `num_envs` | 4096 | `harbor/configs/rl/ppo.parallel.yaml` |
| `total_timesteps` | 40000000 | 训练命令行传入 |

**核心 trick**：将 contact gate 的 `std_gate` 从默认 0.05 加宽到 0.10，避免接触门控过紧导致 mid/fine 精度奖励项被饿死（swat 失效根本原因）。此项修改使成功率从 33% 提升至 76%，再配合 40M 步充分训练达到 94%。

---

## 6. 目录结构

```
isaac_rm/
├── setup_env.sh                          # 一键安装脚本
├── source/chassis_nav/                   # 任务源码（pip install -e 安装）
│   └── chassis_nav/tasks/push/
│       ├── push_env_cfg.py               # 场景 + 奖励配置（含 std_gate=0.10）
│       ├── mdp/rewards.py                # 奖励函数实现
│       ├── mdp/observations.py
│       └── agents/skrl_ppo_cfg.yaml      # PPO 超参（lr=1.5e-4）
├── assets/                               # URDF + mesh 文件
├── scripts/play.py                       # 可视化回放
└── harbor/
    ├── scripts/rl/skrl_local/train.py    # 训练入口
    ├── configs/rl/ppo.parallel.yaml      # harbor 训练配置
    └── outputs/                          # 训练产物（不上传 git）
```

---

## 7. 依赖版本

| 包 | 版本 |
|---|---|
| Isaac Sim | 5.1.0 |
| IsaacLab | v2.3.2 |
| skrl | ≥1.4.3, <2 |
| Python | 3.11 |
| PyTorch | 2.7.0 |
