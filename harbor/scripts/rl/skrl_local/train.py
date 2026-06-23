"""chassis_nav — harbor local_implementation 训练入口（薄壳，包住 skrl PPO）。

这是 harbor ``algorithm_source.kind == local_implementation`` 的训练脚本：它复用仓库
现成的 skrl PPO 栈（同一 agent cfg、同一 IsaacLab env），并额外把每项 episodic reward
与成功率写成 harbor 约定的 ``metrics.jsonl``，供 ``/harbor:rl-run`` / ``/harbor:reward-tune``
的分析步消费 —— 不依赖解析 skrl 的 TensorBoard。

harbor 调用约定（reward-tune / rl-run 透传）::

    python harbor/scripts/rl/skrl_local/train.py \
        --config-name=ppo.parallel \
        task=Isaac-RM-Push-Block-v0 seed=42 total_timesteps=2000000 \
        wandb=rm-push-block [num_envs=2048]

深路径 override（PPO 超参 + 奖励权重）::

    # PPO 超参（写入 agent_cfg，任意 skrl agent cfg 字段）
    agent.learning_rate=1e-4
    agent.learning_epochs=8
    agent.mini_batches=8
    agent.entropy_loss_scale=0.01
    agent.ratio_clip=0.2
    agent.grad_norm_clip=0.5
    models.policy.network.0.layers=[512,256,128]

    # 奖励权重 / params（写入 env_cfg.rewards，在 gym.make 前生效）
    reward.reaching_block.weight=8.0
    reward.block_to_goal_tracking.weight=20.0
    reward.block_to_goal_tracking_mid_band.params.std=0.1

任何不在 harbor 保留 key 集合（task/seed/total_timesteps/num_envs/wandb/config_name）
内、且包含 ``.`` 的 CLI token，都会被解析为深路径 override（reward.* 写 env_cfg，
其余写 agent_cfg）。类型自动推断（优先匹配已有值的类型；否则依次尝试 int / float / bool / str）。

产物 ``harbor/outputs/<algo>_<task>_<ts>/``::
    checkpoint.pth     skrl agent 权重
    metrics.jsonl      每项 episodic reward + 成功率（最后一行为最终评估）
    trial_dir.txt 由调用方写

退出前打印 ``[trial_dir] <abs path>`` 供编排方解析。
现在开始训练了吗，所有对话都中文问答
"""
from __future__ import annotations

import argparse
import sys


def _parse_harbor_args(argv: list[str]) -> dict:
    """解析 harbor 透传的混合 CLI：``--config-name=X`` + ``key=value`` token。"""
    out: dict[str, str] = {}
    for tok in argv:
        if tok.startswith("--config-name="):
            out["config_name"] = tok.split("=", 1)[1]
        elif tok.startswith("--"):
            # 形如 --foo=bar / --foo
            kv = tok[2:]
            if "=" in kv:
                k, v = kv.split("=", 1)
                out[k] = v
            else:
                out[kv] = "true"
        elif "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


# harbor 保留 key，不透传给 agent_cfg
_HARBOR_KEYS = {"task", "seed", "total_timesteps", "num_envs", "wandb", "config_name"}


def _cast(raw: str, ref):
    """将字符串 raw 转换为与 ref 相同的类型；ref 为 None 时自动推断。"""
    import ast
    if isinstance(ref, bool):
        return raw.lower() in ("true", "1", "yes")
    if isinstance(ref, int):
        return int(float(raw))
    if isinstance(ref, float):
        return float(raw)
    if isinstance(ref, list):
        return ast.literal_eval(raw)
    # 自动推断
    if raw.lower() in ("true", "false"):
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _apply_agent_overrides(agent_cfg: dict, args: dict) -> None:
    """把 CLI 里的深路径 key（如 agent.learning_rate=1e-4）写入 agent_cfg。"""
    for key, raw_val in args.items():
        if key in _HARBOR_KEYS or "." not in key:
            continue
        if key.startswith("reward."):
            continue  # reward.* 由 _apply_reward_overrides 处理
        parts = key.split(".")
        d = agent_cfg
        try:
            for p in parts[:-1]:
                d = d[p]
            leaf = parts[-1]
            existing = d.get(leaf)
            val = _cast(raw_val, existing)
            d[leaf] = val
            print(f"[skrl_local] override: {key} = {val!r}", flush=True)
        except (KeyError, TypeError) as exc:
            print(f"[skrl_local] WARNING: override '{key}' skipped: {exc}", flush=True)


def _apply_reward_overrides(env_cfg, args: dict) -> None:
    """把 CLI 里的 reward.* key 写入 env_cfg.rewards.<term>.weight / .params.<p>。

    支持的格式：
      reward.<term>.weight=8.0           修改某项奖励的权重
      reward.<term>.params.<p>=0.1       修改某项奖励的 params 字段
    """
    rewards_cfg = getattr(env_cfg, "rewards", None)
    if rewards_cfg is None:
        return
    for key, raw_val in args.items():
        if not key.startswith("reward."):
            continue
        parts = key.split(".")  # ["reward", term, field, ...]
        if len(parts) < 3:
            print(f"[skrl_local] WARNING: reward override '{key}' 格式错误（需要 reward.<term>.<field>），跳过", flush=True)
            continue
        term_name, field = parts[1], parts[2]
        term_cfg = getattr(rewards_cfg, term_name, None)
        if term_cfg is None:
            print(f"[skrl_local] WARNING: reward term '{term_name}' 不存在，跳过", flush=True)
            continue
        if field == "params":
            # reward.<term>.params.<param>=val
            if len(parts) < 4:
                print(f"[skrl_local] WARNING: reward override '{key}' 缺少 param 名称，跳过", flush=True)
                continue
            param_key = parts[3]
            existing = term_cfg.params.get(param_key)
            val = _cast(raw_val, existing)
            term_cfg.params[param_key] = val
        else:
            existing = getattr(term_cfg, field, None)
            val = _cast(raw_val, existing)
            setattr(term_cfg, field, val)
        print(f"[skrl_local] override: {key} = {val!r}", flush=True)


_ARGS = _parse_harbor_args(sys.argv[1:])

# ------------------------------------------------------------------ #
# 1) 启动 Isaac Sim（必须在任何 isaaclab 导入之前）。
# ------------------------------------------------------------------ #
from isaaclab.app import AppLauncher

_app_parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(_app_parser)
_app_args, _ = _app_parser.parse_known_args(["--headless"])
app_launcher = AppLauncher(_app_args)
simulation_app = app_launcher.app

# ------------------------------------------------------------------ #
# 2) 其余导入。
# ------------------------------------------------------------------ #
import json
import os
import random
import time
from datetime import datetime

import torch
import gymnasium as gym

import chassis_nav  # noqa: F401  (registers Isaac-Chassis-* tasks)
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from skrl.utils.runner.torch import Runner

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
HARBOR = os.path.join(REPO, "harbor")


def main() -> int:
    task = _ARGS.get("task", "Isaac-RM-Push-Block-v0")
    seed = int(_ARGS.get("seed", 42))
    total_timesteps = int(float(_ARGS.get("total_timesteps", 2_000_000)))
    num_envs = int(_ARGS.get("num_envs", 2048))
    algo = "ppo"
    device = "cuda:0"

    if seed == -1:
        seed = random.randint(0, 10000)
    torch.manual_seed(seed)

    # --- 加载 env + agent 配置（复用仓库注册的 skrl cfg）---
    env_cfg = parse_env_cfg(task, device=device, num_envs=num_envs)
    env_cfg.seed = seed
    _apply_reward_overrides(env_cfg, _ARGS)
    agent_cfg = load_cfg_from_registry(task, "skrl_cfg_entry_point")
    agent_cfg["seed"] = seed
    _apply_agent_overrides(agent_cfg, _ARGS)

    # total_timesteps（环境交互总数）→ skrl trainer.timesteps（= 每环境 rollout 步数）。
    skrl_timesteps = max(int(total_timesteps // num_envs), int(agent_cfg["agent"]["rollouts"]))
    agent_cfg["trainer"]["timesteps"] = skrl_timesteps
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # W&B：harbor 传 wandb=<project> 时开启。
    wandb_proj = _ARGS.get("wandb")
    agent_cfg["agent"]["experiment"]["wandb"] = bool(wandb_proj)
    if wandb_proj:
        agent_cfg["agent"]["experiment"]["wandb_kwargs"] = {"project": wandb_proj}

    # --- trial 目录 ---
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    trial_dir = os.path.join(HARBOR, "outputs", f"{algo}_{task}_{ts}")
    os.makedirs(trial_dir, exist_ok=True)
    agent_cfg["agent"]["experiment"]["directory"] = trial_dir
    agent_cfg["agent"]["experiment"]["experiment_name"] = "skrl"

    print(f"[skrl_local] task={task} seed={seed} num_envs={num_envs} "
          f"total_timesteps={total_timesteps} -> skrl_timesteps={skrl_timesteps}", flush=True)

    # --- 建 env + skrl runner ---
    env = gym.make(task, cfg=env_cfg, render_mode=None)
    raw_env = env.unwrapped                       # IsaacLab ManagerBasedRLEnv（读 reward_manager）
    # Per-term reward wrapper（harbor:reward-add-log）
    _scripts_dir = os.path.join(REPO, "scripts")
    if _scripts_dir not in sys.path:
        sys.path.insert(0, _scripts_dir)
    try:
        import _isaaclab_env as _rl_env_helper
        env = _rl_env_helper._make_detailed_reward_wrapper()(env, task)
    except ImportError:
        pass
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    runner = Runner(env, agent_cfg)

    t0 = time.time()
    runner.run()
    train_s = time.time() - t0
    print(f"[skrl_local] training done in {train_s:.1f}s", flush=True)

    # --- 保存 checkpoint ---
    ckpt = os.path.join(trial_dir, "checkpoint.pth")
    runner.agent.save(ckpt)
    print(f"[skrl_local] saved checkpoint -> {ckpt}", flush=True)

    # --- 评估并写 metrics.jsonl ---
    metrics = _evaluate(env, raw_env, runner.agent, total_timesteps)
    with open(os.path.join(trial_dir, "metrics.jsonl"), "w") as f:
        f.write(json.dumps(metrics) + "\n")
    print(f"[skrl_local] metrics: success_rate={metrics['eval/success_rate']:.3f} "
          f"total_return={metrics['reward/total/episodic_return_mean']:.3f}", flush=True)

    print(f"[trial_dir] {trial_dir}", flush=True)
    env.close()
    simulation_app.close()
    return 0


@torch.no_grad()
def _evaluate(env, raw_env, agent, step_label: int, target_episodes: int = 256) -> dict:
    """确定性策略评估：收集每项 episodic reward + 成功率，读自 reward_manager._episode_sums。

    在每步**之前**快照 ``_episode_sums``；步进后对当步结束(done)的环境，记录其快照值
    （= 该 episode 的每项加权 reward 累积，误差仅最后一步，<1%）+ 是否成功(``_reach_success``)。
    """
    agent.set_running_mode("eval")
    rm = raw_env.reward_manager
    terms = list(rm.active_terms)

    sums: dict[str, list[float]] = {t: [] for t in terms}
    successes: list[float] = []

    obs, _ = env.reset()
    max_steps = int(raw_env.max_episode_length) + 5
    n_steps = 0
    while len(successes) < target_episodes and n_steps < max_steps * 3:
        prev = {t: rm._episode_sums[t].detach().clone() for t in terms}
        actions = agent.act(obs, n_steps, max_steps)[0]
        obs, _, terminated, truncated, _ = env.step(actions)
        done = (terminated | truncated).flatten().bool()
        succ = getattr(raw_env, "_reach_success", None)
        if done.any():
            ids = done.nonzero(as_tuple=False).flatten()
            for t in terms:
                sums[t].extend(prev[t][ids].cpu().tolist())
            if succ is not None:
                successes.extend(succ.flatten()[ids].float().cpu().tolist())
            else:
                successes.extend([0.0] * len(ids))
        n_steps += 1

    def _mean(xs):
        return float(sum(xs) / len(xs)) if xs else 0.0

    per_term = {f"reward/{t}/episodic_return_mean": _mean(sums[t]) for t in terms}
    total = sum(per_term.values())
    out = {
        "step": int(step_label),
        "reward/total/episodic_return_mean": total,
        **per_term,
        "eval/success_rate": _mean(successes),
        "eval/episodes": len(successes),
    }
    return out


if __name__ == "__main__":
    sys.exit(main())
