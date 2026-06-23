"""把 rl-tune 最优 overrides 写回源配置文件。

用法::

    python harbor/scripts/rl/skrl_local/save_best_overrides.py \\
        --tune-dir harbor/rl_experiments/tunes/tune_20260623-045552 \\
        --task Isaac-RM-Push-Block-v0

脚本从 result.json 读取 best_overrides_path，解析 overrides.yaml，然后：
  - reward.* 覆盖 → push_env_cfg.py（或对应任务的 *_env_cfg.py）
  - agent.* 覆盖 → agents/skrl_ppo_cfg.yaml
  - 其余 harbor 保留键（task/seed/num_envs/total_timesteps/wandb）忽略

支持的 override 格式（train.py 的 CLI 约定）：
  reward.<term>.weight=<float>
  reward.<term>.params.<param>=<val>
  agent.<key>=<val>
  models.<path>=<val>
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[4]   # isaac_rm/
HARBOR_KEYS = {"task", "seed", "total_timesteps", "num_envs", "wandb", "config_name",
               "wandb_run_name"}


# ──────────────────────────────────────────────────────────────────────────────
# CLI parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="写回调优最优参数到源 cfg")
    p.add_argument("--tune-dir", required=True, help="tune 目录，如 harbor/rl_experiments/tunes/tune_YYYYMMDD-HHMMSS")
    p.add_argument("--task", required=True, help="任务 ID，如 Isaac-RM-Push-Block-v0")
    p.add_argument("--dry-run", action="store_true", help="只打印将要修改的内容，不实际写文件")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# 读取 best overrides.yaml
# ──────────────────────────────────────────────────────────────────────────────

def _load_best_overrides(tune_dir: Path, task: str) -> dict:
    """从 tune 目录找到 result.json，读取 best_overrides_path，解析 overrides.yaml。"""
    # 找到对应 task 的 cell dir
    candidates = list(tune_dir.glob(f"*{task}*/result.json"))
    if not candidates:
        # fallback: search all result.json
        candidates = list(tune_dir.glob("*/result.json"))
    if not candidates:
        sys.exit(f"[save_best] 找不到 result.json（在 {tune_dir}）")

    # 优先找任务名匹配的
    result_path = next((c for c in candidates if task in str(c)), candidates[0])
    import json
    with open(result_path) as f:
        result = json.load(f)

    overrides_path = result.get("best_overrides_path")
    if not overrides_path:
        sys.exit("[save_best] result.json 中没有 best_overrides_path 字段")

    overrides_path = Path(overrides_path)
    if not overrides_path.is_absolute():
        overrides_path = REPO / overrides_path

    if not overrides_path.exists():
        sys.exit(f"[save_best] overrides 文件不存在: {overrides_path}")

    with open(overrides_path) as f:
        data = yaml.safe_load(f)

    overrides = data.get("overrides", {}) if isinstance(data, dict) else {}
    # 过滤 harbor 保留 key
    overrides = {k: v for k, v in overrides.items() if k not in HARBOR_KEYS}
    print(f"[save_best] 读取 overrides: {overrides_path}")
    print(f"[save_best] 有效 override 数量: {len(overrides)}")
    return overrides


# ──────────────────────────────────────────────────────────────────────────────
# 找源文件路径
# ──────────────────────────────────────────────────────────────────────────────

def _find_task_dir(task: str) -> Path:
    """从任务 ID 推断任务目录，如 Isaac-RM-Push-Block-v0 → tasks/push/"""
    # 搜索注册了该 task id 的 __init__.py
    for init in (REPO / "source").rglob("__init__.py"):
        try:
            content = init.read_text()
            if f'"{task}"' in content or f"'{task}'" in content:
                return init.parent
        except Exception:
            continue
    sys.exit(f"[save_best] 找不到注册 {task} 的任务目录")


# ──────────────────────────────────────────────────────────────────────────────
# 写回 reward.* → env_cfg.py
# ──────────────────────────────────────────────────────────────────────────────

def _apply_reward_override_to_cfg(cfg_path: Path, term: str, field: str, param: str | None,
                                   val, dry_run: bool) -> bool:
    """在 Python 源文件里找到 `<term>` 这个 RewTerm 并修改指定字段。

    支持：
      weight:            weight=<val>
      params.<param>:    "<param>": <val>  （dict literal）
    """
    src = cfg_path.read_text()

    if field == "weight":
        # 找 weight=<old> 在 term 块内
        # 模式：term = RewTerm(... weight=<old> ...)，跨多行
        pattern = rf'({re.escape(term)}\s*=\s*RewTerm\([^)]*?weight\s*=\s*)([^\s,)]+)'
        new_src = re.sub(pattern, lambda m: f"{m.group(1)}{val}", src, flags=re.DOTALL)
        if new_src == src:
            print(f"  [warn] 未找到 {term}.weight 可替换位置")
            return False
        changed_val = val

    elif field == "params" and param:
        # 找 "<param>": <old> 在 term 块内
        # 先定位 term 块的起始位置，再在其中替换
        term_start = src.find(f"{term} = RewTerm(")
        if term_start == -1:
            term_start = src.find(f"{term}=RewTerm(")
        if term_start == -1:
            print(f"  [warn] 未找到 RewTerm 定义: {term}")
            return False

        # 找到这个 RewTerm 块的结束括号
        depth = 0
        block_end = term_start
        in_block = False
        for i, ch in enumerate(src[term_start:], start=term_start):
            if ch == '(':
                depth += 1
                in_block = True
            elif ch == ')':
                depth -= 1
                if in_block and depth == 0:
                    block_end = i
                    break

        block = src[term_start:block_end + 1]
        # 替换 "<param>": <old>
        pattern = rf'("{re.escape(param)}"\s*:\s*)([^\s,\n}}]+)'
        new_block = re.sub(pattern, lambda m: f'{m.group(1)}{val}', block)
        if new_block == block:
            # 也尝试不带引号的 key
            pattern2 = rf"('{re.escape(param)}'\s*:\s*)([^\s,\n}}]+)"
            new_block = re.sub(pattern2, lambda m: f"{m.group(1)}{val}", block)
        if new_block == block:
            print(f"  [warn] 未找到 params.{param} 可替换位置（在 {term} 块内）")
            return False
        new_src = src[:term_start] + new_block + src[block_end + 1:]
        changed_val = val
    else:
        print(f"  [warn] 不支持的 reward override 格式: {field}")
        return False

    if dry_run:
        print(f"  [dry-run] 将修改 {cfg_path.name}: {term}.{field}" +
              (f".{param}" if param else "") + f" → {changed_val}")
        return True

    cfg_path.write_text(new_src)
    print(f"  [write] {cfg_path.name}: {term}.{field}" +
          (f".{param}" if param else "") + f" = {changed_val}")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 写回 agent.* → skrl_ppo_cfg.yaml
# ──────────────────────────────────────────────────────────────────────────────

def _apply_agent_override_to_yaml(yaml_path: Path, key_path: str, val, dry_run: bool) -> bool:
    """修改 skrl_ppo_cfg.yaml 中的深路径 key（如 agent.learning_rate）。"""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    parts = key_path.split(".")  # e.g. ["agent", "learning_rate"]
    d = data
    try:
        for p in parts[:-1]:
            d = d[p]
        leaf = parts[-1]
        old_val = d.get(leaf, "<missing>")
        d[leaf] = val
    except (KeyError, TypeError) as e:
        print(f"  [warn] 路径 {key_path} 不存在: {e}")
        return False

    if dry_run:
        print(f"  [dry-run] 将修改 {yaml_path.name}: {key_path} {old_val!r} → {val!r}")
        return True

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
    print(f"  [write] {yaml_path.name}: {key_path} = {val!r}")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()
    tune_dir = Path(args.tune_dir)
    if not tune_dir.is_absolute():
        tune_dir = REPO / tune_dir
    if not tune_dir.exists():
        sys.exit(f"[save_best] tune 目录不存在: {tune_dir}")

    overrides = _load_best_overrides(tune_dir, args.task)
    if not overrides:
        print("[save_best] 没有需要写回的 override（最优配置即为默认值）")
        return

    task_dir = _find_task_dir(args.task)
    print(f"[save_best] 任务目录: {task_dir}")

    # 找 env_cfg.py 和 skrl_ppo_cfg.yaml
    cfg_files = list(task_dir.glob("*_env_cfg.py"))
    if not cfg_files:
        sys.exit(f"[save_best] 找不到 *_env_cfg.py 在 {task_dir}")
    env_cfg_path = cfg_files[0]
    yaml_path = task_dir / "agents" / "skrl_ppo_cfg.yaml"
    if not yaml_path.exists():
        sys.exit(f"[save_best] 找不到 {yaml_path}")

    changed = 0
    for key, val in overrides.items():
        if key.startswith("reward."):
            # reward.<term>.<field> 或 reward.<term>.params.<param>
            parts = key.split(".")
            if len(parts) < 3:
                print(f"  [skip] 格式错误: {key}")
                continue
            term, field = parts[1], parts[2]
            param = parts[3] if len(parts) >= 4 else None
            ok = _apply_reward_override_to_cfg(env_cfg_path, term, field, param, val, args.dry_run)
            if ok:
                changed += 1

        elif key.startswith(("agent.", "models.", "memory.", "trainer.")):
            ok = _apply_agent_override_to_yaml(yaml_path, key, val, args.dry_run)
            if ok:
                changed += 1
        else:
            print(f"  [skip] 未知 override 前缀: {key}")

    mode = "（dry-run）" if args.dry_run else ""
    print(f"\n[save_best] 完成{mode}：写回 {changed}/{len(overrides)} 个 override")
    if args.dry_run:
        print("  加 --dry-run=false 或去掉 --dry-run 实际写入")


if __name__ == "__main__":
    main()
