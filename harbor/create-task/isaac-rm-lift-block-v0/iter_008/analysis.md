# iter_008 Analysis

## Numerical

| Term | Value | Weight | Notes |
|------|-------|--------|-------|
| total | 6.961 | — | |
| reaching_block | 7.054 | 2.0 | Dominant — still ~100% of total |
| **lift_with_grip** | **0.0022** | 200.0 | **首次非零** — 阈值修复有效 |
| **pre_grasp** | **0.0010** | 10.0 | **首次非零** — 方块偶尔被闭爪抬起 |
| success | 2e-10 | 1e-6 | success_rate = 0.0 |

## Visual (13 frames, 20s rollout, 3 envs side-by-side)

**相机问题**：渲染使用了宽角全景相机，3 个环境并排极小，方块（4cm）在此分辨率下不可见。

**观察到的行为**：
- f001: 黑帧（相机初始化）
- f002–f013: 3 台机器人姿态几乎相同，跨全部 13 帧变化极小
- 手臂整体处于「向前下方伸展」的 reaching 姿态，指向桌面区域（左侧机器人最明显）
- 无任何帧可见方块被抬起
- inference_moved=True，frame_diff=107.16 → 手臂有运动，但幅度小且在所有 env 中一致

**行为结论**：
策略收敛到「持续靠近方块」的姿态（reaching_block=7.05），偶尔（约 0.33mm 高度增量，<1 步/episode）触发 lift_with_grip。
pre_grasp=0.001 对应约 0.003 步/episode 触发，说明方块极少被抬至 8cm 以上。
策略尚未学会「稳定维持闭爪+保持接触+持续抬升」的连续行为。

## Root cause

lift_with_grip=0.0022 说明奖励可以触发（阈值修复有效），但梯度信号太弱：
- lift_with_grip 平均贡献 ≈ 0.067/step（1cm 高度），等于 reaching_block max
- 在 20M 步探索中，策略还未找到持续维持抬升的轨迹

## iter_009 plan

1. **归一化 lift_with_grip**：`near * closed * tanh(height_gain / 0.05)` → [0,1]，weight=20.0
   - 0.5cm 高度 → 0.10 → 0.067/step（≈reaching）
   - 5cm 高度 → 0.76 → 0.507/step（7.5× reaching）
   - 强烈激励持续抬升
2. **增加训练步数**：20M → 40M，给策略更多探索时间
3. **相机修复**：调近相机至方块-EE 交互区域
