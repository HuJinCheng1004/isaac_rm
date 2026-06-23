# iter_009 Analysis

## Numerical

| Term | Episodic | Weight | Notes |
|------|----------|--------|-------|
| total | 6.742 | — | |
| reaching_block | 6.864 | 2.0 | ≈97% of theoretical max (~7.1) — near-saturated |
| **lift_with_grip** | **0.0054** | 20.0 | ≈2.5× iter_008 (0.0022); tanh归一化有效，但仍极小 |
| **pre_grasp** | **0.0039** | 10.0 | cube z>0.08 ∩ near ∩ closed — 极低频 |
| success | ~0 | 1e-6 | success_rate=0.0 |
| joint_vel | -0.039 | -5e-4 | 正常 |
| joint_jerk | -0.082 | -5e-4 | 正常 |
| table_avoidance | -0.010 | -0.2 | 正常 |

## Visual (13 frames, 20s rollout, 单环境近景相机)

**相机修复成功**：eye=(0.8,0.5,0.6) lookat=(-0.05,-0.15,0.12)，方块-夹爪交互区清晰可见。

### 逐帧行为

| 帧段 | 观察 |
|------|------|
| f001 | 黑帧（相机初始化） |
| f002–f005 | 夹爪从左上方向方块（黄色，桌面上）靠近，臂处于"伸展+下压"姿态 |
| f006–f009 | 夹爪抵近方块，方块仍静止在桌面；手指略张开，未包裹方块 |
| **f010–f013** | **方块向右横向位移（帧间明显偏移）；夹爪从左侧推压方块向右滑动** |

### 核心行为结论

策略执行的是 **侧向推压（side-push）**：
- 夹爪从方块左侧接触，横向推动方块；方块在桌面上向右滑动
- 全程方块未离开桌面（无任何抬升）
- 夹爪未形成包裹：手指触碰一个面，未闭合成抓取构型
- frame_diff=145.90（iter_008 的 107.16），运动量更大，但方向是横向而非垂直

## Root Cause

**奖励景观局部最优**：策略收敛到"接近+侧推"：

1. `reaching_block`（weight=2.0）已近饱和（6.86/max~7.1=97%）→ EE 持续贴近方块 ✓
2. `lift_with_grip` 需要 near × closed × tanh(height_gain) — 三条件同时满足：
   - near ✓（EE 已很近）
   - gripper_closed：夹爪未紧闭（side-push 时手指接触方块表面但未对称包裹）
   - height_gain > 0：方块从未离桌 → lift_with_grip≈0
3. 策略没有获得"主动关闭夹爪"的梯度信号（lift_with_grip 要求先抬升才有梯度）
4. 侧推方向导致方块水平位移而非垂直位移 → lift_with_grip 始终不触发

**关键缺口**：缺少独立的夹爪闭合梯度信号，以及对侧向推压的显式惩罚。

## iter_010 Plan

### 问题
- 策略学到"侧推"作为稳定策略：reaching+push 等于达到接近奖励最大值
- 夹爪没有闭合动机（lift_with_grip 需要先抬升，但抬升需要先夹住）
- 缺少"不要推动方块"约束

### 修复

1. **加回 `gripper_near_cube_shaping`（weight=0.5）**
   - 提供独立的夹爪闭合梯度：near × (1 - tanh(finger_pos/0.02))
   - 不依赖抬升，策略可以先学会闭爪
   - hover悬停上限：0.5 × 200步 × (1/30) ≈ 3.3/episode << reaching_block 13.3/episode，不会成为主要策略

2. **新增 `cube_lateral_velocity_penalty`（weight=-2.0）**
   - 惩罚方块横向（x-y）速度：-2.0 × ‖v_xy‖
   - 直接封堵观察到的侧推行为：推动方块 → 负奖励
   - 策略被迫选择：不动方块（hover） 或 竖直抬升（grasp-and-lift）
   - 与 lift_with_grip 配合：封堵侧推后，垂直方向是唯一正梯度方向

3. **移除 `pre_grasp`（weight=10.0 → 不启用）**
   - pre_grasp=0.0039/episode → 实际梯度贡献极小（0.039/episode，占total<1%）
   - 删除减少奖励项数量，聚焦信号

4. **训练步数**：40M（与iter_009相同）
