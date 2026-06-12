# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""``chassis_nav`` — 一个无渲染的、基于边界框特征的底盘靠近/对齐 RL 任务包。

导入此包会触发 Gym 环境的注册（见
:mod:`chassis_nav.tasks`）。设计为由 Isaac Lab 的 RL
训练脚本简单地通过在 ``pip install -e .`` 后
在 ``sys.path`` 上可导入来发现。
"""

# 导入任务子包作为副作用注册 Gym 环境。
from . import tasks  # noqa: F401
