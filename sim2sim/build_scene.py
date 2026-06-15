#!/usr/bin/env python3
# Copyright (c) 2026, Chassis-Nav contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""从原始导出的 MJCF 生成一个可仿真的 sim-to-sim 场景（不修改原文件）。

原始文件 ``assets/.../overseas_65_b_v_description_rmg24.urdf`` 旁边的
``..._rmg24.xml`` 只是从 URDF 导出的*运动学骨架*：整机直接挂在 ``worldbody``
下（底盘固定在世界系），且没有 ``<actuator>`` / freejoint。本脚本把它转换成
一个独立、可驱动的场景 ``sim2sim/chassis_scene.xml``：

1. 把底座几何 + 所有顶层 body 包进一个带 ``<freejoint>`` 的浮动底盘 body
   （``base_link_underpan``），并补回 URDF 里的底座惯量（underpan + 焊接的
   ``body_base_link``，各自保留质量/惯量）。
2. 加 ``<actuator>``：两个驱动轮用 velocity 控制；``platform_joint`` 升降杆用
   position 控制；其余上身关节用 position 锁在 0（复现 Isaac 的刚性上身）。脚轮
   不加执行器 = 被动自由滚。
3. 加地面 plane、灯光、目标方块（mocap，便于每回合瞬移到视野内）、跟踪相机。
4. 设碰撞掩码：机器人几何只和地面碰、互不自碰（对齐 Isaac self_collision=False）。
5. ``meshdir`` 写成绝对路径，使场景文件与位置无关。

用法::

    conda run -n realman python sim2sim/build_scene.py
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SRC_XML = os.path.join(
    REPO, "assets", "overseas_65_b_v_description", "urdf",
    "overseas_65_b_v_description_rmg24.xml",
)
MESH_DIR = os.path.join(REPO, "assets", "overseas_65_b_v_description", "meshes")
OUT_XML = os.path.join(HERE, "chassis_scene.xml")

# 底座两个刚性段的惯量（来自 URDF，单位 SI）。fullinertia 顺序：
# ixx iyy izz ixy ixz iyz。
UNDERPAN_INERTIAL = dict(
    pos="-0.012628 -0.016092 -0.060646", mass="2.5521",
    fullinertia="0.066197 0.034903 0.097938 7.0297e-05 0.00017628 -0.00059595",
)
BODY_BASE_INERTIAL = dict(
    pos="-0.0002128 -0.02749 0.37118", mass="12.159",
    fullinertia="0.69175 0.68839 0.15316 3.0303e-06 -1.7947e-05 0.012363",
)

# 升降杆动作起步高度（落在 reset 的 lift_range=(0.2,0.5) 内）。
BASE_SPAWN_Z = 0.2

# 上身需要锁在 0 的关节（position 执行器保持刚性，复现 Isaac upper_body）。
HOLD_JOINTS = [
    "head_joint1", "head_joint2",
    *[f"l_joint{i}" for i in range(1, 8)],
    *[f"r_joint{i}" for i in range(1, 8)],
    "l_Joint_finger1", "l_Joint_finger2",
    "r_Joint_finger1", "r_Joint_finger2",
]


def _find_geom(parent: ET.Element, mesh_name: str) -> ET.Element:
    for g in parent.findall("geom"):
        if g.get("mesh") == mesh_name:
            return g
    raise RuntimeError(f"geom mesh={mesh_name} not found")


def main() -> None:
    tree = ET.parse(SRC_XML)
    root = tree.getroot()

    # --- compiler：绝对 meshdir，使场景文件位置无关 ---
    comp = root.find("compiler")
    comp.set("meshdir", MESH_DIR)
    comp.set("autolimits", "true")

    # --- option：100 Hz 物理，稳定的隐式积分器 ---
    if root.find("option") is None:
        opt = ET.Element("option")
        opt.set("timestep", "0.01")
        opt.set("integrator", "implicitfast")
        root.insert(list(root).index(comp) + 1, opt)

    # --- 碰撞掩码：global geom default = robot(contype1, conaffinity2) ---
    #     floor = contype2/conaffinity1 -> 机器人只和地面碰、互不自碰。
    default = root.find("default")
    gdef = default.find("geom")
    if gdef is None:
        gdef = ET.SubElement(default, "geom")
    gdef.set("contype", "1")
    gdef.set("conaffinity", "2")
    # 安全代理盒（collision 子类）设为纯视觉，避免无关接触。
    for d in default.findall("default"):
        if d.get("class") == "collision":
            cg = d.find("geom")
            cg.set("contype", "0")
            cg.set("conaffinity", "0")

    worldbody = root.find("worldbody")

    # --- 抽出底座几何与所有顶层 body ---
    underpan_geom = _find_geom(worldbody, "base_link_underpan")
    bodybase_geom = _find_geom(worldbody, "body_base_link")
    top_bodies = list(worldbody.findall("body"))
    for el in (underpan_geom, bodybase_geom, *top_bodies):
        worldbody.remove(el)

    # --- 构造浮动底盘 body ---
    base = ET.SubElement(worldbody, "body", name="base_link_underpan",
                         pos=f"0 0 {BASE_SPAWN_Z}")
    ET.SubElement(base, "freejoint", name="base_free")
    ET.SubElement(base, "inertial", **UNDERPAN_INERTIAL)
    base.append(underpan_geom)
    # body_base_link 在 URDF 里固定连到底座（origin 0）-> 焊接子 body（无 joint）。
    bb = ET.SubElement(base, "body", name="body_base_link")
    ET.SubElement(bb, "inertial", **BODY_BASE_INERTIAL)
    bb.append(bodybase_geom)
    # 其余顶层 body（轮子/脚轮/升降平台）位姿本就相对底座原点，直接挂回。
    for b in top_bodies:
        base.append(b)

    # --- 相机光学系参考点：camera_link（head_link2 + 固定偏移，恒等旋转）---
    head2_el = next(b for b in base.iter("body") if b.get("name") == "head_link2")
    ET.SubElement(head2_el, "site", name="camera_link",
                  pos="-0.0032391 -0.051866 0.061606", quat="1 0 0 0",
                  size="0.012", rgba="0 1 0 1")

    # --- 世界：灯光 / 地面 / 目标(mocap) / 跟踪相机 ---
    ET.SubElement(worldbody, "light", name="top", pos="0 0 4", dir="0 0 -1",
                  directional="true")
    ET.SubElement(worldbody, "geom", name="floor", type="plane",
                  size="50 50 0.05", contype="2", conaffinity="1",
                  rgba="0.3 0.3 0.35 1")
    tgt = ET.SubElement(worldbody, "body", name="target", mocap="true",
                        pos="2 0 1")
    # 目标完整尺寸 (0.25,0.25,0.40) -> 半边长；contype0 = 仅可视，碰撞按解析判定。
    ET.SubElement(tgt, "geom", name="target_geom", type="box",
                  size="0.125 0.125 0.20", contype="0", conaffinity="0",
                  rgba="0.8 0.1 0.1 1")
    ET.SubElement(worldbody, "camera", name="track", mode="trackcom",
                  pos="-2.5 -2.5 2.0", xyaxes="0.707 -0.707 0 0.35 0.35 0.87")

    # --- 执行器 ---
    act = ET.SubElement(root, "actuator")
    # 驱动轮：velocity（kv 对齐 Isaac damping=50，力矩上限 20 N·m）。
    for j in ("joint_left_wheel", "joint_right_wheel"):
        ET.SubElement(act, "velocity", name=f"{j}_v", joint=j, kv="50",
                      ctrlrange="-60 60", forcerange="-20 20")
    # 升降杆：position（kp/kv 对齐 Isaac upper_body stiffness/damping）。
    ET.SubElement(act, "position", name="platform_joint_p", joint="platform_joint",
                  kp="2000", kv="100", ctrlrange="0 1", forcerange="-1000 1000")
    # 其余上身关节锁在 0（刚性上身）。
    for j in HOLD_JOINTS:
        ET.SubElement(act, "position", name=f"{j}_p", joint=j, kp="2000",
                      kv="100", forcerange="-1000 1000")

    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(OUT_XML, encoding="unicode", xml_declaration=False)
    print(f"[build_scene] wrote {OUT_XML}")
    print(f"[build_scene] meshdir = {MESH_DIR}")


if __name__ == "__main__":
    main()
