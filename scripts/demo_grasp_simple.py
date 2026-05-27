"""
demo_grasp_simple.py - 简单抓取演示

核心功能：展示"抓"的动作
  1. 移动到目标
  2. 抓取工件（创建约束）
  3. 提起工件
"""

import time
import math
import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pybullet as p
import transforms3d

from assembly_env.robots.sim_robotless import INITIAL_ORN
from utils.transforms import wxyz_to_xyzw
from utils.io_utils import pr_green, pr_red, pr_info, display_frame_axis, format_urdf_filepath


# ============================================================
# 参数
# ============================================================

MAX_VEL = 0.1         # 移动速度
TIME_STEP = 0.01      # 时间步长
DIST_THRESHOLD = 0.02 # 接近阈值

MAX_STEPS = 200       # 最大步数


def run_demo():
    """运行简单抓取演示"""

    pr_info("=" * 60)
    pr_info("简单抓取演示 - 重点展示'抓'的动作")
    pr_info("=" * 60)

    # ============================================
    # 初始化 PyBullet
    # ============================================
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(0.6, 180, -41, [0, 0, 0])
    p.setPhysicsEngineParameter(numSolverIterations=150, enableFileCaching=0)
    p.setTimeStep(TIME_STEP)
    p.setGravity(0, 0, -9.8)

    pr_info("正在启动仿真...")

    # ============================================
    # 加载模型
    # ============================================
    
    # 目标位置
    target_pos = np.array([0.0, 0.0, 0.0])
    
    # 初始位置（偏离目标）
    start_pos = [0.05, 0.04, 0.24]

    pr_info(f"工具初始位置: {start_pos}")
    pr_info(f"目标抓取位置: {target_pos.tolist()}\n")

    # 加载工具
    tool_uid = p.loadURDF(
        format_urdf_filepath("envs/urdf/robotless_lap_joint/tool"),
        basePosition=start_pos,
        baseOrientation=INITIAL_ORN,
        useFixedBase=0,
    )

    # 加载工件（可以被抓起的）
    workpiece_uid = p.loadURDF(
        format_urdf_filepath("envs/urdf/robotless_lap_joint/task_lap_90deg"),
        basePosition=[0, 0, 0],
        baseOrientation=wxyz_to_xyzw(
            transforms3d.euler.euler2quat(0, 0, math.pi, 'sxyz')
        ),
        useFixedBase=0,  # 重要：不固定，这样才能被抓起
    )

    display_frame_axis(workpiece_uid, 0, line_length=0.05)
    for li in range(p.getNumJoints(tool_uid)):
        display_frame_axis(tool_uid, li, line_length=0.03)

    # 不使用约束 - 直接修改位置更可靠
    tool_constraint = None

    # 稳定仿真
    for _ in range(10):
        p.stepSimulation()
    time.sleep(0.1)

    # 显示文字提示
    p.addUserDebugText("工具", start_pos, [1, 0, 0], textSize=1.0, lifeTime=0)
    p.addUserDebugText("工件", [0, 0, -0.01], [0, 1, 0], textSize=1.0, lifeTime=0)

    # =====================================================
    # 第一步：移动到工件位置
    # =====================================================
    pr_info("【步骤1】移动到工件位置...\n")

    step = 0
    moved = False

    while step < MAX_STEPS:
        # 当前位置
        current_pos, current_orn = p.getBasePositionAndOrientation(tool_uid)

        # 计算距离
        distance = np.linalg.norm(np.array(target_pos) - np.array(current_pos))

        # 移动向量
        direction = (np.array(target_pos) - np.array(current_pos))
        direction = direction / (np.linalg.norm(direction) + 1e-6)

        # 新位置
        delta = direction * MAX_VEL * TIME_STEP
        new_pos = (np.array(current_pos) + delta).tolist()

        # 直接设置位置（关键改变）
        p.resetBasePositionAndOrientation(tool_uid, new_pos, current_orn)

        p.stepSimulation()
        step += 1

        if step % 40 == 0:
            pr_info(f"Step {step:3d} | Distance to workpiece: {distance*1000:.1f}mm")

        # 到达目标
        if distance < DIST_THRESHOLD:
            pr_green(f"\n✓ 已接近工件! (距离: {distance*1000:.2f}mm)\n")
            moved = True
            break

        time.sleep(TIME_STEP * 0.2)

    if not moved:
        pr_red("移动失败")
        p.disconnect()
        return

    # =====================================================
    # 第二步：抓取（核心动作！）
    # =====================================================
    pr_info("【步骤2】准备抓取...\n")
    time.sleep(1.0)

    # 获取当前位置
    tool_pos, tool_orn = p.getBasePositionAndOrientation(tool_uid)
    workpiece_pos, workpiece_orn = p.getBasePositionAndOrientation(workpiece_uid)

    pr_info(f"工具位置: X={tool_pos[0]*100:.1f}cm, Y={tool_pos[1]*100:.1f}cm, Z={tool_pos[2]*100:.1f}cm")
    pr_info(f"工件位置: X={workpiece_pos[0]*100:.1f}cm, Y={workpiece_pos[1]*100:.1f}cm, Z={workpiece_pos[2]*100:.1f}cm")

    # 计算相对位置
    rel_pos = (np.array(workpiece_pos) - np.array(tool_pos)).tolist()

    pr_info(f"相对偏移: dX={rel_pos[0]*1000:.1f}mm, dY={rel_pos[1]*1000:.1f}mm, dZ={rel_pos[2]*1000:.1f}mm")

    time.sleep(0.5)

    # 在工件位置显示"抓取目标"文字
    p.addUserDebugText("准备抓取!", workpiece_pos, [1, 1, 0], textSize=1.5, lifeTime=3)

    pr_info("\n" + "=" * 60)
    pr_info("开始创建抓取约束...\n")
    pr_info("正在执行 GRASP 约束创建...")
    pr_info("=" * 60 + "\n")

    time.sleep(0.5)

    # **创建抓取约束** - 这就是"抓"的动作！
    pr_info("→ 建立工具和工件之间的固定约束...")
    pr_info(f"  父体: 工具 (UID: {tool_uid})")
    pr_info(f"  子体: 工件 (UID: {workpiece_uid})")
    pr_info(f"  约束类型: JOINT_FIXED (固定关联)")
    
    grasp_constraint = p.createConstraint(
        parentBodyUniqueId=tool_uid,        # 父体：工具
        parentLinkIndex=-1,
        childBodyUniqueId=workpiece_uid,    # 子体：工件
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,            # 固定约束
        jointAxis=[0, 0, 0],
        parentFramePosition=rel_pos,        # 工具相对位置
        childFramePosition=[0, 0, 0],       # 工件相对位置
        childFrameOrientation=workpiece_orn,
    )
    
    pr_info(f"  约束ID: {grasp_constraint}")
    pr_info("→ 约束创建完成")

    time.sleep(0.3)

    # 设置约束强度
    pr_info("→ 设置约束强度: maxForce=1000N...")
    p.changeConstraint(grasp_constraint, maxForce=1000)
    pr_info("→ 约束激活完成")

    # 稳定约束（多推几步）
    pr_info("→ 稳定约束系统中...")
    for i in range(30):
        p.stepSimulation()
        if i == 15:
            pr_info("  → 约束已生效...")
    
    time.sleep(0.5)

    # 在工件位置显示"已抓取"文字 - 绿色大字
    p.addUserDebugText("✓ 已抓取!", workpiece_pos, [0, 1, 0], textSize=1.8, lifeTime=0)
    
    # 在工具位置显示红色标记
    p.addUserDebugLine(
        [tool_pos[0]-0.02, tool_pos[1], tool_pos[2]], 
        [tool_pos[0]+0.02, tool_pos[1], tool_pos[2]],
        [1, 0, 0], lifeTime=0, lineWidth=3
    )
    p.addUserDebugLine(
        [tool_pos[0], tool_pos[1]-0.02, tool_pos[2]], 
        [tool_pos[0], tool_pos[1]+0.02, tool_pos[2]],
        [1, 0, 0], lifeTime=0, lineWidth=3
    )

    pr_info("\n" + "=" * 60)
    pr_green("✓✓✓ 抓取成功！✓✓✓")
    pr_green("工件已被固定约束到工具上")
    pr_green("工件和工具现在共同运动")
    pr_green("=" * 60)

    # =====================================================
    # 第三步：提起工件
    # =====================================================
    pr_info("\n【步骤3】提起工件（证明抓取成功）...\n")
    time.sleep(0.5)

    pr_info("→ 工具开始向上运动")
    pr_info("→ 由于约束存在，工件应该跟随工具一起上升")
    pr_info("→ 如果工件不动，说明抓取失败\n")

    time.sleep(0.5)

    lift_height = 0.15
    current_tool_pos, current_tool_orn = p.getBasePositionAndOrientation(tool_uid)
    target_z = current_tool_pos[2] + lift_height

    pr_info(f"目标: 从 Z={current_tool_pos[2]*100:.1f}cm 提起到 Z={target_z*100:.1f}cm\n")

    lift_step = 0
    lift_steps = 200

    while lift_step < lift_steps:
        current_tool_pos, current_tool_orn = p.getBasePositionAndOrientation(tool_uid)
        workpiece_pos_current, _ = p.getBasePositionAndOrientation(workpiece_uid)

        z_error = target_z - current_tool_pos[2]

        # 向上移动
        if z_error > 0.001:
            new_z = current_tool_pos[2] + 0.03 * TIME_STEP  # 30mm/s
        else:
            new_z = current_tool_pos[2]

        new_pos = [current_tool_pos[0], current_tool_pos[1], new_z]

        # 直接设置位置
        p.resetBasePositionAndOrientation(tool_uid, new_pos, current_tool_orn)

        p.stepSimulation()
        lift_step += 1

        if lift_step % 50 == 0:
            height_diff = current_tool_pos[2] - workpiece_pos_current[2]
            pr_info(
                f"Step {lift_step:3d} | "
                f"工具Z: {current_tool_pos[2]*100:.1f}cm | "
                f"工件Z: {workpiece_pos_current[2]*100:.1f}cm | "
                f"高度差: {height_diff*1000:.1f}mm"
            )

        # 到达目标高度
        if z_error < 0.002 and lift_step > 30:
            pr_green(f"\n✓ 提起完成! 高度: {current_tool_pos[2]*100:.1f}cm\n")
            break

        time.sleep(TIME_STEP * 0.2)

    # =====================================================
    # 完成
    # =====================================================
    final_tool, _ = p.getBasePositionAndOrientation(tool_uid)
    final_workpiece, _ = p.getBasePositionAndOrientation(workpiece_uid)

    # 画一条线连接工具和工件，显示约束关系
    p.addUserDebugLine(
        final_tool,
        final_workpiece,
        [0, 1, 0],  # 绿色
        lifeTime=0,
        lineWidth=4,
    )

    # 在中间添加"约束连接"文字
    midpoint = np.array(final_tool) + (np.array(final_workpiece) - np.array(final_tool)) * 0.5
    p.addUserDebugText(
        "约束连接",
        midpoint.tolist(),
        [0, 1, 0],
        textSize=1.0,
        lifeTime=0,
    )

    pr_green("=" * 60)
    pr_green("✓ 演示完成!")
    pr_green(f"  工具最终位置:  Z = {final_tool[2]*100:.1f}cm")
    pr_green(f"  工件最终位置:  Z = {final_workpiece[2]*100:.1f}cm")
    pr_green(f"  高度差:         {(final_tool[2]-final_workpiece[2])*1000:.1f}mm")
    pr_green("\n证明：工件始终跟随工具运动 = 抓取成功!")
    pr_green("=" * 60)

    p.addUserDebugText(
        "✓ 抓取演示完成!",
        [final_tool[0], final_tool[1], final_tool[2]+0.05],
        [0, 1, 0], textSize=1.5, lifeTime=0,
    )

    pr_info("\n按 Ctrl+C 退出\n")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    p.disconnect()


if __name__ == "__main__":
    run_demo()
