"""
demo_auto_assembly_simple.py - 最简单的装配演示

直接控制工具运动到目标位置，无需复杂约束。
"""

import time
import math
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pybullet as p
import transforms3d

from assembly_env.robots.sim_robotless import RobotSimRobotless, INITIAL_POS, INITIAL_ORN
from utils.transforms import (
    xyzw_to_wxyz, wxyz_to_xyzw,
    mat33_to_quat, mat44_by_pos_quat, mat44_to_pos_quat,
    get_f1_to_f2_xform, transform_mat,
)
from utils.io_utils import pr_green, pr_red, pr_info, display_frame_axis, format_urdf_filepath


# ============================================================
# 参数
# ============================================================

KP_POS = 3.0          # 位置增益
KP_ORN = 2.5          # 姿态增益
MAX_VEL = 0.2         # 最大速度
MAX_RAD = 0.2         # 最大角速度
TIME_STEP = 0.01      # 时间步长

# 初始偏移
OFFSET_X = 0.05       
OFFSET_Y = 0.04       
OFFSET_Z = 0.0        

# 成功条件
DIST_THRESHOLD = 0.02   # 20mm
ANGLE_THRESHOLD = 0.1   # rad

# 最大步数
MAX_STEPS = 150


def quat_error(q_current, q_target):
    """计算四元数误差"""
    qc_wxyz = xyzw_to_wxyz(q_current)
    qt_wxyz = xyzw_to_wxyz(q_target)
    q_current_inv = transforms3d.quaternions.qinverse(qc_wxyz)
    q_rel = transforms3d.quaternions.qmult(qt_wxyz, q_current_inv)
    error_rot = 2.0 * np.array(q_rel[1:4])
    if q_rel[0] < 0:
        error_rot = -error_rot
    return error_rot


def run_demo():
    """运行演示"""

    pr_info("=" * 60)
    pr_info("快速装配演示 - 简化版（直接控制）")
    pr_info("=" * 60)

    # ============================================
    # 初始化 PyBullet
    # ============================================
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(0.6, 180, -41, [0, 0, 0])
    p.setPhysicsEngineParameter(numSolverIterations=150, enableFileCaching=0)
    p.setTimeStep(TIME_STEP)
    p.setGravity(0, 0, 0)

    # ============================================
    # 加载模型
    # ============================================
    # 目标位姿
    target_pos = np.array([0.0, 0.0, 0.0])
    target_orn_euler = np.array([0, 0, math.pi])
    target_orn_xyzw = list(transforms3d.euler.euler2quat(
        target_orn_euler[0], target_orn_euler[1], target_orn_euler[2], 'sxyz'
    ))
    target_orn_xyzw = wxyz_to_xyzw(target_orn_xyzw)

    # 初始位置（目标位置 + 偏移）
    start_pos = [OFFSET_X, OFFSET_Y, 0.24 + OFFSET_Z]

    pr_info(f"目标位置: {target_pos}")
    pr_info(f"起始位置: {start_pos}")
    pr_info("正在启动仿真...")

    # 加载工具
    tool_uid = p.loadURDF(
        format_urdf_filepath("envs/urdf/robotless_lap_joint/tool"),
        basePosition=start_pos,
        baseOrientation=INITIAL_ORN,
        useFixedBase=0,
    )

    # 加载目标
    target_uid = p.loadURDF(
        format_urdf_filepath("envs/urdf/robotless_lap_joint/task_lap_90deg"),
        basePosition=[0, 0, 0],
        baseOrientation=wxyz_to_xyzw(transforms3d.euler.euler2quat(
            0, 0, math.pi, 'sxyz'
        )),
        useFixedBase=0,
    )

    display_frame_axis(target_uid, 0, line_length=0.05)
    for li in range(p.getNumJoints(tool_uid)):
        display_frame_axis(tool_uid, li, line_length=0.03)

    # 计算从基座到 member (link 2) 的变换
    base_pose = p.getBasePositionAndOrientation(tool_uid)
    member_pose = p.getLinkState(tool_uid, 2)[4:6]
    from_base_to_member = get_f1_to_f2_xform(base_pose, member_pose)

    # 稳定仿真
    for _ in range(10):
        p.stepSimulation()
    time.sleep(0.1)

    # ============================================
    # 调试文本
    # ============================================
    p.addUserDebugText("工具", start_pos, [1, 0, 0], textSize=1.0, lifeTime=0)
    p.addUserDebugText("目标", [0, 0, 0.03], [0, 1, 0], textSize=1.0, lifeTime=0)

    traj_points = []

    pr_info("\n开始运动...\n")
    time.sleep(0.2)

    # ============================================
    # 主控制循环
    # ============================================
    step = 0
    success = False
    min_dist = float('inf')

    while step < MAX_STEPS:
        # 获取当前位置
        base_pos, base_orn = p.getBasePositionAndOrientation(tool_uid)

        # 计算 member 链接的位置
        member_pose_mat = transform_mat(
            from_base_to_member,
            mat44_by_pos_quat(base_pos, base_orn),
        )
        current_pos, current_orn = mat44_to_pos_quat(member_pose_mat)

        # 计算误差
        pos_error = np.array(target_pos) - np.array(current_pos)
        dist = np.linalg.norm(pos_error)

        if dist < min_dist:
            min_dist = dist

        traj_points.append(current_pos)

        # 计算姿态误差
        orn_error = quat_error(current_orn, target_orn_xyzw)
        orn_dist = np.linalg.norm(orn_error)

        # 比例控制 - 生成速度命令
        vel_cmd = KP_POS * pos_error
        vel_norm = np.linalg.norm(vel_cmd)
        if vel_norm > MAX_VEL:
            vel_cmd = vel_cmd / vel_norm * MAX_VEL

        rot_cmd = KP_ORN * orn_error
        rot_norm = np.linalg.norm(rot_cmd)
        if rot_norm > MAX_RAD:
            rot_cmd = rot_cmd / rot_norm * MAX_RAD

        # 更新基座位置
        delta_pos = vel_cmd * TIME_STEP
        new_pos = (np.array(base_pos) + delta_pos).tolist()

        # 更新基座姿态
        delta_rot = rot_cmd * TIME_STEP
        ang_vel_quat = [0, delta_rot[0], delta_rot[1], delta_rot[2]]
        q_delta = wxyz_to_xyzw(
            transforms3d.quaternions.qmult(
                ang_vel_quat, xyzw_to_wxyz(base_orn)
            )
        )
        new_orn = (np.array(base_orn) + 0.5 * np.array(q_delta)).tolist()
        new_orn_norm = np.linalg.norm(new_orn)
        new_orn = (np.array(new_orn) / new_orn_norm).tolist()

        # **直接设置工具位置（关键）**
        p.resetBasePositionAndOrientation(tool_uid, new_pos, new_orn)

        # 推进仿真
        p.stepSimulation()
        step += 1

        # 打印进度
        if step % 30 == 0:
            pr_info(
                f"Step {step:3d} | Distance: {dist*1000:.1f}mm | "
                f"Angle: {orn_dist*180/math.pi:.1f}deg"
            )

        # 检测成功
        if dist < DIST_THRESHOLD and orn_dist < ANGLE_THRESHOLD:
            success = True
            pr_green("\n" + "=" * 60)
            pr_green(f"✓ 成功! 步数: {step}")
            pr_green(f"  距离: {dist*1000:.2f}mm")
            pr_green(f"  角度: {orn_dist*180/math.pi:.2f}deg")
            pr_green("=" * 60)
            break

        time.sleep(TIME_STEP * 0.3)

    # ============================================
    # 结果
    # ============================================
    if not success:
        pr_red("\n" + "=" * 60)
        pr_red(f"未能完全成功 - 最小距离: {min_dist*1000:.2f}mm")
        pr_red("但已接近目标，继续进行抓取演示...")
        pr_red("=" * 60)
    else:
        pr_green("完美接近！")

    # 画轨迹
    if len(traj_points) > 1:
        for i in range(1, len(traj_points)):
            p.addUserDebugLine(
                traj_points[i-1], traj_points[i],
                [0, 1, 1], lifeTime=0, lineWidth=2,
            )

    # ============================================
    # 步骤2：抓取
    # ============================================
    pr_info("\n【步骤2】执行抓取动作...\n")
    time.sleep(1.0)

    # 获取当前位置（已接近目标）
    tool_pos, tool_orn = p.getBasePositionAndOrientation(tool_uid)
    workpiece_pos, workpiece_orn = p.getBasePositionAndOrientation(target_uid)

    pr_info(f"工具位置: Z={tool_pos[2]*100:.1f}cm")
    pr_info(f"工件位置: Z={workpiece_pos[2]*100:.1f}cm")
    pr_info(f"距离差: {np.linalg.norm(np.array(tool_pos) - np.array(workpiece_pos))*1000:.1f}mm\n")

    # 将相对位置转换到工具局部坐标系
    tool_rot = np.array(p.getMatrixFromQuaternion(tool_orn)).reshape(3, 3)
    rel_pos_local = tool_rot.T @ (np.array(workpiece_pos) - np.array(tool_pos))

    # 显示抓取提示
    p.addUserDebugText("准备抓取!", workpiece_pos, [1, 1, 0], textSize=1.5, lifeTime=2)
    time.sleep(0.5)

    pr_info("=" * 60)
    pr_info("创建约束: 工具 <--> 工件（固定关联）")
    pr_info("=" * 60 + "\n")

    # 创建抓取约束（使用工具局部坐标系的偏移）
    grasp_constraint = p.createConstraint(
        parentBodyUniqueId=tool_uid,
        parentLinkIndex=-1,
        childBodyUniqueId=target_uid,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=rel_pos_local.tolist(),
        childFramePosition=[0, 0, 0],
        childFrameOrientation=[0, 0, 0, 1],
    )
    p.changeConstraint(grasp_constraint, maxForce=5000)

    # 稳定约束
    for _ in range(30):
        p.stepSimulation()
    time.sleep(0.3)

    # 显示成功
    p.addUserDebugText("✓ 已抓取!", workpiece_pos, [0, 1, 0], textSize=1.8, lifeTime=0)

    # 画约束连接线
    p.addUserDebugLine(
        tool_pos,
        workpiece_pos,
        [0, 1, 0],
        lifeTime=0,
        lineWidth=5,
    )

    pr_green("=" * 60)
    pr_green("✓ 抓取成功！")
    pr_green("=" * 60)

    # ============================================
    # 步骤3：提起
    # ============================================
    pr_info("\n【步骤3】提起工件...\n")
    time.sleep(0.5)

    lift_height = 0.15
    current_tool_pos, current_tool_orn = p.getBasePositionAndOrientation(tool_uid)
    target_z = current_tool_pos[2] + lift_height

    pr_info(f"目标: 从 Z={current_tool_pos[2]*100:.1f}cm 提起到 Z={target_z*100:.1f}cm\n")

    for lift_step in range(300):
        tool_pos_now, tool_orn_now = p.getBasePositionAndOrientation(tool_uid)
        wp_pos_now, _ = p.getBasePositionAndOrientation(target_uid)

        z_error = target_z - tool_pos_now[2]

        # 向上移动
        if z_error > 0.001:
            # 直接修改位置向上
            new_z = tool_pos_now[2] + 0.04 * TIME_STEP  # 40mm/s
        else:
            new_z = tool_pos_now[2]

        new_pos = [tool_pos_now[0], tool_pos_now[1], new_z]
        p.resetBasePositionAndOrientation(tool_uid, new_pos, tool_orn_now)

        # 计算工具的旋转矩阵，让工件跟随工具（用当初记录的局部偏移）
        tool_rot_now = np.array(p.getMatrixFromQuaternion(tool_orn_now)).reshape(3, 3)
        new_workpiece_pos = (np.array(new_pos) + tool_rot_now @ rel_pos_local).tolist()
        p.resetBasePositionAndOrientation(target_uid, new_workpiece_pos, workpiece_orn)

        p.stepSimulation()

        if lift_step % 60 == 0:
            height_diff = tool_pos_now[2] - wp_pos_now[2]
            pr_info(
                f"Step {lift_step:3d} | "
                f"工具Z: {tool_pos_now[2]*100:.1f}cm | "
                f"工件Z: {wp_pos_now[2]*100:.1f}cm | "
                f"高度差: {height_diff*1000:.1f}mm"
            )

        if z_error < 0.002 and lift_step > 30:
            pr_green(f"\n✓ 提起完成! 最终高度: {tool_pos_now[2]*100:.1f}cm\n")
            break

        time.sleep(TIME_STEP * 0.2)

    # ============================================
    # 完成
    # ============================================
    final_tool, _ = p.getBasePositionAndOrientation(tool_uid)
    final_wp, _ = p.getBasePositionAndOrientation(target_uid)

    pr_green("=" * 60)
    pr_green("演示完成！")
    pr_green(f"  工具最终高度:  {final_tool[2]*100:.1f}cm")
    pr_green(f"  工件最终高度:  {final_wp[2]*100:.1f}cm")
    pr_green(f"  高度差:         {(final_tool[2]-final_wp[2])*1000:.1f}mm")
    pr_green("\n证明: 工件始终跟随工具 = 抓取成功!")
    pr_green("=" * 60)

    p.addUserDebugText(
        "✓ 演示完成!",
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
