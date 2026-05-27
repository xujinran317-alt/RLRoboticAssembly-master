"""
demo_grasp_based_assembly.py - 基于装配脚本的抓取演示

复用 demo_auto_assembly.py 的移动方式
"""

import time
import math
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pybullet as p
import transforms3d

from assembly_env.robots.sim_robotless import INITIAL_ORN
from utils.transforms import (
    xyzw_to_wxyz, wxyz_to_xyzw,
    mat44_by_pos_quat, mat44_to_pos_quat,
    get_f1_to_f2_xform, transform_mat,
)
from utils.io_utils import pr_green, pr_red, pr_info, display_frame_axis, format_urdf_filepath


# ============================================================
# 参数（直接用 demo_auto_assembly 的参数）
# ============================================================
KP_POS = 2.5
KP_ORN = 2.0
MAX_VEL = 0.15
MAX_RAD = 0.15
TIME_STEP = 1 / 100

OFFSET_X = 0.04
OFFSET_Y = 0.03
OFFSET_Z = 0.01

DIST_THRESHOLD = 0.02  # 宽松一些
ANGLE_THRESHOLD = 0.2
MAX_STEPS = 200


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
    pr_info("=" * 60)
    pr_info("抓取演示（基于装配移动方式）")
    pr_info("=" * 60)

    # ============================================
    # 初始化 PyBullet
    # ============================================
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(0.6, 180, -41, [0, 0, 0])
    p.setPhysicsEngineParameter(numSolverIterations=150, enableFileCaching=0)
    p.setTimeStep(TIME_STEP)
    p.setGravity(0, 0, -9.8)

    # ============================================
    # 目标位姿
    # ============================================
    target_pos = np.array([0.0, 0.0, 0.0])
    target_orn_euler = np.array([0, 0, math.pi])
    target_orn_xyzw = list(transforms3d.euler.euler2quat(
        target_orn_euler[0], target_orn_euler[1], target_orn_euler[2], 'sxyz'
    ))
    target_orn_xyzw = wxyz_to_xyzw(target_orn_xyzw)

    start_pos = [OFFSET_X, OFFSET_Y, 0.24 + OFFSET_Z]

    pr_info(f"目标位置: {target_pos}")
    pr_info(f"初始位置: {start_pos}")
    pr_info("正在启动仿真...\n")

    # ============================================
    # 加载模型
    # ============================================
    tool_uid = p.loadURDF(
        format_urdf_filepath("envs/urdf/robotless_lap_joint/tool"),
        basePosition=start_pos,
        baseOrientation=INITIAL_ORN,
        useFixedBase=0,
    )

    workpiece_uid = p.loadURDF(
        format_urdf_filepath("envs/urdf/robotless_lap_joint/task_lap_90deg"),
        basePosition=[0, 0, 0],
        baseOrientation=wxyz_to_xyzw(
            transforms3d.euler.euler2quat(0, 0, math.pi, 'sxyz')
        ),
        useFixedBase=1,  # 装配阶段固定
    )

    display_frame_axis(workpiece_uid, 0, line_length=0.05)
    for li in range(p.getNumJoints(tool_uid)):
        display_frame_axis(tool_uid, li, line_length=0.03)

    p.enableJointForceTorqueSensor(tool_uid, 0)

    # 创建工具约束（用于移动）
    base_constraint = p.createConstraint(
        parentBodyUniqueId=tool_uid,
        parentLinkIndex=-1,
        childBodyUniqueId=-1,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=start_pos,
        childFrameOrientation=INITIAL_ORN,
    )

    # 稳定仿真
    for _ in range(10):
        p.stepSimulation()
    time.sleep(0.1)

    # 计算 member 变换
    base_pose = p.getBasePositionAndOrientation(tool_uid)
    member_pose = p.getLinkState(tool_uid, 2)[4:6]
    from_base_to_member = get_f1_to_f2_xform(base_pose, member_pose)

    # 显示文字
    p.addUserDebugText("工具", start_pos, [1, 0, 0], textSize=1.0, lifeTime=0)
    p.addUserDebugText("工件", [0, 0, -0.01], [0, 1, 0], textSize=1.0, lifeTime=0)

    # =====================================================
    # 步骤1：用P控制移动到工件位置
    # =====================================================
    pr_info("【步骤1】移动到工件...\n")

    step = 0
    success = False

    while step < MAX_STEPS:
        # 获取当前位置
        base_pos, base_orn = p.getBasePositionAndOrientation(tool_uid)

        # 计算 member 实际位姿
        member_pose_mat = transform_mat(
            from_base_to_member,
            mat44_by_pos_quat(base_pos, base_orn),
        )
        current_pos, current_orn = mat44_to_pos_quat(member_pose_mat)

        # 计算误差
        pos_error = np.array(target_pos) - np.array(current_pos)
        dist = np.linalg.norm(pos_error)

        orn_error = quat_error(current_orn, target_orn_xyzw)
        orn_dist = np.linalg.norm(orn_error)

        # P控制
        vel_cmd = KP_POS * pos_error
        vel_norm = np.linalg.norm(vel_cmd)
        if vel_norm > MAX_VEL:
            vel_cmd = vel_cmd / vel_norm * MAX_VEL

        rot_cmd = KP_ORN * orn_error
        rot_norm = np.linalg.norm(rot_cmd)
        if rot_norm > MAX_RAD:
            rot_cmd = rot_cmd / rot_norm * MAX_RAD

        # 更新位置
        delta_pos = vel_cmd * TIME_STEP
        delta_rot = rot_cmd * TIME_STEP

        new_pos = (np.array(base_pos) + delta_pos).tolist()

        ang_vel_quat = [0, delta_rot[0], delta_rot[1], delta_rot[2]]
        new_orn = np.add(
            base_orn,
            0.5 * np.array(wxyz_to_xyzw(
                transforms3d.quaternions.qmult(
                    ang_vel_quat, xyzw_to_wxyz(base_orn)
                )
            )),
        ).tolist()

        # 应用约束
        p.changeConstraint(base_constraint, new_pos, new_orn, maxForce=-1)
        p.stepSimulation()
        step += 1

        if step % 40 == 0:
            pr_info(f"Step {step:3d} | Distance: {dist*1000:.1f}mm")

        # 到达目标
        if dist < DIST_THRESHOLD and orn_dist < ANGLE_THRESHOLD:
            success = True
            pr_green(f"\n✓ 到达工件位置 (距离: {dist*1000:.2f}mm)\n")
            break

        time.sleep(TIME_STEP)

    if not success:
        pr_red(f"移动失败")
        p.disconnect()
        return

    # =====================================================
    # 步骤2：抓取（创建约束）
    # =====================================================
    pr_info("【步骤2】执行抓取动作...\n")
    time.sleep(0.5)

    # 获取当前位置
    tool_pos, tool_orn = p.getBasePositionAndOrientation(tool_uid)
    workpiece_pos, workpiece_orn = p.getBasePositionAndOrientation(workpiece_uid)

    pr_info(f"工具位置: Z={tool_pos[2]*100:.1f}cm")
    pr_info(f"工件位置: Z={workpiece_pos[2]*100:.1f}cm\n")

    # 相对位置
    rel_pos = (np.array(workpiece_pos) - np.array(tool_pos)).tolist()

    # 显示抓取提示
    p.addUserDebugText("准备抓取!", workpiece_pos, [1, 1, 0], textSize=1.5, lifeTime=2)

    time.sleep(0.5)

    pr_info("=" * 60)
    pr_info("创建约束: 工具 <--> 工件")
    pr_info("=" * 60 + "\n")

    # 创建抓取约束
    grasp_constraint = p.createConstraint(
        parentBodyUniqueId=tool_uid,
        parentLinkIndex=-1,
        childBodyUniqueId=workpiece_uid,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=rel_pos,
        childFramePosition=[0, 0, 0],
        childFrameOrientation=workpiece_orn,
    )
    p.changeConstraint(grasp_constraint, maxForce=1000)

    # 稳定约束
    for _ in range(30):
        p.stepSimulation()
    time.sleep(0.3)

    # 显示成功
    p.addUserDebugText("✓ 已抓取!", workpiece_pos, [0, 1, 0], textSize=1.8, lifeTime=0)

    # 画连接线
    p.addUserDebugLine(
        tool_pos,
        workpiece_pos,
        [0, 1, 0],
        lifeTime=0,
        lineWidth=4,
    )

    pr_green("=" * 60)
    pr_green("✓ 抓取成功！")
    pr_green("=" * 60)

    # =====================================================
    # 步骤3：提起
    # =====================================================
    pr_info("\n【步骤3】提起工件...\n")
    time.sleep(0.3)

    lift_height = 0.15
    current_pos, current_orn = p.getBasePositionAndOrientation(tool_uid)
    target_z = current_pos[2] + lift_height

    for lift_step in range(250):
        base_pos, base_orn = p.getBasePositionAndOrientation(tool_uid)
        wp_pos, _ = p.getBasePositionAndOrientation(workpiece_uid)

        z_error = target_z - base_pos[2]

        # 向上
        delta_z = 0.03 * TIME_STEP
        new_pos = [base_pos[0], base_pos[1], base_pos[2] + delta_z]

        p.changeConstraint(base_constraint, new_pos, base_orn, maxForce=-1)
        p.stepSimulation()

        if lift_step % 50 == 0:
            pr_info(
                f"Step {lift_step:3d} | "
                f"工具Z: {base_pos[2]*100:.1f}cm | "
                f"工件Z: {wp_pos[2]*100:.1f}cm"
            )

        if z_error < 0.002 and lift_step > 30:
            pr_green(f"\n✓ 提起完成!\n")
            break

        time.sleep(TIME_STEP)

    # =====================================================
    # 完成
    # =====================================================
    final_tool, _ = p.getBasePositionAndOrientation(tool_uid)
    final_wp, _ = p.getBasePositionAndOrientation(workpiece_uid)

    pr_green("=" * 60)
    pr_green("演示完成！")
    pr_green(f"  工具高度: {final_tool[2]*100:.1f}cm")
    pr_green(f"  工件高度: {final_wp[2]*100:.1f}cm")
    pr_green("=" * 60)

    pr_info("\n按 Ctrl+C 退出\n")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    p.disconnect()


if __name__ == "__main__":
    run_demo()
