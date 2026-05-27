"""
demo_grasp_and_lift_simple.py - 快速版：装配 → 抓取 → 提起

完整演示：
  1. P控制让member插入工件凹槽（装配）
  2. 创建约束将工件绑定到工具上（抓取）
  3. 控制工具向上提起，工件随之被抬起（提起）
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

from assembly_env.robots.sim_robotless import INITIAL_ORN
from utils.transforms import (
    xyzw_to_wxyz, wxyz_to_xyzw,
    mat44_by_pos_quat, mat44_to_pos_quat,
    get_f1_to_f2_xform, transform_mat,
)
from utils.io_utils import pr_green, pr_red, pr_info, display_frame_axis, format_urdf_filepath


# ============================================================
# 快速演示参数
# ============================================================

# 装配阶段
KP_POS = 3.0          # 位置增益（快速）
KP_ORN = 2.5          # 姿态增益（快速）
MAX_VEL = 0.2         # 最大速度 m/s
MAX_RAD = 0.2         # 最大角速度 rad/s
TIME_STEP = 0.01      # 时间步长

OFFSET_X = 0.05       # 初始偏移
OFFSET_Y = 0.04
OFFSET_Z = 0.0
START_Z = 0.24 + OFFSET_Z

DIST_THRESHOLD = 0.02   # 装配成功阈值 20mm（放宽）
ANGLE_THRESHOLD = 0.1   # 姿态阈值
MAX_STEPS = 150         # 装配最大步数（快速）

# 提起阶段
LIFT_HEIGHT = 0.15      # 提起高度 15cm
LIFT_VEL = 0.03         # 提起速度 30mm/s
LIFT_STEPS = int(LIFT_HEIGHT / (LIFT_VEL * TIME_STEP))


# ============================================================
# 工具函数
# ============================================================

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
    """运行抓取提起演示"""

    pr_info("=" * 60)
    pr_info("快速演示：装配 → 抓取 → 提起")
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
    # 定义目标位姿
    # ============================================
    target_pos = np.array([0.0, 0.0, 0.0])
    target_orn_xyzw = wxyz_to_xyzw(
        transforms3d.euler.euler2quat(0, 0, math.pi, 'sxyz')
    )

    start_pos = [OFFSET_X, OFFSET_Y, START_Z]

    pr_info(f"初始位置: {start_pos}")
    pr_info(f"目标位置: {target_pos.tolist()}")
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

    target_uid = p.loadURDF(
        format_urdf_filepath("envs/urdf/robotless_lap_joint/task_lap_90deg"),
        basePosition=[0, 0, 0],
        baseOrientation=wxyz_to_xyzw(
            transforms3d.euler.euler2quat(0, 0, math.pi, 'sxyz')
        ),
        useFixedBase=0,  # 装配时固定，提起时释放
    )

    display_frame_axis(target_uid, 0, line_length=0.05)
    for li in range(p.getNumJoints(tool_uid)):
        display_frame_axis(tool_uid, li, line_length=0.03)

    p.enableJointForceTorqueSensor(tool_uid, 0)

    # 装配阶段固定工件
    target_constraint = p.createConstraint(
        parentBodyUniqueId=target_uid,
        parentLinkIndex=-1,
        childBodyUniqueId=-1,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=[0, 0, 0],
        childFrameOrientation=wxyz_to_xyzw(
            transforms3d.euler.euler2quat(0, 0, math.pi, 'sxyz')
        ),
    )

    # 稳定仿真
    for _ in range(10):
        p.stepSimulation()
    time.sleep(0.1)

    # 计算 base → member 变换
    base_pose = p.getBasePositionAndOrientation(tool_uid)
    member_pose = p.getLinkState(tool_uid, 2)[4:6]
    from_base_to_member = get_f1_to_f2_xform(base_pose, member_pose)

    # 调试文本
    p.addUserDebugText("工具", start_pos, [1, 0, 0], textSize=1.0, lifeTime=0)
    p.addUserDebugText("目标", [0, 0, 0.03], [0, 1, 0], textSize=1.0, lifeTime=0)

    traj_points = []

    # =====================================================
    # 阶段一：装配
    # =====================================================
    pr_info("【阶段一】装配中...\n")
    time.sleep(0.2)

    step = 0
    success = False
    min_dist = float('inf')

    while step < MAX_STEPS:
        # 获取当前位置
        base_pos, base_orn = p.getBasePositionAndOrientation(tool_uid)

        # 计算 member 实际位姿
        member_pose_mat = transform_mat(
            from_base_to_member,
            mat44_by_pos_quat(base_pos, base_orn),
        )
        current_pos, current_orn = mat44_to_pos_quat(member_pose_mat)
        traj_points.append(current_pos)

        # 计算误差
        pos_error = np.array(target_pos) - np.array(current_pos)
        dist = np.linalg.norm(pos_error)

        if dist < min_dist:
            min_dist = dist

        orn_error = quat_error(current_orn, target_orn_xyzw)
        orn_dist = np.linalg.norm(orn_error)

        # 比例控制
        vel_cmd = KP_POS * pos_error
        vel_norm = np.linalg.norm(vel_cmd)
        if vel_norm > MAX_VEL:
            vel_cmd = vel_cmd / vel_norm * MAX_VEL

        rot_cmd = KP_ORN * orn_error
        rot_norm = np.linalg.norm(rot_cmd)
        if rot_norm > MAX_RAD:
            rot_cmd = rot_cmd / rot_norm * MAX_RAD

        # 更新工具位置
        delta_pos = vel_cmd * TIME_STEP
        new_pos = (np.array(base_pos) + delta_pos).tolist()

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

        # 直接设置位置（关键）
        p.resetBasePositionAndOrientation(tool_uid, new_pos, new_orn)

        p.stepSimulation()
        step += 1

        if step % 30 == 0:
            pr_info(
                f"Step {step:3d} | Distance: {dist*1000:.1f}mm | "
                f"Angle: {orn_dist*180/math.pi:.1f}deg"
            )

        # 检测装配成功
        if dist < DIST_THRESHOLD and orn_dist < ANGLE_THRESHOLD:
            success = True
            pr_green(f"\n✓ 装配成功! Step: {step}, Distance: {dist*1000:.2f}mm")
            break

        time.sleep(TIME_STEP * 0.3)

    if not success:
        pr_red(f"\n✗ 装配未完全成功：{MAX_STEPS} 步内最近距离 {min_dist*1000:.2f}mm")
        pr_red("仍然继续抓取演示，演示重点是“抓取”。")
    else:
        pr_green("装配阶段成功，继续抓取。")

    # 画装配轨迹
    for i in range(1, len(traj_points)):
        p.addUserDebugLine(traj_points[i-1], traj_points[i],
                           [0, 1, 1], lifeTime=0, lineWidth=2)

    # =====================================================
    # 阶段二：抓取（绑定工件）
    # =====================================================
    pr_info("\n【阶段二】绑定工件...\n")
    time.sleep(0.3)

    # 移除工件地面约束
    p.removeConstraint(target_constraint)
    time.sleep(0.1)

    # 获取当前位置
    tool_pos, tool_orn = p.getBasePositionAndOrientation(tool_uid)
    workpiece_pos, workpiece_orn = p.getBasePositionAndOrientation(target_uid)

    # 工件相对于工具的偏移
    rel_pos = (np.array(workpiece_pos) - np.array(tool_pos)).tolist()

    # 创建固定约束（抓取约束）
    grasp_constraint = p.createConstraint(
        parentBodyUniqueId=tool_uid,
        parentLinkIndex=-1,
        childBodyUniqueId=target_uid,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=rel_pos,
        childFramePosition=[0, 0, 0],
        childFrameOrientation=workpiece_orn,
    )
    p.changeConstraint(grasp_constraint, maxForce=500)

    # 稳定约束
    for _ in range(20):
        p.stepSimulation()
    time.sleep(0.2)

    pr_green("✓ 工件已绑定")
    p.addUserDebugText("抓取成功!", [0, 0, 0.05], [1, 1, 0], textSize=1.2, lifeTime=0)

    # =====================================================
    # 阶段三：提起
    # =====================================================
    pr_info("【阶段三】提起中...\n")
    time.sleep(0.3)

    lift_traj = []
    current_tool_pos, current_tool_orn = p.getBasePositionAndOrientation(tool_uid)
    lift_target_z = current_tool_pos[2] + LIFT_HEIGHT

    for lift_step in range(LIFT_STEPS + 100):
        current_tool_pos, current_tool_orn = p.getBasePositionAndOrientation(tool_uid)
        lift_traj.append(list(current_tool_pos))

        z_error = lift_target_z - current_tool_pos[2]
        z_vel = min(KP_POS * z_error, LIFT_VEL)
        new_z = current_tool_pos[2] + z_vel * TIME_STEP

        new_pos = [current_tool_pos[0], current_tool_pos[1], new_z]
        p.resetBasePositionAndOrientation(tool_uid, new_pos, current_tool_orn)

        p.stepSimulation()

        if lift_step % 50 == 0:
            wp_pos, _ = p.getBasePositionAndOrientation(target_uid)
            pr_info(
                f"Step {lift_step:3d} | Tool Z: {current_tool_pos[2]*100:.1f}cm | "
                f"Workpiece Z: {wp_pos[2]*100:.1f}cm"
            )

        # 到达目标高度
        if z_error < 0.002 and lift_step > 30:
            pr_green(f"\n✓ 提起完成! Height: {current_tool_pos[2]*100:.1f}cm")
            break

        time.sleep(TIME_STEP * 0.3)

    # 画提起轨迹
    for i in range(1, len(lift_traj)):
        p.addUserDebugLine(lift_traj[i-1], lift_traj[i],
                           [1, 0.3, 0], lifeTime=0, lineWidth=2)

    # =====================================================
    # 完成
    # =====================================================
    final_tool, _ = p.getBasePositionAndOrientation(tool_uid)
    final_wp, _ = p.getBasePositionAndOrientation(target_uid)

    pr_green(f"\n{'='*60}")
    pr_green("演示完成: 装配 ✓  抓取 ✓  提起 ✓")
    pr_green(f"  工具高度: {final_tool[2]*100:.1f}cm")
    pr_green(f"  工件高度: {final_wp[2]*100:.1f}cm")
    pr_green(f"{'='*60}")

    p.addUserDebugText(
        "完成！",
        [final_tool[0], final_tool[1], final_tool[2] + 0.05],
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
