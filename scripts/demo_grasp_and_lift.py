"""
demo_grasp_and_lift.py - 抓取提起演示

流程:
  1. 用 demo_auto_assembly 的装配逻辑让 member 插入工件凹槽（P控制）
  2. 插入成功后创建固定约束把工件绑在工具上
  3. 控制工具向上提起，工件随之被抬起

用法:
    python -m scripts.demo_grasp_and_lift
    assembly grasp
"""

import time
import math
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
# 与 demo_auto_assembly 完全一致的参数（勿修改）
# ============================================================
KP_POS = 0.5          # 位置比例增益
KP_ORN = 0.3          # 姿态比例增益
MAX_VEL = 0.02        # 最大线速度 (m/s)
MAX_RAD = 0.02        # 最大角速度 (rad/s)
TIME_STEP = 1 / 250   # 控制周期

OFFSET_X = 0.06       # X方向初始偏移 6cm
OFFSET_Y = 0.04       # Y方向初始偏移 4cm
OFFSET_Z = 0.02       # Z方向额外偏移
START_Z = 0.24 + OFFSET_Z

DIST_THRESHOLD = 0.005  # 装配成功阈值 5mm
MAX_STEPS = 1000        # 装配阶段最大步数

# 提起阶段参数
LIFT_HEIGHT = 0.15      # 提起高度 15cm
LIFT_VEL = 0.01         # 提起速度 10mm/s
LIFT_STEPS = int(LIFT_HEIGHT / (LIFT_VEL * TIME_STEP))  # 自动计算步数


# ============================================================
# 工具函数（与 demo_auto_assembly 完全相同）
# ============================================================

def quat_error(q_current, q_target):
    """计算两个四元数之间的姿态误差（角速度形式）"""
    qc_wxyz = xyzw_to_wxyz(q_current)
    qt_wxyz = xyzw_to_wxyz(q_target)

    q_current_inv = transforms3d.quaternions.qinverse(qc_wxyz)
    q_rel = transforms3d.quaternions.qmult(qt_wxyz, q_current_inv)

    error_rot = 2.0 * np.array(q_rel[1:4])
    if q_rel[0] < 0:
        error_rot = -error_rot

    return error_rot


def p_control_step(base_pos, base_orn, target_pos, target_orn_xyzw, base_constraint):
    """
    执行一步P控制，更新约束位置。
    返回 (current_pos, dist, orn_dist)，供外部判断是否到达目标。
    """
    # 位置误差
    pos_error = np.array(target_pos) - np.array(base_pos)
    dist = np.linalg.norm(pos_error)

    # 姿态误差
    orn_error = quat_error(base_orn, target_orn_xyzw)
    orn_dist = np.linalg.norm(orn_error)

    # 位置控制指令（限幅）
    vel_cmd = KP_POS * pos_error
    vel_norm = np.linalg.norm(vel_cmd)
    if vel_norm > MAX_VEL:
        vel_cmd = vel_cmd / vel_norm * MAX_VEL

    # 姿态控制指令（限幅）
    rot_cmd = KP_ORN * orn_error
    rot_norm = np.linalg.norm(rot_cmd)
    if rot_norm > MAX_RAD:
        rot_cmd = rot_cmd / rot_norm * MAX_RAD

    # 计算新位姿
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

    return dist, orn_dist


# ============================================================
# 主演示函数
# ============================================================

def run_demo():
    pr_info("=" * 60)
    pr_info("抓取提起演示：装配 → 抓取 → 提起")
    pr_info("=" * 60)
    pr_info(f"装配参数: Kp_pos={KP_POS}, Kp_orn={KP_ORN}, "
            f"MAX_VEL={MAX_VEL}, MAX_RAD={MAX_RAD}")
    pr_info(f"提起高度: {LIFT_HEIGHT*100:.0f}cm，提起速度: {LIFT_VEL*1000:.0f}mm/s")
    pr_info("正在启动 PyBullet 仿真...")

    # --------------------------------------------------
    # 1. 初始化 PyBullet
    # --------------------------------------------------
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(0.6, 180, -41, [0, 0, 0])
    p.setPhysicsEngineParameter(numSolverIterations=150, enableFileCaching=0)
    p.setTimeStep(TIME_STEP)
    p.setGravity(0, 0, -9.8)

    # --------------------------------------------------
    # 2. 目标位姿定义
    # --------------------------------------------------
    target_pos = np.array([0.0, 0.0, 0.0])
    target_orn_xyzw = wxyz_to_xyzw(
        transforms3d.euler.euler2quat(0, 0, math.pi, 'sxyz')
    )

    start_pos = [OFFSET_X, OFFSET_Y, START_Z]

    pr_info(f"初始位置: {start_pos}")
    pr_info(f"目标位置: {target_pos.tolist()}")

    # --------------------------------------------------
    # 3. 加载 URDF
    # --------------------------------------------------
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
        useFixedBase=0,   # 不固定，这样才能被提起
    )

    # 显示坐标轴
    display_frame_axis(target_uid, 0, line_length=0.05)
    for li in range(p.getNumJoints(tool_uid)):
        display_frame_axis(tool_uid, li, line_length=0.03)

    p.enableJointForceTorqueSensor(tool_uid, 0)

    # --------------------------------------------------
    # 4. 创建工具约束（用于P控制）
    # --------------------------------------------------
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

    # 固定工件（装配阶段工件不动）
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

    # 计算 base → member 变换
    for _ in range(10):
        p.stepSimulation()
    time.sleep(0.1)

    base_pose = p.getBasePositionAndOrientation(tool_uid)
    member_pose = p.getLinkState(tool_uid, 2)[4:6]
    from_base_to_member = get_f1_to_f2_xform(base_pose, member_pose)

    # --------------------------------------------------
    # 5. 显示提示文字
    # --------------------------------------------------
    p.addUserDebugText("工具（初始位置）", start_pos, [1, 0, 0], textSize=1.0)
    p.addUserDebugText("目标装配位置", [0, 0, 0.03], [0, 1, 0], textSize=1.0)

    traj_points = []

    # =====================================================
    # 阶段一：装配（P控制，与 demo_auto_assembly 完全一致）
    # =====================================================
    pr_info("\n【阶段一】开始自动装配...")
    time.sleep(0.3)

    step = 0
    success = False
    min_dist = float('inf')

    while step < MAX_STEPS:
        base_pos, base_orn = p.getBasePositionAndOrientation(tool_uid)

        # 计算 member 实际位姿（用于判断距离）
        member_pose_mat = transform_mat(
            from_base_to_member,
            mat44_by_pos_quat(base_pos, base_orn),
        )
        current_pos, current_orn = mat44_to_pos_quat(member_pose_mat)
        traj_points.append(current_pos)

        dist, orn_dist = p_control_step(
            base_pos, base_orn,
            target_pos, target_orn_xyzw,
            base_constraint,
        )

        if dist < min_dist:
            min_dist = dist

        p.stepSimulation()
        step += 1

        if step % 50 == 0:
            pr_info(
                f"[装配] 步数 {step:4d}  |  "
                f"距目标: {dist*1000:.1f}mm  |  "
                f"姿态误差: {orn_dist*180/math.pi:.1f}°"
            )

        if dist < DIST_THRESHOLD and orn_dist < 0.05:
            success = True
            pr_green(f"\n✅ 装配成功！步数: {step}，"
                     f"位置误差: {dist*1000:.2f}mm，"
                     f"姿态误差: {orn_dist*180/math.pi:.2f}°")
            break

        time.sleep(TIME_STEP)

    if not success:
        pr_red(f"\n❌ 装配失败：{MAX_STEPS} 步内未到达目标（最近: {min_dist*1000:.2f}mm）")
        # 画轨迹后退出
        for i in range(1, len(traj_points)):
            p.addUserDebugLine(traj_points[i-1], traj_points[i],
                               [0, 1, 1], lifeTime=0, lineWidth=2)
        pr_info("按 Ctrl+C 退出")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        p.disconnect()
        return

    # 画装配轨迹
    for i in range(1, len(traj_points)):
        p.addUserDebugLine(traj_points[i-1], traj_points[i],
                           [0, 1, 1], lifeTime=0, lineWidth=2)

    # =====================================================
    # 阶段二：解除工件固定约束，绑定到工具上
    # =====================================================
    pr_info("\n【阶段二】绑定工件...")
    time.sleep(0.5)

    # 移除工件的地面固定约束
    p.removeConstraint(target_constraint)

    # 获取当前工具和工件位姿，创建相对约束
    tool_pos, tool_orn = p.getBasePositionAndOrientation(tool_uid)
    workpiece_pos, workpiece_orn = p.getBasePositionAndOrientation(target_uid)

    # 工件相对于工具的偏移
    rel_pos = (np.array(workpiece_pos) - np.array(tool_pos)).tolist()

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

    # 推进几步让约束稳定
    for _ in range(20):
        p.stepSimulation()
    time.sleep(0.2)

    pr_green("✅ 工件已绑定，准备提起")
    p.addUserDebugText("工件已抓取！", [0, 0, 0.05], [1, 1, 0], textSize=1.2, lifeTime=3)

    # =====================================================
    # 阶段三：提起
    # =====================================================
    pr_info(f"\n【阶段三】开始提起（目标高度: {LIFT_HEIGHT*100:.0f}cm）...")
    time.sleep(0.3)

    lift_traj = []
    current_tool_pos, current_tool_orn = p.getBasePositionAndOrientation(tool_uid)
    lift_target_z = current_tool_pos[2] + LIFT_HEIGHT

    for lift_step in range(LIFT_STEPS + 200):  # 额外200步让系统稳定
        current_tool_pos, current_tool_orn = p.getBasePositionAndOrientation(tool_uid)
        lift_traj.append(list(current_tool_pos))

        # 目标：保持XY不变，Z向上
        lift_target = [current_tool_pos[0], current_tool_pos[1], lift_target_z]

        z_error = lift_target_z - current_tool_pos[2]
        z_vel = min(KP_POS * z_error, LIFT_VEL)
        new_z = current_tool_pos[2] + z_vel * TIME_STEP

        new_pos = [current_tool_pos[0], current_tool_pos[1], new_z]
        p.changeConstraint(base_constraint, new_pos, current_tool_orn, maxForce=-1)

        p.stepSimulation()

        if lift_step % 100 == 0:
            wp_pos, _ = p.getBasePositionAndOrientation(target_uid)
            pr_info(
                f"[提起] 步数 {lift_step:4d}  |  "
                f"工具Z: {current_tool_pos[2]*100:.1f}cm  |  "
                f"工件Z: {wp_pos[2]*100:.1f}cm  |  "
                f"还需上升: {z_error*100:.1f}cm"
            )

        # 到达目标高度后再稳定100步
        if z_error < 0.002 and lift_step > 50:
            pr_green(f"\n✅ 提起完成！当前高度: {current_tool_pos[2]*100:.1f}cm")
            break

        time.sleep(TIME_STEP)

    # 画提起轨迹（红色）
    for i in range(1, len(lift_traj)):
        p.addUserDebugLine(lift_traj[i-1], lift_traj[i],
                           [1, 0.3, 0], lifeTime=0, lineWidth=2)

    # =====================================================
    # 完成
    # =====================================================
    final_tool, _ = p.getBasePositionAndOrientation(tool_uid)
    final_wp, _ = p.getBasePositionAndOrientation(target_uid)

    pr_green(f"\n{'='*60}")
    pr_green("演示完成：装配 ✅  抓取 ✅  提起 ✅")
    pr_green(f"  工具最终位置:  Z = {final_tool[2]*100:.1f}cm")
    pr_green(f"  工件最终位置:  Z = {final_wp[2]*100:.1f}cm")
    pr_green(f"{'='*60}")

    p.addUserDebugText(
        "完成！工件已被提起",
        [final_tool[0], final_tool[1], final_tool[2] + 0.05],
        [0, 1, 0], textSize=1.5, lifeTime=0,
    )

    pr_info("\n按 Ctrl+C 退出。")
    try:
        while True:
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass

    p.disconnect()
    pr_info("PyBullet 已断开。")


if __name__ == "__main__":
    run_demo()
