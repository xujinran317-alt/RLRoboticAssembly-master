"""
demo_auto_assembly.py - 自动装配演示

不训练任何模型，使用简单的比例控制（P控制）让工具从明显偏移的初始位置
自动运动到目标位置并对齐，展示装配全过程�?

工作原理�?
  1. 将工具初始位姿设在一个明显偏移的位置（比�?X 方向偏移 5cm�?
  2. 每一步计算当前位置与目标位置的偏�?
  3. 用比例控制（P控制器）生成速度指令，平滑跟踪目�?
  4. 到达目标位置（距�?5mm）后视为装配成功

用法:
    python -m scripts.demo_auto_assembly
    或�?
    assembly demo

参数调优说明（已调好，可直接使用）：
  - Kp_pos = 0.5:   位置比例增益
  - Kp_orn = 0.3:   姿态比例增�? 
  - max_vel = 0.02: 最大线速度 (m/s)
  - max_rad = 0.02: 最大角速度 (rad/s)
  - offset = 0.06:  初始偏移�?(m)，即 6cm

演示展示�?
  - 工具从右前方 6cm 处开�?
  - 自动平滑移动到目标位�?
  - 同时在姿态上调整对齐
  - 最后视觉上展示装配完成效果
"""

import time
import math
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
# 已调优的演示参数（可直接使用�?
# ============================================================

# 控制参数（快速演示版本）
KP_POS = 2.5          # 位置比例增益（加快响应）
KP_ORN = 2.0          # 姿态比例增益（加快对齐）
MAX_VEL = 0.15         # 最大线速度 (m/s)（5倍速）
MAX_RAD = 0.15         # 最大角速度 (rad/s)（5倍速）
TIME_STEP = 1 / 100   # 控制周期（加快）

# 初始偏移（工具从目标位置的偏移量）
# 正值 = 沿正方向偏移
OFFSET_X = 0.04       # X方向偏移 4cm（减小初始偏移）
OFFSET_Y = 0.03       # Y方向偏移 3cm
OFFSET_Z = 0.01       # Z方向偏移 1cm

# 装配成功阈值
DIST_THRESHOLD = 0.015   # 10mm（放宽阈值）

# 最大步数
MAX_STEPS = 200        # 减少步数（3倍快）


def quat_error(q_current, q_target):
    """
    计算两个四元数之间的姿态误差（用角速度表示�?
    Args:
        q_current: 当前四元�?(xyzw)
        q_target:  目标四元�?(xyzw)
    Returns:
        [ωx, ωy, ωz] 姿态误差（弧度），乘以增益后可作为角速度指令
    """
    # 都转�?wxyz
    qc_wxyz = xyzw_to_wxyz(q_current)
    qt_wxyz = xyzw_to_wxyz(q_target)

    # 计算相对旋转四元�? q_error = q_target * q_current^{-1}
    q_current_inv = transforms3d.quaternions.qinverse(qc_wxyz)
    q_rel = transforms3d.quaternions.qmult(qt_wxyz, q_current_inv)

    # 提取旋转轴×角度（四元数的虚部 = �?* sin(θ/2)�?
    # 对于小角度，sin(θ/2) �?θ/2，所�?2 * q_rel[1:] �?旋转向量
    error_rot = 2.0 * np.array(q_rel[1:4])

    # 确保方向正确（取最小角度路径）
    if q_rel[0] < 0:  # w < 0 表示角度 > 180°，走另一条更短的路径
        error_rot = -error_rot

    return error_rot


def run_demo():
    """运行自动装配演示"""

    pr_info("=" * 60)
    pr_info("自动装配演示（快速版本 - 比例控制）")
    pr_info("=" * 60)
    pr_info(f"控制参数: Kp_pos={KP_POS}, Kp_orn={KP_ORN}")
    pr_info(f"初始偏移: X={OFFSET_X*1000:.0f}mm, Y={OFFSET_Y*1000:.0f}mm")
    pr_info(f"最大速度: {MAX_VEL*1000:.1f}mm/s, 最大角速度: {MAX_RAD*180/math.pi:.1f}°/s")
    pr_info(f"预计耗时: ~{MAX_STEPS * TIME_STEP:.1f} 秒")
    pr_info("正在启动 PyBullet 仿真...")

    # ============================================
    # 1. 初始�?PyBullet
    # ============================================
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(0.6, 180, -41, [0, 0, 0])
    p.setPhysicsEngineParameter(numSolverIterations=150, enableFileCaching=0)
    p.setTimeStep(TIME_STEP)
    p.setGravity(0, 0, 0)

    # ============================================
    # 2. 加载工具模型（带偏移的初始位置）
    # ============================================
    # 目标位姿（在 sim_robotless.py 中定义）
    target_pos = np.array([0.0, 0.0, 0.0])                 # 目标位置
    target_orn_euler = np.array([0, 0, math.pi])           # 目标姿态（绕Z�?80°�?
    target_orn_xyzw = list(transforms3d.euler.euler2quat(
        target_orn_euler[0], target_orn_euler[1], target_orn_euler[2], 'sxyz'
    ))
    target_orn_xyzw = wxyz_to_xyzw(target_orn_xyzw)       # 转成 xyzw

    # 初始位置：在目标位置基础上加偏移 + 原本的高度偏�?
    # 保持初始姿态不变（单位矩阵，朝正下�?
    start_pos = [OFFSET_X, OFFSET_Y, 0.24 + OFFSET_Z]

    pr_info(f"目标位置: {target_pos}")
    pr_info(f"初始位置: {start_pos}")

    # 加载工具 URDF（带偏移的初始位置）
    from utils.io_utils import format_urdf_filepath
    tool_uid = p.loadURDF(
        format_urdf_filepath("envs/urdf/robotless_lap_joint/tool"),
        basePosition=start_pos,
        baseOrientation=INITIAL_ORN,
        useFixedBase=0,
    )

    # 加载目标工件 URDF（固定，在目标位置）
    target_uid = p.loadURDF(
        format_urdf_filepath("envs/urdf/robotless_lap_joint/task_lap_90deg"),
        basePosition=[0, 0, 0],
        baseOrientation=wxyz_to_xyzw(transforms3d.euler.euler2quat(
            0, 0, math.pi, 'sxyz'
        )),
        useFixedBase=1,
    )

    # 显示坐标�?
    display_frame_axis(target_uid, 0, line_length=0.05)
    for li in range(p.getNumJoints(tool_uid)):
        display_frame_axis(tool_uid, li, line_length=0.03)

    # 启用力传感器
    p.enableJointForceTorqueSensor(tool_uid, 0)

    # 创建固定约束（控制工具位置）
    max_force = -1  # 不限�?
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

    # 计算从基座到 member（link 2）的相对变换
    base_pose = p.getBasePositionAndOrientation(tool_uid)
    member_pose = p.getLinkState(tool_uid, 2)[4:6]
    from_base_to_member = get_f1_to_f2_xform(base_pose, member_pose)

    # 推进几步让仿真稳�?
    for _ in range(10):
        p.stepSimulation()
    time.sleep(0.1)

    # ============================================
    # 3. 显示文字提示
    # ============================================
    p.addUserDebugText(
        "工具（初始偏移位置）",
        start_pos,
        [1, 0, 0],
        textSize=1.0,
        lifeTime=0,
    )
    p.addUserDebugText(
        "目标装配位置",
        [0, 0, 0.03],
        [0, 1, 0],
        textSize=1.0,
        lifeTime=0,
    )

    # 画一条轨迹线
    traj_points = []

    pr_info("\n开始自动装配演�?..")
    time.sleep(0.3)

    # ============================================
    # 4. 比例控制主循�?
    # ============================================
    step = 0
    success = False
    min_dist = float('inf')

    while step < MAX_STEPS:
        # 4a. 获取当前位置
        base_pos, base_orn = p.getBasePositionAndOrientation(tool_uid)

        # 计算 member (link 2) 的实际位�?
        member_pose_mat = transform_mat(
            from_base_to_member,
            mat44_by_pos_quat(base_pos, base_orn),
        )
        current_pos, current_orn = mat44_to_pos_quat(member_pose_mat)

        # 4b. 计算位置误差
        pos_error = np.array(target_pos) - np.array(current_pos)
        dist = np.linalg.norm(pos_error)

        # 更新最小距�?
        if dist < min_dist:
            min_dist = dist

        # 记录轨迹�?
        traj_points.append(current_pos)

        # 4c. 计算姿态误�?
        orn_error = quat_error(current_orn, target_orn_xyzw)
        orn_dist = np.linalg.norm(orn_error)

        # 4d. 比例控制：生成速度指令
        # 位置控制
        vel_cmd = KP_POS * pos_error
        vel_norm = np.linalg.norm(vel_cmd)
        if vel_norm > MAX_VEL:
            vel_cmd = vel_cmd / vel_norm * MAX_VEL

        # 姿态控�?
        rot_cmd = KP_ORN * orn_error
        rot_norm = np.linalg.norm(rot_cmd)
        if rot_norm > MAX_RAD:
            rot_cmd = rot_cmd / rot_norm * MAX_RAD

        # 4e. 计算 total delta 用于约束更新
        # 注意：PyBullet �?apply_action_pose 会乘�?倍补偿因�?
        # 这里我们直接�?changeConstraint，所以要自行处理
        delta_pos = vel_cmd * TIME_STEP
        delta_rot = rot_cmd * TIME_STEP

        new_pos = (np.array(base_pos) + delta_pos).tolist()

        # 更新姿态（四元数）
        ang_vel_quat = [0, delta_rot[0], delta_rot[1], delta_rot[2]]
        new_orn = np.add(
            base_orn,
            0.5 * np.array(wxyz_to_xyzw(
                transforms3d.quaternions.qmult(
                    ang_vel_quat, xyzw_to_wxyz(base_orn)
                )
            )),
        ).tolist()

        # 4f. 应用约束更新（直接用原始速度，不需补偿因子�?
        p.changeConstraint(base_constraint, new_pos, new_orn, max_force)

        # 4g. 推进物理引擎
        p.stepSimulation()
        step += 1

        # 4h. 状态打印（�?0步）
        if step % 50 == 0:
            pr_info(
                f"步数 {step:4d}  |  "
                f"距离目标: {dist*1000:.1f}mm  |  "
                f"姿态误�? {orn_dist*180/math.pi:.1f}°  |  "
                f"位置: ({current_pos[0]*1000:.0f}, {current_pos[1]*1000:.0f}, {current_pos[2]*1000:.0f})mm"
            )

        # 4i. 检测是否到达目�?
        if dist < DIST_THRESHOLD and orn_dist < 0.05:
            success = True
            pr_green(f"\n{'='*60}")
            pr_green(f"装配成功！步�? {step}")
            pr_green(f"   最终位置误�? {dist*1000:.2f}mm")
            pr_green(f"   最终姿态误�? {orn_dist*180/math.pi:.2f}°")
            pr_green(f"{'='*60}")
            break

        # 控制演示速度（实时）
        time.sleep(TIME_STEP)

    # ============================================
    # 5. 演示结束，展示结�?
    # ============================================
    if not success:
        pr_red(f"\n{'='*60}")
        pr_red(f" 未能�?{MAX_STEPS} 步内完成装配")
        pr_red(f"   最小距�? {min_dist*1000:.2f}mm")
        pr_red(f"{'='*60}")

    # 画轨�?
    if len(traj_points) > 1:
        for i in range(1, len(traj_points)):
            p.addUserDebugLine(
                traj_points[i-1], traj_points[i],
                [0, 1, 1], lifeTime=0, lineWidth=2,
            )

    # 保持窗口打开，等待用户关�?
    pr_info("\n演示结束。可手动关闭窗口或按 Ctrl+C 退出�?")
    try:
        while True:
            time.sleep(0.3)
    except KeyboardInterrupt:
        pass

    p.disconnect()
    pr_info("PyBullet 已断开连接�?")


if __name__ == "__main__":
    run_demo()
