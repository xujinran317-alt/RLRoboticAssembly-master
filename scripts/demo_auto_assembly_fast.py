"""
demo_auto_assembly_fast.py - Fast, simple assembly demo

A quick demonstration that shows simple proportional control movement.
No complex training, just smooth motion from offset position to target.

Quick Start:
    python -m scripts.demo_auto_assembly_fast
    or
    python scripts/demo_auto_assembly_fast.py
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
# FAST DEMO PARAMETERS
# ============================================================

# Control gains (tuned for quick response)
KP_POS = 2.5          # Position gain (faster convergence)
KP_ORN = 2.0          # Orientation gain (faster alignment)
MAX_VEL = 0.15        # Max linear velocity (m/s)
MAX_RAD = 0.15        # Max angular velocity (rad/s)
TIME_STEP = 1 / 100   # Control loop frequency (faster)

# Initial offset (smaller for quicker demo)
OFFSET_X = 0.04       # 4cm offset in X
OFFSET_Y = 0.03       # 3cm offset in Y
OFFSET_Z = 0.01       # 1cm offset in Z

# Success threshold
DIST_THRESHOLD = 0.015  # 15mm (relaxed for faster success)

# Max steps (reduced for quick demo)
MAX_STEPS = 200


def quat_error(q_current, q_target):
    """
    Calculate orientation error between two quaternions (in xyzw format).
    Returns rotation vector (angular velocity equivalent).
    """
    qc_wxyz = xyzw_to_wxyz(q_current)
    qt_wxyz = xyzw_to_wxyz(q_target)

    q_current_inv = transforms3d.quaternions.qinverse(qc_wxyz)
    q_rel = transforms3d.quaternions.qmult(qt_wxyz, q_current_inv)

    error_rot = 2.0 * np.array(q_rel[1:4])

    if q_rel[0] < 0:
        error_rot = -error_rot

    return error_rot


def run_demo():
    """Run the fast assembly demo"""

    pr_info("=" * 60)
    pr_info("FAST ASSEMBLY DEMO - Simple Proportional Control")
    pr_info("=" * 60)
    pr_info(f"Gains: Kp_pos={KP_POS}, Kp_orn={KP_ORN}")
    pr_info(f"Initial offset: X={OFFSET_X*1000:.0f}mm, Y={OFFSET_Y*1000:.0f}mm")
    pr_info(f"Speed: {MAX_VEL*1000:.1f}mm/s, Rotation: {MAX_RAD*180/math.pi:.1f}deg/s")
    pr_info(f"Expected duration: ~{MAX_STEPS * TIME_STEP:.1f}s")
    pr_info("Starting PyBullet simulation...")

    # ============================================
    # 1. Initialize PyBullet
    # ============================================
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(0.6, 180, -41, [0, 0, 0])
    p.setPhysicsEngineParameter(numSolverIterations=150, enableFileCaching=0)
    p.setTimeStep(TIME_STEP)
    p.setGravity(0, 0, 0)

    # ============================================
    # 2. Load models
    # ============================================
    # Target pose
    target_pos = np.array([0.0, 0.0, 0.0])
    target_orn_euler = np.array([0, 0, math.pi])
    target_orn_xyzw = list(transforms3d.euler.euler2quat(
        target_orn_euler[0], target_orn_euler[1], target_orn_euler[2], 'sxyz'
    ))
    target_orn_xyzw = wxyz_to_xyzw(target_orn_xyzw)

    # Initial position (offset from target)
    start_pos = [OFFSET_X, OFFSET_Y, 0.24 + OFFSET_Z]

    pr_info(f"Target position: {target_pos}")
    pr_info(f"Start position: {start_pos}")

    # Load tool URDF
    tool_uid = p.loadURDF(
        format_urdf_filepath("envs/urdf/robotless_lap_joint/tool"),
        basePosition=start_pos,
        baseOrientation=INITIAL_ORN,
        useFixedBase=0,
    )

    # Load target URDF
    target_uid = p.loadURDF(
        format_urdf_filepath("envs/urdf/robotless_lap_joint/task_lap_90deg"),
        basePosition=[0, 0, 0],
        baseOrientation=wxyz_to_xyzw(transforms3d.euler.euler2quat(
            0, 0, math.pi, 'sxyz'
        )),
        useFixedBase=1,
    )

    # Display frames
    display_frame_axis(target_uid, 0, line_length=0.05)
    for li in range(p.getNumJoints(tool_uid)):
        display_frame_axis(tool_uid, li, line_length=0.03)

    p.enableJointForceTorqueSensor(tool_uid, 0)

    # Create constraint for tool control
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

    # Get transformation from base to member link
    base_pose = p.getBasePositionAndOrientation(tool_uid)
    member_pose = p.getLinkState(tool_uid, 2)[4:6]
    from_base_to_member = get_f1_to_f2_xform(base_pose, member_pose)

    # Stabilize simulation
    for _ in range(10):
        p.stepSimulation()
    time.sleep(0.1)

    # ============================================
    # 3. Show debug information
    # ============================================
    p.addUserDebugText(
        "Tool (offset)",
        start_pos,
        [1, 0, 0],
        textSize=1.0,
        lifeTime=0,
    )
    p.addUserDebugText(
        "Target position",
        [0, 0, 0.03],
        [0, 1, 0],
        textSize=1.0,
        lifeTime=0,
    )

    traj_points = []

    pr_info("\nStarting autonomous motion...")
    time.sleep(0.2)

    # ============================================
    # 4. Main control loop
    # ============================================
    step = 0
    success = False
    min_dist = float('inf')

    while step < MAX_STEPS:
        # Get current state
        base_pos, base_orn = p.getBasePositionAndOrientation(tool_uid)

        # Calculate member link position
        member_pose_mat = transform_mat(
            from_base_to_member,
            mat44_by_pos_quat(base_pos, base_orn),
        )
        current_pos, current_orn = mat44_to_pos_quat(member_pose_mat)

        # Position error
        pos_error = np.array(target_pos) - np.array(current_pos)
        dist = np.linalg.norm(pos_error)

        if dist < min_dist:
            min_dist = dist

        traj_points.append(current_pos)

        # Orientation error
        orn_error = quat_error(current_orn, target_orn_xyzw)
        orn_dist = np.linalg.norm(orn_error)

        # Proportional control
        vel_cmd = KP_POS * pos_error
        vel_norm = np.linalg.norm(vel_cmd)
        if vel_norm > MAX_VEL:
            vel_cmd = vel_cmd / vel_norm * MAX_VEL

        rot_cmd = KP_ORN * orn_error
        rot_norm = np.linalg.norm(rot_cmd)
        if rot_norm > MAX_RAD:
            rot_cmd = rot_cmd / rot_norm * MAX_RAD

        # Update position and orientation
        delta_pos = vel_cmd * TIME_STEP
        delta_rot = rot_cmd * TIME_STEP

        new_pos = (np.array(base_pos) + delta_pos).tolist()

        # Update quaternion
        ang_vel_quat = [0, delta_rot[0], delta_rot[1], delta_rot[2]]
        q_delta = wxyz_to_xyzw(
            transforms3d.quaternions.qmult(
                ang_vel_quat, xyzw_to_wxyz(base_orn)
            )
        )
        new_orn = (np.array(base_orn) + 0.5 * np.array(q_delta)).tolist()
        new_orn_norm = np.linalg.norm(new_orn)
        new_orn = (np.array(new_orn) / new_orn_norm).tolist()

        # Apply constraint
        p.changeConstraint(base_constraint, new_pos, new_orn, -1)

        # Step simulation
        p.stepSimulation()
        step += 1

        # Print progress
        if step % 40 == 0:
            pr_info(
                f"Step {step:3d} | "
                f"Distance: {dist*1000:.1f}mm | "
                f"Orientation: {orn_dist*180/math.pi:.1f}deg"
            )

        # Check for success
        if dist < DIST_THRESHOLD and orn_dist < 0.05:
            success = True
            pr_green("\n" + "=" * 60)
            pr_green(f"SUCCESS! Reached target in {step} steps")
            pr_green(f"   Final distance: {dist*1000:.2f}mm")
            pr_green(f"   Final orientation error: {orn_dist*180/math.pi:.2f}deg")
            pr_green("=" * 60)
            break

        # Control loop timing
        time.sleep(TIME_STEP * 0.5)

    # ============================================
    # 5. Show results
    # ============================================
    if not success:
        pr_red("\n" + "=" * 60)
        pr_red(f"Did not reach target within {MAX_STEPS} steps")
        pr_red(f"Closest distance: {min_dist*1000:.2f}mm")
        pr_red("=" * 60)

    # Draw trajectory
    if len(traj_points) > 1:
        for i in range(1, len(traj_points)):
            p.addUserDebugLine(
                traj_points[i-1], traj_points[i],
                [0, 1, 1], lifeTime=0, lineWidth=2,
            )

    pr_info("\nDemo complete. Close window or press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass

    p.disconnect()
    pr_info("PyBullet disconnected.")


if __name__ == "__main__":
    run_demo()
