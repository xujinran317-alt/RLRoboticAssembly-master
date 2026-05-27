"""
demo_insert-1.py
纯平移装配演示：工具保持初始姿态不变，直接平移插入下方工件（任务工件姿态与工具对齐）。
"""

import time
import math
import numpy as np
import pybullet as p
import transforms3d

from assembly_env.robots.sim_robotless import INITIAL_ORN
from utils.transforms import wxyz_to_xyzw
from utils.io_utils import pr_green, pr_red, pr_info, format_urdf_filepath

# ── 参数 ────────────────────────────────────────────────
TIME_STEP = 1 / 250
MAX_STEPS = 10000

KP_POS  = 3.0
MAX_VEL = 0.08

START_POS  = [0.06, 0.04, 0.24]
TARGET_POS = [0.0,  0.0,  0.0 ]

DONE_DIST = 0.005
# ────────────────────────────────────────────────────────


def run():
    p.connect(p.GUI)
    p.resetDebugVisualizerCamera(0.6, 180, -41, [0, 0, 0])
    p.setPhysicsEngineParameter(numSolverIterations=150, enableFileCaching=0)
    p.setTimeStep(TIME_STEP)
    p.setGravity(0, 0, 0)

    tool_uid = p.loadURDF(
        format_urdf_filepath("envs/urdf/robotless_lap_joint/tool"),
        basePosition=START_POS,
        baseOrientation=INITIAL_ORN,
        useFixedBase=0,
    )
    p.loadURDF(
        format_urdf_filepath("envs/urdf/robotless_lap_joint/task_lap_90deg"),
        basePosition=[0, 0, 0],
        baseOrientation=wxyz_to_xyzw(transforms3d.euler.euler2quat(0, 0, 0, 'sxyz')),
        useFixedBase=1,
    )

    # 约束锚定在世界坐标系中，初始位置为 START_POS
    constraint = p.createConstraint(
        parentBodyUniqueId=tool_uid, parentLinkIndex=-1,
        childBodyUniqueId=-1,       childLinkIndex=-1,
        jointType=p.JOINT_FIXED,    jointAxis=[0, 0, 0],
        parentFramePosition=[0, 0, 0],
        childFramePosition=START_POS,
        childFrameOrientation=INITIAL_ORN,
    )

    for _ in range(10):
        p.stepSimulation()

    pr_info("开始平移插入...")

    for step in range(1, MAX_STEPS + 1):
        pos, orn = p.getBasePositionAndOrientation(tool_uid)

        pe   = np.array(TARGET_POS) - np.array(pos)
        dist = np.linalg.norm(pe)

        vc = KP_POS * pe
        vn = np.linalg.norm(vc)
        if vn > MAX_VEL:
            vc = vc / vn * MAX_VEL

        new_pos = (np.array(pos) + vc * TIME_STEP).tolist()

        # 姿态锁死，直接传回原 orn 不做任何修改
        p.changeConstraint(constraint, new_pos, orn, maxForce=-1)
        p.stepSimulation()
        time.sleep(TIME_STEP)

        if step % 100 == 0:
            pr_info(f"步 {step:4d} | 距离 {dist*1000:.1f}mm")

        if dist < DONE_DIST:
            pr_green(f"✅ 到达目标！步数 {step}（{step*TIME_STEP:.1f}s），误差 {dist*1000:.2f}mm")
            break
    else:
        pr_red(f"❌ {MAX_STEPS} 步未完成，最终距离 {dist*1000:.1f}mm")

    pr_info("按 Ctrl+C 退出")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    p.disconnect()


if __name__ == "__main__":
    run()
