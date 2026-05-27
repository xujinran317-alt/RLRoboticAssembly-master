"""
无机器人仿真模型（搭接关节任务）
================================

模拟"没有机器人的末端执行器"——直接用 PyBullet 约束控制末端位姿。
用于搭接关节（lap joint）装配任务，绕过具体机器人运动学，直接测试装配策略。
"""

import math
from typing import List, Tuple

import numpy as np
import pybullet as p
import transforms3d

from assembly_env.robots.base import RobotBase
from utils.transforms import (
    mat33_to_quat,
    xyzw_by_euler,
    get_f1_to_f2_xform,
    transform_mat,
    mat44_by_pos_quat,
    mat44_to_pos_quat,
    wxyz_to_xyzw,
    xyzw_to_wxyz,
)
from utils.io_utils import format_urdf_filepath, display_frame_axis

# ---- 搭接关节任务初始/目标位姿 ----
# 初始位置设在目标正上方仅 3 厘米处，姿态完全对齐目标
INITIAL_POS = np.array([0.0, 0.0, 0.03])
INITIAL_ORN = xyzw_by_euler([0, 0, math.pi], "sxyz")  # 与 TARGET 姿态完全一致
TARGET_POS = np.array([0, 0, 0])
TARGET_ORN = np.array([0, 0, math.pi])

URDF_PATH_TOOL = "envs/urdf/robotless_lap_joint/tool"
URDF_PATH_TARGET = "envs/urdf/robotless_lap_joint/task_lap_90deg"


class RobotSimRobotless(RobotBase):
    """无机器人模型：直接用约束控制末端位姿"""

    def __init__(self):
        # 加载末端工具 URDF（可移动）
        self.uid = p.loadURDF(
            format_urdf_filepath(URDF_PATH_TOOL),
            basePosition=INITIAL_POS,
            baseOrientation=INITIAL_ORN,
            useFixedBase=0,
        )
        self.link_member = 2
        self.link_gripper = 1
        self.link_sensor = 0

        # 加载目标工件 URDF（固定）
        self.target_uid = p.loadURDF(
            format_urdf_filepath(URDF_PATH_TARGET),
            basePosition=TARGET_POS,
            baseOrientation=xyzw_by_euler(TARGET_ORN, "sxyz"),
            useFixedBase=1,
        )
        self.link_target = 0
        display_frame_axis(self.target_uid, self.link_target)

        self.max_force = -1

        # 创建固定约束
        self.base_constraint = p.createConstraint(
            parentBodyUniqueId=self.uid,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=INITIAL_POS,
            childFrameOrientation=INITIAL_ORN,
        )

        # 计算从基座到连接件的相对变换
        base_pose = p.getBasePositionAndOrientation(self.uid)
        member_pose = p.getLinkState(self.uid, self.link_member)[4:6]
        self._from_base_to_member = get_f1_to_f2_xform(base_pose, member_pose)

        # 显示所有关节坐标系
        for link_index in range(p.getNumJoints(self.uid)):
            display_frame_axis(self.uid, link_index)

    def get_member_pose(self) -> Tuple[List[float], List[float]]:
        """获取末端连接件位姿"""
        base_pose = p.getBasePositionAndOrientation(self.uid)
        member_pose_mat = transform_mat(
            self._from_base_to_member,
            mat44_by_pos_quat(base_pose[0], base_pose[1]),
        )
        member_pose = mat44_to_pos_quat(member_pose_mat)
        return [member_pose[0], member_pose[1]]

    def get_target_pose(self) -> Tuple[List[float], List[float]]:
        """获取目标工件位姿"""
        return p.getLinkState(self.target_uid, self.link_target)[4:6]

    def enable_force_torque_sensor(self):
        """启用力/力矩传感器"""
        p.enableJointForceTorqueSensor(self.uid, self.link_sensor)

    def get_force_torque(self) -> List[float]:
        """读取力/力矩传感器数据"""
        return np.multiply(-0.1, p.getJointState(self.uid, self.link_sensor)[2]).tolist()

    def apply_action_pose(self, delta: np.ndarray):
        """
        应用位姿动作（平移+旋转）
        delta: [dx, dy, dz, drx, dry, drz]
        """
        # PyBullet 速度补偿因子
        delta = np.array(delta) * 5.0

        base_pos, base_orn = p.getBasePositionAndOrientation(self.uid)
        new_pos = (np.array(base_pos) + delta[0:3]).tolist()

        # 四元数更新: Δq = 0.5 * ω * q
        ang_vel_quat = [0, delta[3], delta[4], delta[5]]
        new_orn = np.add(
            base_orn,
            0.5 * np.array(wxyz_to_xyzw(
                transforms3d.quaternions.qmult(
                    ang_vel_quat, xyzw_to_wxyz(base_orn)
                )
            )),
        ).tolist()

        p.changeConstraint(self.base_constraint, new_pos, new_orn, self.max_force)

    def apply_action_position(self, delta: np.ndarray):
        """
        仅应用位置动作（纯平移）
        delta: [dx, dy, dz]
        """
        delta = np.array(delta) * 5.0

        base_pos, _ = p.getBasePositionAndOrientation(self.uid)
        new_pos = (np.array(base_pos) + delta).tolist()

        p.changeConstraint(self.base_constraint, new_pos, INITIAL_ORN, self.max_force)
