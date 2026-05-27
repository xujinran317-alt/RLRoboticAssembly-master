"""
Franka Panda 机器人仿真模型（轴孔装配任务）
=========================================

使用 Panda 机器人的逆运动学进行末端位姿控制。
适用于 peg-in-hole（轴孔装配）任务。
"""

import math
from typing import List, Tuple

import numpy as np
import pybullet as p

from assembly_env.robots.base import RobotBase
from utils.io_utils import format_urdf_filepath, display_frame_axis

INITIAL_POS = [0.0, 0.0, 0.0]
INITIAL_ORN = [0, 0, 0, 1]
TARGET_POS = [0.6, 0.044, 0]
TARGET_ORN = [0, 0, 0, 1]
HOLE_OFFSET = [0.048485, -0.04305, -0.03]

URDF_PATH_TOOL = "envs/urdf/panda_peg_in_hole/panda_peg"
URDF_PATH_TARGET = "envs/urdf/panda_peg_in_hole/insertion_box"

# Panda 初始关节角度（度 -> 弧度）
INITIAL_PANDA_JOINTS = [
    math.radians(0),
    math.radians(20),
    math.radians(0),
    math.radians(-103),
    math.radians(0),
    math.radians(122),
    math.radians(45),
]


class RobotSimPanda(RobotBase):
    """Franka Panda 机器人仿真模型"""

    def __init__(self):
        # 加载 Panda 机器人 URDF
        self.uid = p.loadURDF(
            format_urdf_filepath(URDF_PATH_TOOL),
            basePosition=INITIAL_POS,
            baseOrientation=INITIAL_ORN,
            useFixedBase=1,
        )

        # 加载目标工件
        self.target_uid = p.loadURDF(
            format_urdf_filepath(URDF_PATH_TARGET),
            basePosition=TARGET_POS,
            baseOrientation=TARGET_ORN,
            useFixedBase=1,
        )
        self.link_target = 0
        display_frame_axis(self.target_uid, self.link_target)

        self.num_arm_joints = 7

        # 力传感器（最后一个关节）
        self.link_sensor = 6
        display_frame_axis(self.uid, self.link_sensor)

        # 末端执行器连杆
        self.link_ee = 7
        display_frame_axis(self.uid, self.link_ee)

        # 末端连接件
        self.link_member = 8
        display_frame_axis(self.uid, self.link_member)

        self.max_force = 200
        self.max_velocity = 0.35

        self.joint_positions = INITIAL_PANDA_JOINTS.copy()

        # 设置初始关节位置
        for joint_idx in range(self.num_arm_joints):
            p.resetJointState(self.uid, joint_idx, self.joint_positions[joint_idx])
            p.setJointMotorControl2(
                self.uid,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=self.joint_positions[joint_idx],
                force=self.max_force,
                maxVelocity=self.max_velocity,
                positionGain=0.3,
                velocityGain=1,
            )

        # 缓存末端位姿（避免 PyBullet bug 导致不稳定）
        ee_pose = p.getLinkState(self.uid, self.link_ee)
        self.ee_position = list(ee_pose[0])
        self.ee_orientation = list(ee_pose[1])

    def get_member_pose(self) -> Tuple[List[float], List[float]]:
        """获取末端连接件位姿"""
        link_member_pose = p.getLinkState(self.uid, self.link_member)
        return [link_member_pose[0], link_member_pose[1]]

    def get_target_pose(self) -> Tuple[List[float], List[float]]:
        """获取目标位姿（含孔中心偏移）"""
        pose = p.getLinkState(self.target_uid, self.link_target)[4:6]
        return [
            (np.array(pose[0]) + HOLE_OFFSET).tolist(),
            pose[1],
        ]

    def enable_force_torque_sensor(self):
        """启用力/力矩传感器"""
        p.enableJointForceTorqueSensor(self.uid, self.link_sensor)

    def get_force_torque(self) -> List[float]:
        """读取力/力矩传感器"""
        return np.multiply(-0.1, p.getJointState(self.uid, self.link_sensor)[2]).tolist()

    def apply_action_pose(self, delta: np.ndarray):
        """
        应用6D位姿动作（通过逆运动学）
        delta: [dx, dy, dz, drx, dry, drz]
        """
        for i in range(3):
            self.ee_position[i] += delta[i]

        orn_euler = list(p.getEulerFromQuaternion(self.ee_orientation))
        for i in range(3):
            orn_euler[i] += delta[i + 3]
        self.ee_orientation = p.getQuaternionFromEuler(orn_euler)

        # 逆运动学计算关节角度
        joint_positions = p.calculateInverseKinematics(
            self.uid,
            self.num_arm_joints,
            self.ee_position,
            self.ee_orientation,
        )

        for i in range(self.num_arm_joints):
            p.setJointMotorControl2(
                self.uid,
                i,
                p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                targetVelocity=0,
                force=self.max_force,
                maxVelocity=self.max_velocity,
                positionGain=0.3,
                velocityGain=1,
            )

    def apply_action_position(self, delta: np.ndarray):
        """
        应用3D位置动作（保持当前姿态）
        delta: [dx, dy, dz]
        """
        for i in range(3):
            self.ee_position[i] += delta[i]

        joint_positions = p.calculateInverseKinematics(
            self.uid,
            self.num_arm_joints,
            self.ee_position,
            self.ee_orientation,
        )

        for i in range(self.num_arm_joints):
            p.setJointMotorControl2(
                self.uid,
                i,
                p.POSITION_CONTROL,
                targetPosition=joint_positions[i],
                targetVelocity=0,
                force=self.max_force,
                maxVelocity=self.max_velocity,
                positionGain=0.3,
                velocityGain=1,
            )
