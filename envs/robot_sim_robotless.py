"""
robot_sim_robotless.py - 无机器人的仿真模型（搭接关节任务）

这个文件模拟一个"没有机器人的末端执行器"，用于搭接关节（lap joint）装配任务。
末端执行器直接悬浮在空间中，通过约束来控制位置。
这样可以绕过具体的机器人运动学，直接测试装配策略。
"""

import math
import pybullet as p
import numpy as np
import transforms3d
import utilities as util

# ---- 搭接关节任务的初始/目标位姿 ----
INITIAL_POS = np.array([0.0, 0.0, 0.03])      # 初始位置：高出目标 3cm（简化初始化）
INITIAL_ORN = util.mat33_to_quat(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))  # 初始姿态：朝正下
TARGET_POS = np.array([0, 0, 0])               # 目标位置：原点
TARGET_ORN = np.array([0, 0, math.pi])         # 目标姿态：绕Z轴旋转180度

# URDF 模型文件路径
URDF_PATH_TOOL = 'envs/urdf/robotless_lap_joint/tool'
URDF_PATH_TARGET = 'envs/urdf/robotless_lap_joint/task_lap_90deg'


class RobotSimRobotless():
    """无机器人模型：直接用约束控制末端位姿"""

    def __init__(self):
        # 加载末端工具的 URDF（可移动物体）
        self.uid = p.loadURDF(util.format_urdf_filepath(URDF_PATH_TOOL),
                              basePosition=INITIAL_POS,
                              baseOrientation=INITIAL_ORN,
                              useFixedBase=0)  # 不固定基座，可自由移动
        self.link_member = 2    # 末端连接件的连杆索引
        self.link_gripper = 1   # 夹爪的连杆索引
        self.link_sensor = 0    # 力传感器的连杆索引

        # 加载目标工件的 URDF（固定物体）
        self.target_uid = p.loadURDF(util.format_urdf_filepath(URDF_PATH_TARGET),
                                     basePosition=TARGET_POS,
                                     baseOrientation=util.xyzw_by_euler(TARGET_ORN, 'sxyz'),
                                     useFixedBase=1)  # 固定基座
        self.link_target = 0
        util.display_frame_axis(self.target_uid, self.link_target)

        self.max_force = -1  # -1 表示不限制约束力

        # 创建固定约束：将末端执行器"钉"在当前位姿
        # 之后通过改变约束的 target 位姿来移动
        self.base_constraint = p.createConstraint(
            parentBodyUniqueId=self.uid,
            parentLinkIndex=-1,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=INITIAL_POS,
            childFrameOrientation=INITIAL_ORN
        )

        # 计算从基座到末端连接件的相对变换（用于获取末端精确位姿）
        self.base_pose = p.getBasePositionAndOrientation(self.uid)
        self.gripper_pose = p.getLinkState(self.uid, self.link_gripper)[4:6]
        self.member_pose = p.getLinkState(self.uid, self.link_member)[4:6]
        self.from_base_to_member = util.get_f1_to_f2_xform(self.base_pose, self.member_pose)

        # 显示所有关节的坐标系
        for link_index in range(p.getNumJoints(self.uid)):
            util.display_frame_axis(self.uid, link_index)

    def get_member_pose(self):
        """
        获取末端连接件的位姿
        由于基座位姿变化后需要重新计算连杆位姿，这里做了坐标变换
        """
        base_pose = p.getBasePositionAndOrientation(self.uid)
        member_pose_mat = util.transform_mat(
            self.from_base_to_member,
            util.mat44_by_pos_quat(base_pose[0], base_pose[1])
        )
        member_pose = util.mat44_to_pos_quat(member_pose_mat)
        return [member_pose[0], member_pose[1]]

    def get_target_pose(self):
        """获取目标工件的位姿"""
        return p.getLinkState(self.target_uid, self.link_target)[4:6]

    def enable_force_torque_sensor(self):
        """启用力/力矩传感器（在 PyBullet 中是在质心位置测量）"""
        p.enableJointForceTorqueSensor(self.uid, self.link_sensor)

    def get_force_torque(self):
        """
        读取力/力矩传感器数据
        注意：PyBullet 的 FT 读数方向是反的，需要取负并缩放
        """
        return np.multiply(-0.1, p.getJointState(self.uid, self.link_sensor)[2]).tolist()

    def apply_action_pose(self, delta_pose):
        """
        应用位姿动作（平移+旋转）
        通过修改约束的 target 位姿来移动末端
        注意 PyBullet 的奇特行为：命令速度的 5 倍才是实际速度
        """
        delta_pose = np.multiply(np.array(delta_pose), 5.0)

        relative_pos = np.array(delta_pose[0:3])
        base_pos, base_orn = p.getBasePositionAndOrientation(self.uid)
        new_pos = np.add(np.array(base_pos), relative_pos).tolist()

        # 四元数旋转更新：Δq = 0.5 * ω * q
        ang_vel_quat = [0, delta_pose[3], delta_pose[4], delta_pose[5]]
        new_orn = np.add(base_orn, np.multiply(0.5,
                  util.wxyz_to_xyzw(transforms3d.quaternions.qmult(
                      ang_vel_quat, util.xyzw_to_wxyz(base_orn)))))

        p.changeConstraint(self.base_constraint, new_pos, new_orn, self.max_force)

    def apply_action_position(self, delta_pos):
        """
        仅应用位置动作（纯平移，保持初始姿态）
        """
        delta_pos = np.multiply(np.array(delta_pos), 5.0)

        base_pos, base_orn = p.getBasePositionAndOrientation(self.uid)
        new_pos = np.add(np.array(base_pos), delta_pos).tolist()

        p.changeConstraint(self.base_constraint, new_pos, INITIAL_ORN, self.max_force)
