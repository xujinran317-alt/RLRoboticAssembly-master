"""
robot_real_example.py - 真实机器人接口示例

这个文件展示如何连接真实的机器人。
实际使用时需要替换 ControlInterface 和 FTInterface 为你自己机器人的通信库。

接口约定：
  - 接收：机器人返回的位姿数据（7维：位置xyz + 四元数xyzw）
  - 发送：动作指令（6维：位置偏移xyz + 姿态偏移rpy + done标志）
  - 力传感器：返回6维力/力矩数据
"""

import utilities as util

# 实际使用时需要导入你自己的机器人控制库和力传感器库
# from robot_control_interface import ControlInterface
# from ft_sensor_interface import FTInterface


class RobotRealExample():
    """真实机器人控制接口示例"""

    def __init__(self):
        # 连接机器人控制和力传感器接口
        self.robot_interface = ControlInterface()  # 替换为你自己的机器人接口
        self.ft_interface = FTInterface()          # 替换为你自己的力传感器接口

    @staticmethod
    def decompose_incoming_pose_data(data):
        """解析机器人返回的位姿数据：前3个是位置，后4个是四元数"""
        position = data[:3]
        rotation = data[3:7]
        return [position, rotation]

    def get_member_pose(self):
        """从真实机器人读取当前末端位姿"""
        self.robot_interface.receive()
        data_in = self.robot_interface.message_in.values
        values = self.decompose_incoming_pose_data(data_in)
        position_m = values[0]
        rotation_quat = values[1]
        return [position_m, rotation_quat]

    @staticmethod
    def get_target_pose():
        """目标位姿：世界坐标系原点，无旋转"""
        return [0, 0, 0], util.xyzw_by_euler([0, 0, 0], 'sxyz')

    def get_force_torque(self):
        """从真实力传感器读取6维力/力矩数据"""
        self.ft_interface.receive()
        data_in = self.ft_interface.message_in.values
        force_torque = data_in
        return force_torque

    def apply_action_pose(self, delta, done):
        """
        发送6维位姿动作到真实机器人
        delta: [dx, dy, dz, drx, dry, drz]
        done: 0=任务进行中, 1=任务结束（让机器人安全停机）
        """
        relative_pos = delta[0:3]
        relative_orn = delta[3:6]
        data_out = list(relative_pos) + list(relative_orn) + [done]
        self.robot_interface.send(data_out)

    def apply_action_position(self, delta, done):
        """
        仅发送3维位置动作（保持当前姿态）
        """
        data_out = list(delta) + [0, 0, 0] + [done]
        self.robot_interface.send(data_out)
