"""
机器人接口抽象基类

所有机器人模型（仿真/真实）必须实现以下接口：
  - get_member_pose()      -> (position, orientation)  末端位姿
  - get_target_pose()      -> (position, orientation)  目标位姿
  - get_force_torque()     -> [Fx, Fy, Fz, Tx, Ty, Tz] 力/力矩
  - enable_force_torque_sensor()  启用力传感器
  - apply_action_pose(delta)      应用6D位姿动作
  - apply_action_position(delta)  应用3D位置动作
"""

from abc import ABC, abstractmethod
from typing import List, Tuple


class RobotBase(ABC):
    """机器人基类"""

    @abstractmethod
    def get_member_pose(self) -> Tuple[List[float], List[float]]:
        """
        获取末端执行器位姿
        Returns:
            (position: [x, y, z], orientation: [qx, qy, qz, qw])  四元数 xyzw 格式
        """
        ...

    @abstractmethod
    def get_target_pose(self) -> Tuple[List[float], List[float]]:
        """
        获取目标装配位姿
        Returns:
            (position: [x, y, z], orientation: [qx, qy, qz, qw])  四元数 xyzw 格式
        """
        ...

    @abstractmethod
    def get_force_torque(self) -> List[float]:
        """
        获取力/力矩传感器读数
        Returns:
            [Fx, Fy, Fz, Tx, Ty, Tz]
        """
        ...

    def enable_force_torque_sensor(self):
        """启用力/力矩传感器（仿真环境需要，真实机器人可能不需要）"""
        pass

    @abstractmethod
    def apply_action_pose(self, delta: List[float]):
        """
        应用6D位姿动作（平移+旋转）
        Args:
            delta: [dx, dy, dz, drx, dry, drz]
        """
        ...

    @abstractmethod
    def apply_action_position(self, delta: List[float]):
        """
        应用3D位置动作（纯平移）
        Args:
            delta: [dx, dy, dz]
        """
        ...
