"""
装配任务基类（gymnasium.Env 接口）
==================================

与 gymnasium 兼容的装配任务环境基类。
所有具体环境（仿真/真实）继承此类。

观测空间：
  - 3D 动作: [pos_x, pos_y, pos_z, Fx, Fy, Fz, Tx, Ty, Tz]  共 9 维
  - 6D 动作: [pos_x, pos_y, pos_z, qx, qy, qz, qw, Fx, Fy, Fz, Tx, Ty, Tz]  共 13 维
  - ft_only: [Fx, Fy, Fz, Tx, Ty, Tz]  共 6 维

动作空间：
  - 3D: [-1, 1]^3  （线速度比例）
  - 6D: [-1, 1]^6  （线速度 + 角速度比例）

奖励函数：
  - 负距离奖励：r = -dist
  - 成功奖励：距离 < 阈值时 +1000
  - 超时惩罚：超过最大步数结束
"""

import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union, Dict, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from utils.io_utils import pr_green, pr_red, write_csv

# 调试用：是否将数据写入 CSV
WRITE_CSV = False


class AssemblyBaseTask(gym.Env, ABC):
    """
    装配任务基类

    参数:
        time_step: 控制周期（秒）
        max_steps: 每个 episode 最大步数
        step_limit: 是否启用步数上限
        action_dim: 动作维度（3=纯平移，6=平移+旋转）
        max_vel: 最大线速度（m/s）
        max_rad: 最大角速度（rad/s）
        ft_obs_only: 观测是否只含力/力矩
        limit_ft: 是否限制力/力矩
        max_ft: 最大允许力/力矩 [Fx,Fy,Fz,Tx,Ty,Tz]
        max_position_range: 观测中位置的范围（米）
        dist_threshold: 距离小于此值算成功
        render_mode: gymnasium 渲染模式（"human" 或 None）
    """

    metadata = {"render_modes": ["human"], "render_fps": 250}

    def __init__(
        self,
        time_step: float = 1 / 250,
        max_steps: int = 200,
        step_limit: bool = True,
        action_dim: int = 6,
        max_vel: float = 0.01,
        max_rad: float = 0.01,
        ft_obs_only: bool = False,
        limit_ft: bool = False,
        max_ft: Optional[list] = None,
        max_position_range: Optional[list] = None,
        dist_threshold: float = 0.005,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        # 基础参数
        self._time_step = time_step
        self._max_step = max_steps
        self._step_limit = step_limit
        self._max_vel = max_vel
        self._max_rad = max_rad
        self._ft_obs_only = ft_obs_only
        self._dist_threshold = dist_threshold

        # 状态变量
        self._observation: list = []
        self._env_step_counter = 0
        self._num_success = 0
        self.member_pose: list = []
        self.force_torque: list = []

        # 力/力矩限制
        self._limit_force_torque = limit_ft
        self._max_force_torque = max_ft if max_ft else [1000, 1000, 2500, 100, 100, 100]
        self._force_torque_violations = [0.0] * len(self._max_force_torque)
        self._ft_range_ratio = 1
        self._max_pos_range = max_position_range if max_position_range else [2] * 3

        self.render_mode = render_mode

        # ---- 定义观测空间 ----
        if self._ft_obs_only:
            self.observation_dim = len(self._max_force_torque)
            obs_high = np.array(self._max_force_torque, dtype=np.float32)
            obs_low = -obs_high
        elif action_dim == 6:
            # 位置(3) + 四元数(4) + 力/力矩(6)
            self.observation_dim = 13
            obs_high = np.array(
                self._max_pos_range + [1.0] * 4 + self._max_force_torque,
                dtype=np.float32,
            )
            obs_low = -obs_high
        else:
            # 位置(3) + 力/力矩(6)
            self.observation_dim = 9
            obs_high = np.array(
                self._max_pos_range + self._max_force_torque, dtype=np.float32
            )
            obs_low = -obs_high

        self.observation_space = spaces.Box(
            low=obs_low, high=obs_high, dtype=np.float32
        )

        # ---- 定义动作空间 ----
        self._action_bound = 1.0
        self.action_dim = action_dim
        self.action_space = spaces.Box(
            low=-self._action_bound,
            high=self._action_bound,
            shape=(action_dim,),
            dtype=np.float32,
        )

        # CSV 表头（调试用）
        if WRITE_CSV:
            write_csv(
                ["step", "member_pose", "pos_X", "pos_Y", "pos_Z", "qX", "qY", "qZ", "qW"],
                "member_pose.csv",
                True,
            )
            write_csv(
                ["step", "ft", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
                "ft_reading.csv",
                True,
            )
            if self.action_dim == 3:
                write_csv(
                    ["step", "actions", "vel_X", "vel_Y", "vel_Z"],
                    "data_out.csv",
                    True,
                )
            else:
                write_csv(
                    [
                        "step", "actions", "vel_X", "vel_Y", "vel_Z",
                        "rot_vel_X", "rot_vel_Y", "rot_vel_Z",
                    ],
                    "data_out.csv",
                    True,
                )

    # ============================================================
    # 环境信息获取（子类需实现）
    # ============================================================

    @abstractmethod
    def get_member_pose(self):
        """获取末端执行器位姿"""
        ...

    @abstractmethod
    def get_target_pose(self):
        """获取目标位姿"""
        ...

    @abstractmethod
    def get_force_torque(self):
        """获取力/力矩读数"""
        ...

    # ============================================================
    # gymnasium 标准接口
    # ============================================================

    @abstractmethod
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置环境（gymnasium 接口）
        返回 (observation, info)
        """
        ...

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        执行动作（gymnasium 标准接口）
        返回 (observation, reward, terminated, truncated, info)

        gymnasium 与旧 gym 的区别：
        - terminated: 是否因成功/失败结束
        - truncated: 是否因超时/截断结束
        """
        # 将 [-1, 1] 的动作映射为实际位移
        if len(action) > 3:
            delta_lin = np.array(action[0:3]) * self._max_vel * self._time_step
            delta_rot = np.array(action[3:6]) * self._max_rad * self._time_step
            delta = np.concatenate([delta_lin, delta_rot])
        else:
            delta = np.array(action) * self._max_vel * self._time_step

        if self._limit_force_torque:
            delta = self._constrain_velocity_for_ft(delta)

        if WRITE_CSV:
            write_csv([self._env_step_counter] + list(delta), "data_out.csv", False)

        return self._step_impl(delta)

    @abstractmethod
    def _step_impl(self, delta: np.ndarray):
        """子类实现的具体步进逻辑"""
        ...

    def render(self):
        """渲染（默认无操作，仿真环境在 PyBullet 中渲染）"""
        pass

    def close(self):
        """关闭环境（默认无操作）"""
        pass

    # ============================================================
    # 距离计算
    # ============================================================

    def pos_dist_to_target(self) -> float:
        """计算末端位置到目标位置的距离（米）"""
        self.member_pose = self.get_member_pose()

        if WRITE_CSV:
            write_csv(
                [self._env_step_counter] + self.member_pose[0] + self.member_pose[1],
                "member_pose.csv",
                False,
            )

        member_pos = np.array(self.member_pose[0])
        target_pose = self.get_target_pose()
        target_pos = np.array(target_pose[0])
        return float(np.linalg.norm(member_pos - target_pos))

    def orn_dist_to_target(self) -> float:
        """计算末端姿态到目标姿态的角度差（弧度）"""
        member_orn = np.array(self.member_pose[1])
        target_pose = self.get_target_pose()
        target_orn = np.array(target_pose[1])

        # 四元数夹角：cos(θ/2) = |q1·q2|
        dot = np.abs(np.dot(member_orn, target_orn))
        dot = np.clip(dot, -1.0, 1.0)
        dist_orn = math.fabs(2 * math.acos(dot) - math.pi)
        return dist_orn

    def dist_to_target(self) -> float:
        """综合距离（位置距离 + 姿态距离加权）"""
        dist_pos = self.pos_dist_to_target()
        dist = dist_pos

        if self.action_dim > 3:
            dist_orn = self.orn_dist_to_target()
            dist = dist_pos + 0.05 * dist_orn

        return dist

    # ============================================================
    # 观测构建
    # ============================================================

    def get_extended_observation(self) -> list:
        """组装观测向量（先获取最新位姿）"""
        self._observation = []

        # 获取最新末端位姿（覆盖 member_pose 缓存）
        self.member_pose = self.get_member_pose()

        if not self._ft_obs_only:
            if self.action_dim > 3:
                pos, orn = self.member_pose[0], self.member_pose[1]
                self._observation.extend(pos)
                self._observation.extend(orn)
            else:
                self._observation.extend(self.member_pose[0])

        self.force_torque = self.get_force_torque()

        if WRITE_CSV:
            write_csv(
                [self._env_step_counter] + self.force_torque,
                "ft_reading.csv",
                False,
            )

        if self._limit_force_torque:
            self.check_ft_limit(self.force_torque)

        self._observation.extend(self.force_torque)
        return self._observation

    # ============================================================
    # 奖励函数
    # ============================================================

    def compute_reward(self) -> Tuple[float, bool, bool, int]:
        """
        计算奖励和 done 信号

        Returns:
            reward: 奖励值
            terminated: 是否因成功结束
            truncated: 是否因超时截断
            num_success: 0 或 1
        """
        terminated = False
        truncated = False

        dist = self.dist_to_target()
        reward = -dist  # 负距离奖励

        if dist < self._dist_threshold:
            terminated = True
            reward += 1000.0
            pr_green(f"装配成功！步数: {self._env_step_counter}")
            self._num_success = 1

        if self._step_limit and self._env_step_counter > self._max_step:
            truncated = True
            pr_red(f"装配失败（超时）")
            self._num_success = 0

        return reward, terminated, truncated, self._num_success

    # ============================================================
    # 力/力矩相关
    # ============================================================

    @staticmethod
    def check_list_bounds(values: list, bounds: list) -> list:
        """
        检查每个值是否超出边界
        返回: [0, 0, 1, 0, -1, 0] 等（符号表示超限方向）
        """
        assert len(values) == len(bounds)
        result = [0] * len(values)
        for i in range(len(values)):
            if math.fabs(values[i]) >= bounds[i]:
                result[i] = int(np.sign(values[i]))
        return result

    def check_ft_limit(self, force_torque: list):
        """检查力/力矩是否超过安全限制"""
        self._force_torque_violations = self.check_list_bounds(
            force_torque,
            np.multiply(self._ft_range_ratio, self._max_force_torque),
        )

    def _constrain_velocity_for_ft(self, velocity: np.ndarray) -> np.ndarray:
        """
        根据力/力矩超限情况约束速度
        - 力超限且往该方向运动 => 反向微动
        - 扭矩超限 => 锁住其他方向的平动
        """
        velocity = velocity.copy()
        force_list = self._force_torque_violations[0:3]
        torque_list = self._force_torque_violations[3:6]
        lin_vel = velocity[0:3]

        for i in range(3):
            if force_list[i] != 0 and np.sign(force_list[i]) != np.sign(lin_vel[i]):
                lin_vel[i] = -0.1 * lin_vel[i]

        for i in range(3):
            if torque_list[i] != 0:
                for j in range(3):
                    if j != i:
                        lin_vel[j] = 0.0

        velocity[0:3] = lin_vel
        return velocity
