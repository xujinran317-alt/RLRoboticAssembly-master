"""
task.py - 装配任务的抽象基类

这个文件定义了强化学习环境的通用框架：
  - 继承 gym.Env，符合 OpenAI Gym 接口规范
  - 定义了观测空间（位置+力觉）和动作空间（3/6自由度）
  - 提供了奖励函数、距离计算、力/力矩约束等通用逻辑
  - 子类（TaskSim / TaskReal）需要实现 reset() 等具体细节
"""

import math
from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
from gymnasium import spaces

import utilities as util

WRITE_CSV = False  # 是否把数据写入 CSV 文件（调试用）


class Task(ABC, gym.Env):
    """装配任务的抽象基类"""

    def __init__(self,
                 time_step=None,
                 max_steps=None,
                 step_limit=None,
                 action_dim=None,
                 max_vel=None,
                 max_rad=None,
                 ft_obs_only=None,
                 limit_ft=None,
                 max_ft=None,
                 max_position_range=None,
                 dist_threshold=None):
        super().__init__()

        self._max_step = max_steps if max_steps is not None else 200
        self._step_limit = step_limit if step_limit is not None else True
        self._max_vel = max_vel if max_vel is not None else 0.01
        self._max_rad = max_rad if max_rad is not None else 0.01
        self._ft_obs_only = ft_obs_only if ft_obs_only is not None else False

        self._time_step = time_step if time_step is not None else 1/250
        self._observation = []
        self._env_step_counter = 0
        self._num_success = 0

        self._limit_force_torque = limit_ft if limit_ft is not None else False
        self._max_force_torque = max_ft if max_ft is not None else [1000, 1000, 2500, 100, 100, 100]
        self._force_torque_violations = [0.0] * len(self._max_force_torque)
        self._ft_range_ratio = 1

        # ---- 定义观测空间（Observation Space） ----
        self._max_pos_range = max_position_range if max_position_range is not None else [2]*3
        if self._ft_obs_only:
            self.observation_dim = len(self._max_force_torque)
            observation_high = np.array(self._max_force_torque, dtype=np.float32)
            observation_low = -observation_high
        elif action_dim == 6:
            self.observation_dim = 7 + len(self._max_force_torque)
            observation_orn_high = [1] * 4
            observation_high = np.array(self._max_pos_range + observation_orn_high + self._max_force_torque, dtype=np.float32)
            observation_low = -observation_high
        else:
            self.observation_dim = 3 + len(self._max_force_torque)
            observation_high = np.array(self._max_pos_range + self._max_force_torque, dtype=np.float32)
            observation_low = -observation_high
        self.observation_space = spaces.Box(observation_low, observation_high, dtype=np.float32)

        # ---- 定义动作空间（Action Space） ----
        self._action_bound = 1
        action_high = np.array([self._action_bound] * action_dim, dtype=np.float32)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        self.action_dim = action_dim

        self.member_pose = []     # 末端执行器位姿
        self.force_torque = []    # 力/力矩读数

        self.dist_threshold = dist_threshold

        # CSV 表头（调试用）
        if WRITE_CSV:
            util.write_csv(["step_member_pose", "pos_X", "pos_Y", "pos_Z", "qX", "qY", "qZ", "qW"], 'member_pose.csv', True)
            util.write_csv(["step_ft", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"], 'ft_reading.csv', True)
            if self.action_dim == 3:
                util.write_csv(["step_actions", "vel_X", "vel_Y", "vel_Z"], 'data_out.csv', True)
            else:
                util.write_csv(["step_actions", "vel_X", "vel_Y", "vel_Z", "rot_vel_X", "rot_vel_Y", "rot_vel_Z"],
                               'data_out.csv', True)

    @abstractmethod
    def reset(self):
        """重置环境（子类必须实现）"""
        pass

    def pos_dist_to_target(self):
        """计算末端位置到目标位置的距离（米）"""
        self.member_pose = self.get_member_pose()

        if WRITE_CSV:
            util.write_csv([self._env_step_counter] + self.member_pose[0] + self.member_pose[1], 'member_pose.csv', False)

        member_pos = list(self.member_pose[0])
        target_pose = self.get_target_pose()
        target_pos = list(target_pose[0])

        dist_pos = np.linalg.norm(np.subtract(member_pos, target_pos))
        return dist_pos

    def orn_dist_to_target(self):
        """计算末端姿态到目标姿态的角度差（弧度）"""
        member_orn = list(self.member_pose[1])
        target_pose = self.get_target_pose()
        target_orn = list(target_pose[1])

        # 四元数夹角公式：cos(θ/2) = |q1·q2|，相差 π 时 θ 最大
        dist_orn = math.fabs(2 * math.acos(math.fabs(np.dot(member_orn, target_orn))) - math.pi)
        return dist_orn

    def dist_to_target(self):
        """计算末端到目标的综合距离（位置距离 + 姿态距离加权）"""
        dist_pos = self.pos_dist_to_target()
        dist = dist_pos

        if self.action_dim > 3:
            dist_orn = self.orn_dist_to_target()
            dist = dist_pos + 0.05 * dist_orn  # 姿态误差的权重较小

        return dist

    def get_extended_observation(self):
        """
        组装观测向量
        观测包含：位置(xyz) + 姿态(xyzw) + 力/力矩(Fx,Fy,Fz,Tx,Ty,Tz)
        或者只用力/力矩（取决于 ft_obs_only）
        """
        self._observation = []

        if not self._ft_obs_only:
            if self.action_dim > 3:
                pos, orn = self.member_pose[0], self.member_pose[1]
                self._observation.extend(pos)
                self._observation.extend(orn)
            else:
                pos = self.member_pose[0]
                self._observation.extend(pos)

        self.force_torque = self.get_force_torque()

        if WRITE_CSV:
            util.write_csv([self._env_step_counter] + self.force_torque, 'ft_reading.csv', False)

        if self._limit_force_torque:
            self.check_ft_limit(self.force_torque)

        self._observation.extend(self.force_torque)
        return self._observation

    def step(self, action):
        """
        执行动作（Gym 标准接口）
        - 接收 [-1, 1] 范围的动作值
        - 转换为实际的位移/旋转量：delta = action * max_vel * time_step
        - 如果启用了力/力矩限制，会在超限方向减速
        """
        if len(action) > 3:
            # 6维动作：前3维线速度，后3维角速度
            delta_lin = np.array(action[0:3]) * self._max_vel * self._time_step
            delta_rot = np.array(action[3:6]) * self._max_rad * self._time_step
            delta = np.append(delta_lin, delta_rot)
        else:
            # 3维动作：只有线速度
            delta = np.array(action) * self._max_vel * self._time_step

        if self._limit_force_torque:
            self.constrain_velocity_for_ft(delta)

        if WRITE_CSV:
            util.write_csv([self._env_step_counter] + list(delta), 'data_out.csv', False)

        return self.step2(delta)

    def step2(self, delta):
        """子类可重写此方法来实现不同的步进逻辑"""
        self._env_step_counter += 1
        reward, done, num_success = self.reward()
        self._observation = self.get_extended_observation()
        return np.array(self._observation), reward, done, {"num_success": num_success}

    def reward(self):
        """
        奖励函数设计：
        - 负距离奖励：离目标越近，奖励越大（= -dist）
        - 到达奖励：距离 < 阈值时 +1000
        - 超时惩罚：超过最大步数则失败
        """
        done = False

        dist = self.dist_to_target()
        reward_dist = - dist  # 距离越近，负值越小

        reward_ft = 0
        reward = reward_dist + 0.05 * reward_ft

        if dist < self.dist_threshold:
            done = True
            reward += 1000  # 成功完成装配，巨额奖励
            util.prGreen("装配成功！步数：" + str(self._env_step_counter))
            self._num_success = 1

        if self._step_limit and self._env_step_counter > self._max_step:
            done = True
            util.prRed("装配失败（超时）")
            self._num_success = 0

        return reward, done, self._num_success

    # ---- 力/力矩相关的奖励和约束 ----

    def reward_force_torque(self):
        """
        力/力矩额外的奖励惩罚：
        - 当力/力矩超过50%极限时，按超限比例惩罚
        - 如果启用了限力模式，超限时额外 -5 惩罚
        """
        ft_contact_limit = np.multiply(self._max_force_torque, 0.5)
        indices = Task.check_list_bounds(self.force_torque, ft_contact_limit)

        max_ft = 0
        for i in range(len(indices)):
            ft_excess_ratio = (indices[i] * self.force_torque[i] - ft_contact_limit[i]) / self._max_force_torque[i]
            if (indices[i] != 0) & (max_ft < ft_excess_ratio):
                max_ft = ft_excess_ratio
        reward_ft = - max_ft

        if self._limit_force_torque & (self._force_torque_violations != [0.0] * len(self._force_torque_violations)):
            reward_ft -= 5

        return reward_ft

    @staticmethod
    def check_list_bounds(l, l_bounds):
        """
        检查列表中的每个值是否超出边界
        返回格式：[0, 0, 1, 0, -1, 0] 表示第3个方向正超限、第5个方向负超限
        """
        assert len(l) == len(l_bounds)
        index_list = [0] * len(l)
        for i in range(len(l)):
            if math.fabs(l[i]) >= l_bounds[i]:
                index_list[i] = np.sign(l[i])
        return index_list

    def check_ft_limit(self, force_torque):
        """检查力/力矩是否超过安全限制"""
        self._force_torque_violations = Task.check_list_bounds(
            force_torque,
            np.multiply(self._ft_range_ratio, self._max_force_torque)
        )

    def constrain_velocity_for_ft(self, velocity):
        """
        根据力/力矩超限情况来约束速度：
        - 如果某方向力已超限，且还在往该方向运动，则反向微动
        - 如果有扭矩超限，则停止所有平动（只允许沿未超限轴转动）
        """
        force_list = self._force_torque_violations[0:3]
        torque_list = self._force_torque_violations[3:6]
        lin_vel = velocity[0:3]

        for i in range(len(force_list)):
            if (force_list[i] != 0) & (np.sign(force_list[i]) != np.sign(lin_vel[i])):
                lin_vel[i] = -0.1 * lin_vel[i]  # 反向微动释放压力

        for i in range(len(torque_list)):
            if torque_list[i] != 0:
                for j in range(len(torque_list)):
                    if j != i:
                        lin_vel[j] = 0.0  # 扭矩超限时锁住其他方向

        velocity[0:3] = lin_vel
        return velocity
