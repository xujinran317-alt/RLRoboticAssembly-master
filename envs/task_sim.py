"""
task_sim.py - PyBullet 仿真环境实现

在 PyBullet 物理引擎中模拟机器人装配任务。
支持：
  - 域随机化（Domain Randomization）：给传感器/物理参数加噪声，提高 sim-to-real 迁移能力
  - 力/力矩传感器模拟
  - 碰撞检测
  - 可视化渲染
"""

import pybullet as p
import time

import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding

from envs.task import Task


class TaskSim(Task):
    """仿真装配环境（继承 Task 基类）"""

    def __init__(self,
                 env_robot=None,               # 机器人类（Robotless / Panda）
                 self_collision_enabled=None,  # 是否检测自碰撞
                 renders=None,                 # 是否显示可视化窗口
                 ft_noise=None,                # 是否在力/力矩观测上加噪声
                 pose_noise=None,              # 是否在位置观测上加噪声
                 action_noise=None,            # 是否在动作上加噪声
                 physical_noise=None,          # 是否给物理参数（如摩擦）加噪声
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

        super().__init__(max_steps=max_steps,
                         action_dim=action_dim,
                         step_limit=step_limit,
                         max_vel=max_vel,
                         max_rad=max_rad,
                         ft_obs_only=ft_obs_only,
                         limit_ft=limit_ft,
                         time_step=time_step,
                         max_ft=max_ft,
                         max_position_range=max_position_range,
                         dist_threshold=dist_threshold)

        self._env_robot = env_robot
        self._self_collision_enabled = self_collision_enabled
        self._renders = renders

        # ---- 域随机化参数 ----
        self._ft_noise = ft_noise
        self._ft_noise_level = [0.5, 0.5, 0.5, 0.05, 0.05, 0.05]  # 力/力矩噪声标准差 (N / Nm)
        self._ft_bias_level = [2.0, 2.0, 2.0, 0.2, 0.2, 0.2]      # 力/力矩偏置范围 (N / Nm)
        self._ft_bias = 0.0
        self._pose_noise = pose_noise
        self._pos_noise_level = 0.001    # 位置噪声（米）
        self._orn_noise_level = 0.001    # 姿态噪声（弧度）
        self._action_noise = action_noise
        self._action_noise_lin = 0.001   # 动作噪声（线速度倍率）
        self._action_noise_rot = 0.001   # 动作噪声（角速度倍率）
        self._physical_noise = physical_noise
        self._friction_noise_level = 0.1  # 摩擦系数的噪声比例

        # 连接 PyBullet（GUI 模式显示窗口，DIRECT 模式不显示）
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)  # 有图形界面
            p.resetDebugVisualizerCamera(0.6, 180, -41, [0, 0, 0])
        else:
            p.connect(p.DIRECT)  # 无头模式

        self.seed()

    def reset(self):
        """重置仿真环境，开始新的一局"""
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150, enableFileCaching=0)
        p.setTimeStep(self._time_step)
        p.setGravity(0, 0, 0)  # 微重力装配，重力设为0

        # 创建机器人环境（加载 URDF 模型）
        self.env = self._env_robot()
        self.max_dist = self.dist_to_target()

        self._env_step_counter = 0
        self.env.enable_force_torque_sensor()  # 开启力传感器
        p.stepSimulation()

        # 施加每个 episode 固定的噪声（传感器偏置、摩擦系数等）
        self.correlated_noise()

        self._observation = self.get_extended_observation()
        self.add_observation_noise()  # 给观测值加噪声

        return np.array(self._observation)

    def __del__(self):
        """析构时断开 PyBullet"""
        p.disconnect()

    def seed(self, seed=None):
        """设置随机种子"""
        return [seed]  # gymnasium compatibility: seeding handled by reset(seed=)

    def get_member_pose(self):
        """获取末端执行器的位姿"""
        return self.env.get_member_pose()

    def get_target_pose(self):
        """获取目标装配位姿"""
        return self.env.get_target_pose()

    def get_force_torque(self):
        """获取力/力矩传感器读数"""
        return self.env.get_force_torque()

    def step2(self, delta):
        """
        执行一步仿真
        - 计算奖励、判断是否结束
        - 将动作（位移/旋转量）应用到机器人
        - 推进 PyBullet 物理引擎
        """
        reward, done, num_success = self.reward()

        # 如果没结束，执行动作；如果结束了，发零动作（停止）
        if not done:
            if self.action_dim > 3:
                self.env.apply_action_pose(delta)
            else:
                self.env.apply_action_position(delta)
        else:
            if self.action_dim > 3:
                self.env.apply_action_pose([0.0] * 6)
            else:
                self.env.apply_action_position([0.0] * 3)

        p.stepSimulation()
        self._env_step_counter += 1

        if self._renders:
            time.sleep(self._time_step)  # 控制渲染速度

        self._observation = self.get_extended_observation()
        self.add_observation_noise()

        return np.array(self._observation), reward, done, {"num_success": num_success}

    # ---- 域随机化函数 ----

    def correlated_noise(self):
        """
        每个 episode 开始时施加的"相关性噪声"
        - 力/力矩传感器偏置：整个 episode 保持不变
        - 物理噪声：摩擦系数随机变化
        """
        if self._ft_noise:
            self._ft_bias = self.add_gaussian_noise(0.0, self._ft_bias_level, [0] * len(self._ft_bias_level))

        if self._physical_noise:
            self.add_all_friction_noise(self._friction_noise_level)

    def uncorrelated_pose_noise(self, pos, orn):
        """每个 step 给位置和姿态加高斯噪声"""
        if self._pose_noise:
            pos = self.add_gaussian_noise(0.0, self._pos_noise_level, pos)
            orn_euler = p.getEulerFromQuaternion(orn)
            orn_euler = self.add_gaussian_noise(0.0, self._orn_noise_level, orn_euler)
            orn = p.getQuaternionFromEuler(orn_euler)
        return pos, orn

    def uncorrelated_position_noise(self, pos):
        """每个 step 给位置加高斯噪声"""
        if self._pose_noise:
            pos = self.add_gaussian_noise(0.0, self._pos_noise_level, pos)
        return pos

    def add_ft_noise(self, force_torque):
        """给力/力矩观测加高斯噪声 + 传感器偏置"""
        if self._ft_noise:
            force_torque = self.add_gaussian_noise(0.0, self._ft_noise_level, force_torque)
            force_torque = np.add(force_torque, self._ft_bias).tolist()
        return force_torque

    def add_observation_noise(self):
        """给完整的观测向量加噪声"""
        if not self._ft_obs_only:
            if self.action_dim > 3:
                self._observation[0:3], self._observation[3:7] = self.uncorrelated_pose_noise(
                    self._observation[0:3], self._observation[3:7])
                self._observation[7:13] = self.add_ft_noise(self._observation[7:13])
            else:
                self._observation[0:3] = self.uncorrelated_position_noise(self._observation[0:3])
                self._observation[3:9] = self.add_ft_noise(self._observation[3:9])
        else:
            self._observation[0:6] = self.add_ft_noise(self._observation[0:6])

    def add_all_friction_noise(self, noise_level):
        """给所有相关物体添加随机摩擦系数"""
        self.add_body_friction_noise(self.env.target_uid, self.env.link_target, noise_level)
        self.add_body_friction_noise(self.env.uid, self.env.link_member, noise_level)

    @staticmethod
    def add_body_friction_noise(uid, link, noise_level):
        """给单个物体的单个连杆添加随机摩擦系数"""
        dynamics = p.getDynamicsInfo(uid, link)
        noise_range = np.fabs(dynamics[1]) * noise_level
        friction_noise = np.random.normal(0, noise_range)
        p.changeDynamics(uid, link, lateralFriction=dynamics[1] + friction_noise)

    @staticmethod
    def add_gaussian_noise(mean, std, vec):
        """对向量每个元素加高斯噪声: noise ~ N(mean, std)"""
        noise = np.random.normal(mean, std, np.shape(vec))
        return np.add(vec, noise).tolist()
