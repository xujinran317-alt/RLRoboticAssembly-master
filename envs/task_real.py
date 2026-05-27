"""
task_real.py - 真实机器人环境

这个文件是连接真实机器人的模板环境。
它继承 Task 基类，但实际控制委托给 RobotRealExample（或其替代类）。
相比仿真版本，它不需要 PyBullet，而是通过自定义通信接口控制真实机器人。
"""

from abc import ABC, abstractmethod

from devices.callbacks import handle, validate
import numpy as np

from envs.task import Task


class UpdatePattern(ABC):
    """更新模式抽象类"""
    def __init__(self):
        self._on_update = None  # 更新成功的回调函数

    def update(self, *args, **kwargs):
        """
        公开的更新方法
        1. 调用子类实现的 _update()
        2. 执行回调函数
        """
        result = self._update(*args, **kwargs)
        handle(self._on_update, self)
        return result

    @abstractmethod
    def _update(self, *args, **kwargs):
        """子类实现的数据更新逻辑"""
        pass

    @property
    def on_update(self):
        """获取更新回调"""
        return self._on_update

    @on_update.setter
    def on_update(self, func):
        """设置更新回调"""
        validate(func, allow_args=True, allow_return=True)
        self._on_update = func


class TaskReal(Task):
    """真实机器人装配环境（继承 Task 基类）"""

    def __init__(self,
                 env_robot=None,     # 真实机器人类
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

        # 创建真实机器人接口实例
        self.env = env_robot()

    def reset(self):
        """重置环境（比如让机器人回到初始位置）"""
        self.max_dist = self.dist_to_target()
        self._env_step_counter = 0
        self._observation = self.get_extended_observation()
        return np.array(self._observation)

    def get_member_pose(self):
        """从真实机器人读取末端位姿"""
        return self.env.get_member_pose()

    def get_target_pose(self):
        """目标位姿（通常是固定值）"""
        return self.env.get_target_pose()

    def get_force_torque(self):
        """从力传感器读取读数"""
        return self.env.get_force_torque()

    def step2(self, delta):
        """
        执行一步动作
        区别与仿真版：
        - 不需要调用 PyBullet
        - 需要额外传递 done 标志（用于告诉机器人安全停机）
        """
        reward, done, num_success = self.reward()

        if done:
            # 任务结束，发送零动作让机器人停下
            if self.action_dim > 3:
                last_delta = [0.0] * 6
                self.env.apply_action_pose(last_delta, 1)  # 1=任务结束
            else:
                last_delta = [0.0] * 3
                self.env.apply_action_position(last_delta, 1)
        else:
            # 执行正常动作
            if self.action_dim > 3:
                self.env.apply_action_pose(delta, 0)  # 0=任务进行中
            else:
                self.env.apply_action_position(delta, 0)

        self._env_step_counter += 1
        self._observation = self.get_extended_observation()

        return np.array(self._observation), reward, done, {"num_success": num_success}
