"""
人工演示数据采集器
==================

用于采集人类操作员的演示数据（obs, action, reward, next_obs, done），
这些数据可以用于：
  - 模仿学习（Behavioral Cloning）
  - 与强化学习结合的示范回放（Demo Replay）
  - 训练初始策略
"""

import pickle
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

from utils.io_utils import pr_green


class DemoRecorder:
    """
    演示数据采集器

    用法:
        recorder = DemoRecorder()
        recorder.start_episode(obs)
        recorder.record_step(obs, action, reward, next_obs, done)
        recorder.end_episode()
        recorder.save("my_demo_data.pkl")
    """

    def __init__(self):
        self._memory = deque()
        self._episode_data = []
        self._total_steps = 0

    def start_episode(self, initial_obs: np.ndarray):
        """开始一个新的 episode"""
        self._episode_data = [initial_obs]

    def record_step(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        """
        记录一个 transition

        Args:
            obs: 当前观测
            action: 执行的动作
            reward: 获得的奖励
            next_obs: 下一个观测
            done: 是否结束
        """
        # 只记录非零动作的帧（排除静止帧）
        if np.count_nonzero(action) > 0:
            self._memory.append((obs, action, reward, next_obs, done))
            self._total_steps += 1

    def save(self, file_path: str = "human_demo_data/default"):
        """
        保存所有演示数据到文件

        Args:
            file_path: 保存路径
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "wb") as f:
            pickle.dump(self._memory, f)

        pr_green(f"演示数据已保存: {file_path}")
        pr_green(f"   总步数: {self._total_steps}")
