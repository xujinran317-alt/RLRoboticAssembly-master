""""
装配仿真环境（PyBullet + gymnasium）
===================================

在 PyBullet 物理引擎中模拟机器人装配任务。
支持域随机化（Domain Randomization）以提高 sim-to-real 迁移能力。

修改说明（相对原版）：
  1. 新增 compute_reward() 重写 → 稠密奖励成形（Dense Reward Shaping）
  2. reset() 中初始化 _prev_dist，支持逐步进度奖励
  3. __init__ 新增 use_shaped_reward 开关，默认开启，可从 YAML 关闭

用法:
    from assembly_env import AssemblySimEnv

    env = AssemblySimEnv(renders=True)
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import time
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pybullet as p

from assembly_env.base_task import AssemblyBaseTask
from assembly_env.robots import RobotSimRobotless
from utils.io_utils import pr_green, pr_red


class AssemblySimEnv(AssemblyBaseTask):
    """
    PyBullet 仿真装配环境

    参数:
        use_shaped_reward: 是否使用稠密奖励（默认 True）
            True  → 距离奖励 + 进度奖励 + 姿态奖励 + 成功奖励
            False → 原始稀疏奖励（保留父类行为，用于对比）

        reward_weights: 各奖励分量权重字典，可覆盖默认值
            dist_scale    : 距离惩罚缩放（默认 100.0）
            progress_scale: 进度奖励缩放（默认 500.0）
            orn_scale     : 姿态惩罚缩放（默认 30.0）
            success_bonus : 成功奖励（默认 100.0）

    其余参数与原版一致，见下方文档。
    """

    def __init__(
        self,
        env_robot=None,
        self_collision_enabled: bool = True,
        renders: bool = True,
        ft_noise: bool = False,
        pose_noise: bool = False,
        action_noise: bool = False,
        physical_noise: bool = False,
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
        # ---- 新增参数 ----
        use_shaped_reward: bool = True,
        reward_weights: Optional[dict] = None,
        curriculum_phase: int = 2,
    ):
        super().__init__(
            time_step=time_step,
            max_steps=max_steps,
            step_limit=step_limit,
            action_dim=action_dim,
            max_vel=max_vel,
            max_rad=max_rad,
            ft_obs_only=ft_obs_only,
            limit_ft=limit_ft,
            max_ft=max_ft,
            max_position_range=max_position_range,
            dist_threshold=dist_threshold,
            render_mode=render_mode,
        )

        self._env_robot = env_robot or RobotSimRobotless
        self._self_collision_enabled = self_collision_enabled
        self._renders = renders

        # ---- 稠密奖励开关 ----
        self.use_shaped_reward = use_shaped_reward

        # 奖励权重（可通过 YAML env_config 传入）
        default_weights = {
            "dist_scale": 100.0,      # 绝对距离惩罚系数
            "progress_scale": 500.0,  # 进步奖励系数（核心信号）
            "orn_scale": 30.0,        # 姿态对齐惩罚系数
            "success_bonus": 100.0,   # 成功额外奖励
        }
        if reward_weights:
            default_weights.update(reward_weights)
        self.rw = default_weights

        # 上一步距离（用于计算进度奖励），在 reset 中初始化
        self._prev_dist: float = 0.0
        
        # 课程学习阶段：0=近距离(1cm), 1=中距离(2cm), 2=正常(3cm)
        self.curriculum_phase = curriculum_phase

        # ---- 域随机化参数 ----
        self._ft_noise = ft_noise
        self._ft_noise_level = [0.5, 0.5, 0.5, 0.05, 0.05, 0.05]
        self._ft_bias_level = [2.0, 2.0, 2.0, 0.2, 0.2, 0.2]
        self._ft_bias = [0.0] * 6
        self._pose_noise = pose_noise
        self._pos_noise_level = 0.001
        self._orn_noise_level = 0.001
        self._action_noise = action_noise
        self._action_noise_lin = 0.001
        self._action_noise_rot = 0.001
        self._physical_noise = physical_noise
        self._friction_noise_level = 0.1

        if self._renders or self.render_mode == "human":
            p.connect(p.GUI)
            p.resetDebugVisualizerCamera(0.6, 180, -41, [0, 0, 0])
        else:
            p.connect(p.DIRECT)

        self.env = None
        self.max_dist = 0.0
        self.np_random = None

    # ============================================================
    # 稠密奖励重写（核心改动）
    # ============================================================

    def compute_reward(self):
        """
        重写父类的稀疏奖励，改为稠密奖励成形。

        设计原则（确保收敛）：
          1. 进步奖励（主导信号）: (prev_dist - curr_dist) * progress_scale
             → 只要比上一步更近就给正奖励，随机探索也能获得正回报
          2. 距离优势（辅助信号）: (max_dist - curr_dist) * dist_scale
             → 离目标越近得分越高，提供额外梯度（初始为~0）
          3. 姿态对齐惩罚       : -orn_dist * orn_scale
             → 鼓励末端姿态对准目标（6DOF 任务特有）
          4. 时间惩罚            : -time_penalty
             → 鼓励高效完成，每步扣一点
          5. 成功奖励            : +success_bonus（一次性）

        收敛保证：
          - 随机策略偶尔靠近目标 → 进步奖励为正 → 总奖励 > 最差情况
          - 好策略持续靠近 → 进步奖励持续为正 → 总奖励显著提升
          - 成功时获得大奖 → 明确的目标导向

        如果 use_shaped_reward=False，直接调用父类原始稀疏奖励。
        """
        if not self.use_shaped_reward:
            reward, terminated, truncated, num_success = super().compute_reward()
            return reward, terminated, truncated, num_success

        # ---- 1. 一次性获取位姿 ----
        member_pos, member_orn = self.get_member_pose()
        target_pos, target_orn = self.get_target_pose()

        member_pos = np.array(member_pos)
        target_pos = np.array(target_pos)
        member_orn = np.array(member_orn)
        target_orn = np.array(target_orn)

        # ---- 2. 位置距离 ----
        curr_dist = float(np.linalg.norm(member_pos - target_pos))

        # ---- 3. 姿态误差 ----
        dot = abs(np.dot(member_orn, target_orn))
        dot = np.clip(dot, 0.0, 1.0)
        orn_dist = float(2.0 * np.arccos(dot))

        # ---- 4. 判断成功/超时 ----
        terminated = curr_dist < self._dist_threshold
        truncated = self._step_limit and self._env_step_counter > self._max_step
        num_success = 1 if terminated else 0

        if terminated:
            pr_green(f"装配成功！步数: {self._env_step_counter}")
        if truncated:
            pr_red(f"装配失败（超时）")

        # ---- 5. 进步奖励（主导信号：随机探索也能获得正回报）----
        progress = self._prev_dist - curr_dist          # 正数 = 靠近了
        self._prev_dist = curr_dist

        # ---- 6. 距离优势（辅助信号：相对于初始距离的优势）----
        # 初始时 dist ≈ max_dist，advantage ≈ 0；靠近目标时 advantage 增大
        advantage = self.max_dist - curr_dist

        # ---- 7. 组合奖励 ----
        reward = (
            + progress   * self.rw["progress_scale"]    # 进步奖励（主导）
            + advantage  * self.rw["dist_scale"]        # 距离优势（辅助）
            - orn_dist   * self.rw["orn_scale"]         # 姿态惩罚
            - self.rw.get("time_penalty", 0.1)           # 时间惩罚
        )

        if terminated:
            reward += self.rw["success_bonus"]          # 成功大奖

        return reward, terminated, truncated, num_success

    # ============================================================
    # reset：初始化 _prev_dist
    # ============================================================

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        重置仿真环境（gymnasium 接口）
        父类 reset() 只做随机种子初始化，返回 None，不解包。
        与原版的唯一差异：末尾新增 self._prev_dist 初始化。
        """
        super().reset(seed=seed)   # 只用于设置 self.np_random，不解包返回值

        # 清理之前的机器人实例
        if hasattr(self, 'env') and self.env is not None:
            try:
                if hasattr(self.env, 'base_constraint'):
                    p.removeConstraint(self.env.base_constraint)
            except Exception:
                pass

        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150, enableFileCaching=0)
        p.setTimeStep(self._time_step)
        p.setGravity(0, 0, 0)

        self.env = self._env_robot()
        
        # ---- 课程学习：根据阶段设置 member 到目标的距离 ----
        # 渐进式课程，每步只增加 0.5cm
        # Phase 0: 1cm, Phase 1: 1.5cm, Phase 2: 2cm, Phase 3: 2.5cm, Phase 4: 3cm
        curriculum_member_heights = {0: 0.01, 1: 0.012, 2: 0.015, 3: 0.02, 4: 0.025, 5: 0.03}
        target_member_height = curriculum_member_heights.get(self.curriculum_phase, 0.03)
        if self.curriculum_phase in curriculum_member_heights:
            import assembly_env.robots.sim_robotless as sr
            # 获取当前 base 和 member 的 z 偏移
            current_base_z = p.getBasePositionAndOrientation(self.env.uid)[0][2]
            current_member_z = self.env.get_member_pose()[0][2]
            base_to_member_z = current_member_z - current_base_z  # 通常是负数（member 在 base 下方）
            # 计算需要的 base z 位置：target_z + target_height - base_to_member_z
            new_base_z = sr.TARGET_POS[2] + target_member_height - base_to_member_z
            new_base_pos = [sr.TARGET_POS[0], sr.TARGET_POS[1], new_base_z]
            # 移动机器人
            p.resetBasePositionAndOrientation(self.env.uid, new_base_pos, sr.INITIAL_ORN)
            p.removeConstraint(self.env.base_constraint)
            self.env.base_constraint = p.createConstraint(
                parentBodyUniqueId=self.env.uid,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=new_base_pos,
                childFrameOrientation=sr.INITIAL_ORN,
            )
        
        self.max_dist = self.dist_to_target()

        self._env_step_counter = 0
        self.env.enable_force_torque_sensor()
        p.stepSimulation()

        self._correlated_noise()

        self._observation = self.get_extended_observation()
        self._add_observation_noise()

        # ---- 新增：记录初始距离，避免第一步进度奖励异常 ----
        self._prev_dist = self.dist_to_target()

        return np.array(self._observation, dtype=np.float32), {"num_success": 0}

    # ============================================================
    # 以下代码与原版完全一致，未作改动
    # ============================================================

    def close(self):
        try:
            p.disconnect()
        except Exception:
            pass

    def get_member_pose(self):
        return self.env.get_member_pose()

    def get_target_pose(self):
        return self.env.get_target_pose()

    def get_force_torque(self):
        return self.env.get_force_torque()

    def _step_impl(self, delta: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # 先执行动作（应用到物理引擎）
        if self.action_dim > 3:
            self.env.apply_action_pose(delta)
        else:
            self.env.apply_action_position(delta)

        p.stepSimulation()
        self._env_step_counter += 1

        if self._renders or self.render_mode == "human":
            time.sleep(self._time_step)

        # 获取观测
        self._observation = self.get_extended_observation()
        self._add_observation_noise()

        # 再算奖励（基于动作执行后的新状态）
        reward, terminated, truncated, num_success = self.compute_reward()

        return (
            np.array(self._observation, dtype=np.float32),
            float(reward),
            bool(terminated),
            bool(truncated),
            {"num_success": int(num_success)},
        )

    # ---- 域随机化（原版，未改动）----

    def _correlated_noise(self):
        if self._ft_noise:
            self._ft_bias = self._add_gaussian_noise(0.0, self._ft_bias_level, [0] * len(self._ft_bias_level))
        if self._physical_noise:
            self._add_all_friction_noise(self._friction_noise_level)

    def _add_observation_noise(self):
        if not self._ft_obs_only:
            if self.action_dim > 3:
                pos, orn = self._uncorrelated_pose_noise(
                    self._observation[0:3], self._observation[3:7]
                )
                self._observation[0:3] = pos
                self._observation[3:7] = orn
                self._observation[7:13] = self._add_ft_noise(self._observation[7:13])
            else:
                self._observation[0:3] = self._uncorrelated_position_noise(self._observation[0:3])
                self._observation[3:9] = self._add_ft_noise(self._observation[3:9])
        else:
            self._observation[0:6] = self._add_ft_noise(self._observation[0:6])

    def _uncorrelated_pose_noise(self, pos, orn):
        if self._pose_noise:
            pos = self._add_gaussian_noise(0.0, self._pos_noise_level, pos)
            orn_euler = p.getEulerFromQuaternion(orn)
            orn_euler = self._add_gaussian_noise(0.0, self._orn_noise_level, orn_euler)
            orn = p.getQuaternionFromEuler(orn_euler)
        return pos, orn

    def _uncorrelated_position_noise(self, pos):
        if self._pose_noise:
            pos = self._add_gaussian_noise(0.0, self._pos_noise_level, pos)
        return pos

    def _add_ft_noise(self, force_torque):
        if self._ft_noise:
            force_torque = self._add_gaussian_noise(0.0, self._ft_noise_level, force_torque)
            force_torque = (np.array(force_torque) + np.array(self._ft_bias)).tolist()
        return force_torque

    def _add_all_friction_noise(self, noise_level: float):
        self._add_body_friction_noise(self.env.target_uid, self.env.link_target, noise_level)
        self._add_body_friction_noise(self.env.uid, self.env.link_member, noise_level)

    @staticmethod
    def _add_body_friction_noise(uid: int, link: int, noise_level: float):
        dynamics = p.getDynamicsInfo(uid, link)
        noise_range = np.fabs(dynamics[1]) * noise_level
        friction_noise = np.random.normal(0, noise_range)
        p.changeDynamics(uid, link, lateralFriction=dynamics[1] + friction_noise)

    @staticmethod
    def _add_gaussian_noise(mean: float, std: list, vec: list) -> list:
        noise = np.random.normal(mean, std, np.shape(vec))
        return (np.array(vec) + noise).tolist()

