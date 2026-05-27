"""
加载训练好的模型进行回放/部署
==============================

用法:
    assembly play path/to/model.zip --episodes 10
"""

import pickle
from typing import Optional

from stable_baselines3 import SAC, TD3, PPO, DDPG

from assembly_env.sim_task import AssemblySimEnv
from utils.io_utils import pr_green, pr_red

ALGO_MAP = {
    "sac": SAC,
    "td3": TD3,
    "ppo": PPO,
    "ddpg": DDPG,
}


def play_agent(
    model_path: str,
    algo: str = "SAC",
    num_steps: int = 10000,
    num_episodes: int = 5,
    render: bool = True,
    out_file: Optional[str] = None,
):
    """
    加载训练好的模型并回放

    Args:
        model_path: 模型文件路径 (.zip)
        algo: 算法名称
        num_steps: 最大步数
        num_episodes: 最大 episode 数
        render: 是否渲染
        out_file: 如果指定，保存 rollout 数据到此文件
    """
    # 加载模型
    algo_cls = ALGO_MAP.get(algo.lower())
    if algo_cls is None:
        raise ValueError(f"未知算法: {algo}，可选: {', '.join(ALGO_MAP.keys())}")

    pr_green(f"加载模型: {model_path}")
    model = algo_cls.load(model_path)
    print(f"   策略: {algo.upper()}")
    print(f"   观测空间: {model.observation_space}")
    print(f"   动作空间: {model.action_space}")

    # 创建环境
    env = AssemblySimEnv(renders=render, action_dim=model.action_space.shape[0])

    # 回放
    rollouts = [] if out_file else None

    episodes = 0
    steps = 0
    total_reward = 0.0
    success_count = 0

    while steps < num_steps and episodes < num_episodes:
        if rollouts is not None:
            episode_data = []

        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0.0

        while not (terminated or truncated) and steps < num_steps:
            # 使用模型预测动作
            action, _ = model.predict(obs, deterministic=True)

            # 执行动作
            new_obs, reward, terminated, truncated, info = env.step(action)

            if rollouts is not None:
                episode_data.append([obs, action, new_obs, reward, terminated])

            episode_reward += reward
            steps += 1
            obs = new_obs

        success = info.get("num_success", 0)
        success_count += success
        total_reward += episode_reward

        print(
            f"  Episode {episodes + 1}: 奖励={episode_reward:.2f}, "
            f"步数={steps}, 成功={success}"
        )

        if rollouts is not None:
            rollouts.append(episode_data)

        episodes += 1

    # 输出统计
    avg_reward = total_reward / max(episodes, 1)
    success_rate = success_count / max(episodes, 1)
    print()
    pr_green(f"回放统计")
    print(f"   Episode 数: {episodes}")
    print(f"   总步数: {steps}")
    print(f"   平均奖励: {avg_reward:.2f}")
    print(f"   成功率: {success_rate:.1%}")

    # 保存 rollout 数据
    if out_file and rollouts:
        with open(out_file, "wb") as f:
            pickle.dump(rollouts, f)
        pr_green(f"Rollout 数据已保存: {out_file}")

    env.close()

