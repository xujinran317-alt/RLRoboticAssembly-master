"""
使用 Stable-Baselines3 训练强化学习智能体
==========================================

用法:
    assembly train -a sac --total-timesteps 1000000
    assembly train -f configs/assembly_sac.yaml
    assembly train -f configs/assembly_ddpg.yaml
    assembly train -f configs/assembly_ppo.yaml
"""

import os
import pickle
import warnings
from typing import Optional
from collections import deque

import numpy as np
import yaml

# Suppress gym NumPy 2.x compatibility warning (SB3 internal dependency)
warnings.filterwarnings("ignore", message="Gym has been unmaintained", category=UserWarning)

from stable_baselines3 import SAC, TD3, PPO, DDPG
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import (
    OrnsteinUhlenbeckActionNoise,
    NormalActionNoise,
)

from assembly_env.sim_task import AssemblySimEnv
from utils.io_utils import pr_green, pr_red

# 算法映射表
ALGO_MAP = {
    "sac": SAC,
    "td3": TD3,
    "ppo": PPO,
    "ddpg": DDPG,
}

# On-policy 算法（不支持 replay buffer 相关参数）
ON_POLICY_ALGOS = {"ppo", "a2c"}

# Off-policy 专有参数，on-policy 算法需过滤
OFF_POLICY_ONLY_PARAMS = {
    "buffer_size",
    "learning_starts",
    "train_freq",
    "gradient_steps",
    "tau",
    "replay_buffer_class",
    "replay_buffer_kwargs",
    "optimize_memory_usage",
}


def _warmup_replay_buffer(model, demo_file: str, action_dim: int = 6, verbose: bool = True) -> int:
    """
    用演示数据预热 off-policy 模型的 replay buffer。

    从 pickle 文件加载演示 transitions，直接注入到 model.replay_buffer。
    pickle 文件格式：list of deque，每个 deque 包含 (obs, action, reward, next_obs, done) 元组。

    仅对 off-policy 算法（SAC/DDPG/TD3）有效。
    返回注入的 transition 总数。
    """
    if not hasattr(model, "replay_buffer"):
        pr_red(f"[Warmup] 模型 {type(model).__name__} 没有 replay_buffer，跳过热身")
        return 0

    if not os.path.exists(demo_file):
        pr_red(f"[Warmup] 演示文件不存在: {demo_file}，跳过热身")
        return 0

    with open(demo_file, "rb") as f:
        all_episodes = pickle.load(f)

    total_added = 0
    buffer = model.replay_buffer
    # 获取当前填充位置
    pos = buffer.pos

    for ep_idx, episode in enumerate(all_episodes):
        episode_added = 0
        for obs, action, reward, next_obs, done in episode:
            # 确保维度正确
            if isinstance(obs, np.ndarray) and obs.ndim == 1:
                obs = obs.reshape(1, -1)
            if isinstance(next_obs, np.ndarray) and next_obs.ndim == 1:
                next_obs = next_obs.reshape(1, -1)
            if isinstance(action, np.ndarray) and action.ndim == 1:
                action = action.reshape(1, -1)

            # SB3 replay buffer add() 参数: obs, next_obs, action, reward, done, infos
            buffer.add(
                obs=obs,
                next_obs=next_obs,
                action=action,
                reward=np.array([reward], dtype=np.float32),
                done=np.array([done], dtype=np.float32),
                infos=[{"num_success": int(done and reward > 50)}],
            )
            episode_added += 1
            total_added += 1

        if verbose:
            pr_green(f"  [Warmup] Episode {ep_idx+1}/{len(all_episodes)}: {episode_added} transitions")

    # 更新 buffer 位置（SB3 ReplayBuffer 内部维护 pos，不用手动设置）
    if verbose:
        pr_green(f"[Warmup] 共注入 {total_added} 条演示 transitions 到 replay buffer")

    return total_added


def _create_env(render: bool = False, **env_kwargs):
    """创建包装好的环境"""
    env = AssemblySimEnv(renders=render, **env_kwargs)
    return Monitor(env)


def _resolve_action_noise(noise_config: dict, n_actions: int):
    """
    将 YAML 中的 action_noise 配置解析为 SB3 噪声对象。

    YAML 格式示例:
        action_noise:
          type: ornstein_uhlenbeck   # 或 normal
          sigma: 0.1
    """
    noise_type = noise_config.get("type", "ornstein_uhlenbeck").lower()
    sigma = noise_config.get("sigma", 0.1)
    mean = np.zeros(n_actions)
    sigma_arr = float(sigma) * np.ones(n_actions)

    if noise_type == "ornstein_uhlenbeck":
        return OrnsteinUhlenbeckActionNoise(mean=mean, sigma=sigma_arr)
    elif noise_type == "normal":
        return NormalActionNoise(mean=mean, sigma=sigma_arr)
    else:
        raise ValueError(f"未知的 action_noise 类型: {noise_type}，可选: ornstein_uhlenbeck, normal")


def train_agent(
    config_file: Optional[str] = None,
    algo: str = "sac",
    total_timesteps: int = 1_000_000,
    experiment_name: str = "assembly_experiment",
    log_dir: str = "./sb3_logs",
    device: str = "auto",
    render: bool = False,
):
    """
    训练 RL 智能体

    Args:
        config_file: YAML 配置文件（可选，会覆盖其他参数）
        algo: 算法名称 (sac, td3, ppo, ddpg)
        total_timesteps: 总训练步数
        experiment_name: 实验名称
        log_dir: 日志目录
        device: 训练设备
        render: 是否显示环境窗口
    """
    # ----- 加载配置 -----
    config = {}

    if config_file:
        with open(config_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        pr_green(f"[Config] 已加载配置文件: {config_file}")

        algo = config.get("algo", algo)
        total_timesteps = config.get("total_timesteps", total_timesteps)
        experiment_name = config.get("name", experiment_name)
        log_dir = config.get("log_dir", log_dir)
        render = config.get("render", render)

    # ----- 创建环境 -----
    env_kwargs = config.get("env_config", {})
    if "renders" in env_kwargs:
        del env_kwargs["renders"]

    env = _create_env(render=render, **env_kwargs)
    log_path = os.path.join(log_dir, experiment_name)
    os.makedirs(log_path, exist_ok=True)

    # ----- 选择算法 -----
    algo_lower = algo.lower()
    algo_cls = ALGO_MAP.get(algo_lower)
    if algo_cls is None:
        raise ValueError(f"未知算法: {algo}，可选: {', '.join(ALGO_MAP.keys())}")

    pr_green(f"开始训练: {algo.upper()}")
    pr_green(f"   总步数: {total_timesteps}")
    pr_green(f"   日志目录: {log_path}")
    pr_green(f"   设备: {device}")

    # ----- 处理模型超参数 -----
    model_kwargs = config.get("model_config", {}).copy()

    # 1) activation_fn 字符串 -> torch.nn 函数引用
    if "policy_kwargs" in model_kwargs:
        pk = model_kwargs["policy_kwargs"]
        if "activation_fn" in pk:
            act_map = {
                "relu": "ReLU",
                "tanh": "Tanh",
                "sigmoid": "Sigmoid",
                "elu": "ELU",
            }
            act_name = pk["activation_fn"]
            if isinstance(act_name, str):
                import torch.nn as nn
                pk["activation_fn"] = getattr(nn, act_map.get(act_name.lower(), "ReLU"))

    # 2) on-policy 算法过滤掉 off-policy 专有参数，避免 TypeError
    if algo_lower in ON_POLICY_ALGOS:
        removed = [k for k in OFF_POLICY_ONLY_PARAMS if k in model_kwargs]
        for k in removed:
            model_kwargs.pop(k)
        if removed:
            pr_green(f"[PPO] 已自动过滤 off-policy 参数: {removed}")

    # 3) action_noise：DDPG / TD3 需要，从 YAML 解析为 SB3 对象
    if "action_noise" in model_kwargs:
        noise_cfg = model_kwargs.pop("action_noise")
        if isinstance(noise_cfg, dict):
            n_actions = env.action_space.shape[0]
            model_kwargs["action_noise"] = _resolve_action_noise(noise_cfg, n_actions)
            pr_green(f"[{algo.upper()}] action_noise 已解析: {noise_cfg.get('type')}  sigma={noise_cfg.get('sigma')}")

    # ----- TensorBoard（可选）-----
    try:
        import tensorboard  # noqa: F401
        tb_log = log_path
    except ImportError:
        tb_log = None
        pr_green("(未安装 tensorboard，跳过日志记录)")

    # ----- 创建模型 -----
    model = algo_cls(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=tb_log,
        device=device,
        **model_kwargs,
    )

    # ----- 演示数据预热 replay buffer（仅 off-policy 算法）-----
    demo_buffer_file = config.get("demo_buffer_file", "")
    if demo_buffer_file and algo_lower not in ON_POLICY_ALGOS:
        pr_green(f"[Warmup] 使用演示数据预热 replay buffer: {demo_buffer_file}")
        _warmup_replay_buffer(
            model,
            demo_file=demo_buffer_file,
            action_dim=env_kwargs.get("action_dim", 6),
            verbose=True,
        )
    elif demo_buffer_file and algo_lower in ON_POLICY_ALGOS:
        pr_red(f"[Warmup] {algo.upper()} 是 on-policy 算法，不支持 replay buffer 预热")
    else:
        pr_green("(未指定 demo_buffer_file，跳过 replay buffer 预热)")

    # ----- 回调函数 -----
    callbacks = []

    success_threshold = config.get("success_threshold", 0.9)
    eval_env = _create_env(render=False, **env_kwargs)

    stop_callback = StopTrainingOnRewardThreshold(
        reward_threshold=success_threshold, verbose=1
    )
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=config.get("eval_freq", 10000),
        best_model_save_path=log_path,
        verbose=1,
    )
    callbacks.append(eval_callback)

    # ----- 开始训练 -----
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=experiment_name,
        )
        pr_green("训练完成！")

    except KeyboardInterrupt:
        pr_red("\n训练被用户中断")

    finally:
        final_path = os.path.join(log_path, "final_model")
        model.save(final_path)
        pr_green(f"模型已保存: {final_path}.zip")

        env.close()
        eval_env.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="训练 RL 智能体")
    parser.add_argument("-f", "--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument("-a", "--algo", type=str, default="sac", help="算法名称 (sac, td3, ppo, ddpg)")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000, help="总训练步数")
    parser.add_argument("--name", type=str, default="assembly_experiment", help="实验名称")
    parser.add_argument("--log-dir", type=str, default="./sb3_logs", help="日志目录")
    parser.add_argument("--device", type=str, default="auto", help="训练设备 (auto/cpu/cuda)")
    parser.add_argument("--render", action="store_true", default=False, help="是否显示环境窗口")
    args = parser.parse_args()

    train_agent(
        config_file=args.config,
        algo=args.algo,
        total_timesteps=args.total_timesteps,
        experiment_name=args.name,
        log_dir=args.log_dir,
        device=args.device,
        render=args.render,
    )
