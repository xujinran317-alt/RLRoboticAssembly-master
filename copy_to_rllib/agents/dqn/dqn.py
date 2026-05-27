# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from ray.rllib.agents.trainer import with_common_config
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.agents.dqn.dqn_policy import DQNTFPolicy
from ray.rllib.agents.dqn.simple_q_policy import SimpleQPolicy
from ray.rllib.optimizers import SyncReplayOptimizer
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.utils.schedules import ConstantSchedule, LinearSchedule

logger = logging.getLogger(__name__)

# yapf: disable
# __sphinx_doc_begin__
# ==================== DQN 默认配置 ====================
DEFAULT_CONFIG = with_common_config({
    # === 模型相关 (Model) ===
    # 用于表示回报分布的原子数（atoms）。当该值大于 1 时，使用分布式的 Q-learning。
    # 离散支持域由 v_min 和 v_max 限定
    "num_atoms": 1,
    "v_min": -10.0,
    "v_max": 10.0,
    # 是否使用带噪网络（Noisy Network）
    "noisy": False,
    # 控制带噪网络的初始值
    "sigma0": 0.5,
    # 是否使用 Dueling DQN 架构
    "dueling": True,
    # 是否使用 Double DQN
    "double_q": True,
    # 在模型输出后，使用这些隐藏层处理状态值和动作值。
    # 另请参阅 catalog.py 中的模型配置。
    "hiddens": [256],
    # N步 Q-learning（N-Step Q learning）
    "n_step": 1,

    # === 探索策略 (Exploration) ===
    # 用于退火（annealing）调度的最大时间步数。探索率在此时间步数内
    # 从 1.0 线性退火到 exploration_fraction，并按 exploration_fraction 缩放
    "schedule_max_timesteps": 100000,
    # 每次训练调用所需的最小环境步数。该值不影响学习，仅影响迭代的长度。
    "timesteps_per_iteration": 1000,
    # 在整个训练过程中，用于退火探索率的比例
    "exploration_fraction": 0.1,
    # 随机动作概率的最终值
    "exploration_final_eps": 0.02,
    # 每隔 `target_network_update_freq` 步更新一次目标网络
    "target_network_update_freq": 500,
    # 是否使用 softmax 进行动作采样。离线策略估计时需要。
    "soft_q": False,
    # Softmax 温度参数。Q 值在传入 softmax 之前会除以该值。
    # 当温度趋近于零时，softmax 趋近于 argmax。
    "softmax_temp": 1.0,
    # 如果为 True，则使用参数空间噪声进行探索
    # 详见 https://blog.openai.com/better-exploration-with-parameter-noise/
    "parameter_noise": False,
    # 禁用探索的额外配置（用于评估）
    "evaluation_config": {
        "exploration_fraction": 0,
        "exploration_final_eps": 0,
    },

    # === 经验回放缓冲区 (Replay buffer) ===
    # 回放缓冲区的大小。注意，如果设置了 async_updates，
    # 则每个 worker 都会有该大小的回放缓冲区。
    "buffer_size": 50000,
    # 如果为 True，则使用优先级回放缓冲区（Prioritized Replay Buffer）
    "prioritized_replay": True,
    # 优先级回放缓冲区的 Alpha 参数
    "prioritized_replay_alpha": 0.6,
    # 从优先级回放缓冲区采样的 Beta 参数
    "prioritized_replay_beta": 0.4,
    # 在整个训练过程中用于退火 Beta 参数的比例
    "beta_annealing_fraction": 0.2,
    # Beta 参数的最终值
    "final_prioritized_replay_beta": 0.4,
    # 更新优先级时添加到 TD 误差中的小常数 epsilon
    "prioritized_replay_eps": 1e-6,
    # 是否使用 LZ4 压缩观测数据
    "compress_observations": True,

    # === 优化器设置 (Optimization) ===
    # Adam 优化器的学习率
    "lr": 5e-4,
    # 学习率调度策略
    "lr_schedule": None,
    # Adam 优化器的 epsilon 超参数
    "adam_epsilon": 1e-8,
    # 如果不为 None，则在优化过程中以此值裁剪梯度
    "grad_norm_clipping": 40,
    # 在学习开始之前，采样多少步模型数据
    "learning_starts": 1000,
    # 一次性向回放缓冲区更新的样本数量。
    # 注意：如果 num_workers > 1，此设置适用于每个 worker。
    "sample_batch_size": 4,
    # 从回放缓冲区采样进行训练的批次大小。
    # 注意：如果设置了 async_updates，每个 worker 返回的梯度批次为此大小。
    "train_batch_size": 32,

    # === 并行设置 (Parallelism) ===
    # 用于收集样本的 worker 数量。只有当环境采样特别慢，
    # 或使用 Async 或 Ape-X 优化器时，才需要增加此值。
    "num_workers": 0,
    # 是否在多个 worker 之间分配不同的 epsilon 值进行探索
    "per_worker_exploration": False,
    # 是否在 worker 上计算优先级
    "worker_side_prioritization": False,
    # 防止迭代时间低于此时间跨度（秒）
    "min_iter_time_s": 1,
})
# __sphinx_doc_end__
# yapf: enable


def make_optimizer(workers, config):
    """创建同步回放优化器 (SyncReplayOptimizer)"""
    return SyncReplayOptimizer(
        workers,
        learning_starts=config["learning_starts"],           # 学习开始前的步数
        buffer_size=config["buffer_size"],                   # 回放缓冲区大小
        prioritized_replay=config["prioritized_replay"],     # 是否使用优先级回放
        prioritized_replay_alpha=config["prioritized_replay_alpha"],  # 优先级 Alpha 参数
        prioritized_replay_beta=config["prioritized_replay_beta"],    # 优先级 Beta 参数
        schedule_max_timesteps=config["schedule_max_timesteps"],      # 调度最大时间步
        beta_annealing_fraction=config["beta_annealing_fraction"],    # Beta 退火比例
        final_prioritized_replay_beta=config["final_prioritized_replay_beta"],  # Beta 最终值
        prioritized_replay_eps=config["prioritized_replay_eps"],      # 优先级 epsilon
        train_batch_size=config["train_batch_size"],         # 训练批次大小
        sample_batch_size=config["sample_batch_size"],       # 采样批次大小
        **config["optimizer"])


def check_config_and_setup_param_noise(config):
    """根据设置更新配置。

    重写 sample_batch_size 以考虑 n_step 截断的影响，
    并添加必要的回调函数以支持参数空间噪声探索。
    """

    # 更新有效批次大小以包含 n-step
    adjusted_batch_size = max(config["sample_batch_size"],
                              config.get("n_step", 1))
    config["sample_batch_size"] = adjusted_batch_size

    # 如果启用了参数空间噪声
    if config.get("parameter_noise", False):
        if config["batch_mode"] != "complete_episodes":
            raise ValueError("使用参数空间噪声进行探索时，"
                             "batch_mode 必须为 complete_episodes。")
        if config.get("noisy", False):
            raise ValueError(
                "参数空间噪声探索与带噪网络（noisy network）"
                "不能同时使用。")
        # 保存用户原有的 episode 开始回调函数
        if config["callbacks"]["on_episode_start"]:
            start_callback = config["callbacks"]["on_episode_start"]
        else:
            start_callback = None

        def on_episode_start(info):
            """作为回调函数，在 episode 开始时为网络参数采样并施加参数空间噪声"""
            policies = info["policy"]
            for pi in policies.values():
                pi.add_parameter_noise()
            if start_callback:
                start_callback(info)

        config["callbacks"]["on_episode_start"] = on_episode_start

        # 保存用户原有的 episode 结束回调函数
        if config["callbacks"]["on_episode_end"]:
            end_callback = config["callbacks"]["on_episode_end"]
        else:
            end_callback = None

        def on_episode_end(info):
            """作为回调函数，监控带噪声策略与原始策略之间的距离"""
            policies = info["policy"]
            episode = info["episode"]
            model = policies[DEFAULT_POLICY_ID].model
            if hasattr(model, "pi_distance"):
                episode.custom_metrics["policy_distance"] = model.pi_distance
            if end_callback:
                end_callback(info)

        config["callbacks"]["on_episode_end"] = on_episode_end


def get_initial_state(config):
    """获取训练器的初始状态"""
    return {
        "last_target_update_ts": 0,    # 上次更新目标网络的时间步
        "num_target_updates": 0,       # 目标网络更新次数
    }


def make_exploration_schedule(config, worker_index):
    """创建探索率调度器。

    可为每个 worker 使用不同的 epsilon 值，或者使用线性调度。
    """
    if config["per_worker_exploration"]:
        assert config["num_workers"] > 1, \
            "这需要多个 worker"
        if worker_index >= 0:
            # 来自 Ape-X 论文的探索常数
            exponent = (
                1 + worker_index / float(config["num_workers"] - 1) * 7)
            return ConstantSchedule(0.4**exponent)
        else:
            # 本地评估 worker 应将探索率设为 0，以确保评估 rollout 正常进行
            return ConstantSchedule(0.0)
    # 使用线性调度，从 1.0 退火到 exploration_final_eps
    return LinearSchedule(
        schedule_timesteps=int(
            config["exploration_fraction"] * config["schedule_max_timesteps"]),
        initial_p=1.0,
        final_p=config["exploration_final_eps"])


def setup_exploration(trainer):
    """为训练器和各个 worker 设置探索调度器"""
    trainer.exploration0 = make_exploration_schedule(trainer.config, -1)
    trainer.explorations = [
        make_exploration_schedule(trainer.config, i)
        for i in range(trainer.config["num_workers"])
    ]


def update_worker_explorations(trainer):
    """更新所有 worker 的探索率（epsilon）"""
    global_timestep = trainer.optimizer.num_steps_sampled
    # 计算本地 worker 的探索值
    exp_vals = [trainer.exploration0.value(global_timestep)]
    trainer.workers.local_worker().foreach_trainable_policy(
        lambda p, _: p.set_epsilon(exp_vals[0]))
    # 遍历所有远程 worker 更新其探索率
    for i, e in enumerate(trainer.workers.remote_workers()):
        exp_val = trainer.explorations[i].value(global_timestep)
        e.foreach_trainable_policy.remote(lambda p, _: p.set_epsilon(exp_val))
        exp_vals.append(exp_val)
    trainer.train_start_timestep = global_timestep
    trainer.cur_exp_vals = exp_vals


def add_trainer_metrics(trainer, result):
    """添加训练过程中的指标到结果字典中"""
    global_timestep = trainer.optimizer.num_steps_sampled
    result.update(
        timesteps_this_iter=global_timestep - trainer.train_start_timestep,
        info=dict({
            "min_exploration": min(trainer.cur_exp_vals),    # 最小探索率
            "max_exploration": max(trainer.cur_exp_vals),    # 最大探索率
            "num_target_updates": trainer.state["num_target_updates"],  # 目标网络更新次数
        }, **trainer.optimizer.stats()))


def update_target_if_needed(trainer, fetches):
    """如果达到更新频率，则更新目标网络"""
    global_timestep = trainer.optimizer.num_steps_sampled
    if global_timestep - trainer.state["last_target_update_ts"] > \
            trainer.config["target_network_update_freq"]:
        # 更新所有可训练策略的目标网络
        trainer.workers.local_worker().foreach_trainable_policy(
            lambda p, _: p.update_target())
        trainer.state["last_target_update_ts"] = global_timestep
        trainer.state["num_target_updates"] += 1


def collect_metrics(trainer):
    """收集训练过程中的各项指标"""
    if trainer.config["per_worker_exploration"]:
        # 如果使用 per_worker_exploration，只从探索率最低的那部分 worker 收集指标
        result = trainer.collect_metrics(
            selected_workers=trainer.workers.remote_workers()[
                -len(trainer.workers.remote_workers()) // 1:])
    else:
        result = trainer.collect_metrics()
    return result


def disable_exploration(trainer):
    """禁用探索（用于评估时）"""
    trainer.evaluation_workers.local_worker().foreach_policy(
        lambda p, _: p.set_epsilon(0))


# ==================== 构建训练器 ====================
# 创建通用离线策略训练器
GenericOffPolicyTrainer = build_trainer(
    name="GenericOffPolicyAlgorithm",       # 通用离线策略算法
    default_policy=None,
    default_config=DEFAULT_CONFIG,
    validate_config=check_config_and_setup_param_noise,  # 配置验证
    get_initial_state=get_initial_state,                # 初始状态
    make_policy_optimizer=make_optimizer,               # 创建优化器
    before_init=setup_exploration,                      # 初始化前设置探索
    before_train_step=update_worker_explorations,       # 训练前更新探索率
    after_optimizer_step=update_target_if_needed,        # 优化步骤后更新目标网络
    after_train_result=add_trainer_metrics,              # 训练后添加指标
    collect_metrics_fn=collect_metrics,                  # 收集指标函数
    before_evaluate_fn=disable_exploration)              # 评估前禁用探索

# DQN 训练器
DQNTrainer = GenericOffPolicyTrainer.with_updates(
    name="DQN", default_policy=DQNTFPolicy, default_config=DEFAULT_CONFIG)

# Simple Q 训练器（简化版 DQN）
SimpleQTrainer = DQNTrainer.with_updates(default_policy=SimpleQPolicy)
