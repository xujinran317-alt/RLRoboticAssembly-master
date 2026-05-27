# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import collections

import ray
from ray.rllib.evaluation.rollout_metrics import RolloutMetrics
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.rllib.offline.off_policy_estimator import OffPolicyEstimate
from ray.rllib.policy.policy import LEARNER_STATS_KEY
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.memory import ray_get_and_free

logger = logging.getLogger(__name__)


@DeveloperAPI
def get_learner_stats(grad_info):
    """从策略中返回优化统计信息。

    从梯度信息中提取学习器（优化器）的统计指标，如 vf_loss、policy_loss 等。

    示例:
        >>> grad_info = evaluator.learn_on_batch(samples)
        >>> print(get_stats(grad_info))
        {"vf_loss": ..., "policy_loss": ...}

    参数:
        grad_info: 梯度信息字典，可能包含学习器统计信息

    返回:
        学习器统计信息字典，包含损失函数等指标
    """

    if LEARNER_STATS_KEY in grad_info:
        return grad_info[LEARNER_STATS_KEY]

    # 处理多智能体情况：遍历每个策略的梯度信息
    multiagent_stats = {}
    for k, v in grad_info.items():
        if type(v) is dict:
            if LEARNER_STATS_KEY in v:
                multiagent_stats[k] = v[LEARNER_STATS_KEY]

    return multiagent_stats


@DeveloperAPI
def collect_metrics(local_worker=None,
                    remote_workers=[],
                    to_be_collected=[],
                    timeout_seconds=180):
    """从 RolloutWorker 实例收集 episode 指标。

    参数:
        local_worker: 本地 worker（评估器）
        remote_workers: 远程 worker 列表
        to_be_collected: 尚未收集结果的远程对象引用列表
        timeout_seconds: 收集超时时间（秒）

    返回:
        汇总后的指标字典
    """

    episodes, to_be_collected = collect_episodes(
        local_worker,
        remote_workers,
        to_be_collected,
        timeout_seconds=timeout_seconds)
    metrics = summarize_episodes(episodes, episodes)
    return metrics


@DeveloperAPI
def collect_episodes(local_worker=None,
                     remote_workers=[],
                     to_be_collected=[],
                     timeout_seconds=180):
    """从给定的评估器中收集新的 episode 指标元组。

    参数:
        local_worker: 本地 worker（评估器）
        remote_workers: 远程 worker 列表
        to_be_collected: 之前未收集完成的远程对象引用
        timeout_seconds: 收集超时时间（秒）

    返回:
        episodes: 收集到的所有 episode 指标列表
        to_be_collected: 本次仍未收集完成的远程对象引用
    """

    if remote_workers:
        # 向所有远程 worker 发送获取指标的请求
        pending = [
            a.apply.remote(lambda ev: ev.get_metrics()) for a in remote_workers
        ] + to_be_collected
        # 等待所有 worker 返回结果，或超时
        collected, to_be_collected = ray.wait(
            pending, num_returns=len(pending), timeout=timeout_seconds * 1.0)
        if pending and len(collected) == 0:
            logger.warning(
                "WARNING: collected no metrics in {} seconds".format(
                    timeout_seconds))
        # 获取并释放远程对象
        metric_lists = ray_get_and_free(collected)
    else:
        metric_lists = []

    # 收集本地 worker 的指标
    if local_worker:
        metric_lists.append(local_worker.get_metrics())
    episodes = []
    for metrics in metric_lists:
        episodes.extend(metrics)
    return episodes, to_be_collected


@DeveloperAPI
def summarize_episodes(episodes, new_episodes):
    """汇总一组 episode 指标元组。

    参数:
        episodes: 平滑后的 episode 集合（包含历史数据）
        new_episodes: 本轮迭代中新产生的 episode

    返回:
        包含各种汇总统计信息的字典
    """

    # 分离真正的 rollout 数据与离线策略估计数据
    episodes, estimates = _partition(episodes)
    new_episodes, _ = _partition(new_episodes)

    episode_rewards = []        # 每个 episode 的总奖励
    episode_lengths = []        # 每个 episode 的长度
    policy_rewards = collections.defaultdict(list)  # 按策略 ID 分组的奖励
    custom_metrics = collections.defaultdict(list)   # 自定义指标
    perf_stats = collections.defaultdict(list)       # 性能统计

    for episode in episodes:
        episode_lengths.append(episode.episode_length)
        episode_rewards.append(episode.episode_reward)
        # for k, v in episode.custom_metrics.items():
        #     custom_metrics[k].append(v)
        for k, v in episode.perf_stats.items():
            perf_stats[k].append(v)
        # 收集多智能体场景下各策略的奖励
        for (_, policy_id), reward in episode.agent_rewards.items():
            if policy_id != DEFAULT_POLICY_ID:
                policy_rewards[policy_id].append(reward)

    # 仅计算当前迭代中 episode 的自定义指标，
    # 以获得准确的成功率等统计信息
    for episode in new_episodes:
        for k, v in episode.custom_metrics.items():
            custom_metrics[k].append(v)

    # 计算奖励的统计信息
    if episode_rewards:
        min_reward = min(episode_rewards)
        max_reward = max(episode_rewards)
    else:
        min_reward = float("nan")
        max_reward = float("nan")
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)

    # 计算各策略的奖励统计
    policy_reward_min = {}
    policy_reward_mean = {}
    policy_reward_max = {}
    for policy_id, rewards in policy_rewards.copy().items():
        policy_reward_min[policy_id] = np.min(rewards)
        policy_reward_mean[policy_id] = np.mean(rewards)
        policy_reward_max[policy_id] = np.max(rewards)

    # 计算自定义指标的均值、最小值和最大值
    for k, v_list in custom_metrics.copy().items():
        custom_metrics[k + "_mean"] = np.mean(v_list)
        filt = [v for v in v_list if not np.isnan(v)]
        if filt:
            custom_metrics[k + "_min"] = np.min(filt)
            custom_metrics[k + "_max"] = np.max(filt)
        else:
            custom_metrics[k + "_min"] = float("nan")
            custom_metrics[k + "_max"] = float("nan")
        del custom_metrics[k]

    # 计算性能统计的平均值
    for k, v_list in perf_stats.copy().items():
        perf_stats[k] = np.mean(v_list)

    # 汇总离线策略估计器的指标
    estimators = collections.defaultdict(lambda: collections.defaultdict(list))
    for e in estimates:
        acc = estimators[e.estimator_name]
        for k, v in e.metrics.items():
            acc[k].append(v)
    for name, metrics in estimators.items():
        for k, v_list in metrics.items():
            metrics[k] = np.mean(v_list)
        estimators[name] = dict(metrics)

    # 返回所有汇总指标
    return dict(
        episode_reward_max=max_reward,               # 最大 episode 奖励
        episode_reward_min=min_reward,               # 最小 episode 奖励
        episode_reward_mean=avg_reward,              # 平均 episode 奖励
        episode_len_mean=avg_length,                 # 平均 episode 长度
        episodes_this_iter=len(new_episodes),        # 本轮迭代的 episode 数量
        policy_reward_min=policy_reward_min,         # 各策略最小奖励
        policy_reward_max=policy_reward_max,         # 各策略最大奖励
        policy_reward_mean=policy_reward_mean,       # 各策略平均奖励
        custom_metrics=dict(custom_metrics),         # 自定义指标
        sampler_perf=dict(perf_stats),               # 采样器性能统计
        off_policy_estimator=dict(estimators))       # 离线策略估计


def _partition(episodes):
    """将指标数据分为真正的 rollout 数据和离线策略估计数据。

    参数:
        episodes: 包含 RolloutMetrics 和 OffPolicyEstimate 的混合列表

    返回:
        rollouts: RolloutMetrics 对象列表（真实 rollout 数据）
        estimates: OffPolicyEstimate 对象列表（离线策略估计数据）
    """

    rollouts, estimates = [], []
    for e in episodes:
        if isinstance(e, RolloutMetrics):
            rollouts.append(e)
        elif isinstance(e, OffPolicyEstimate):
            estimates.append(e)
        else:
            raise ValueError("未知的指标类型: {}".format(e))
    return rollouts, estimates
