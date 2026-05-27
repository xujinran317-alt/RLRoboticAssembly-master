#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
rollout.py - 加载训练好的模型进行回放/部署

使用方法：
  python rollout.py /path/to/checkpoint-xxx/checkpoint-xxx --steps 10000

功能：
  - 加载训练好的 checkpoint，让智能体在环境中执行
  - 支持保存 rollout 结果到文件
  - 可用于看训练效果，或部署到真实机器人
"""

import argparse
import collections
import json
import os
import pickle

import gymnasium as gym
import ray
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.env import MultiAgentEnv
from ray.rllib.env.base_env import _DUMMY_AGENT_ID
from ray.rllib.evaluation.episode import _flatten_action
from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
from ray.tune.util import merge_dicts

from ray.tune.registry import register_env
import envs_launcher as el

EXAMPLE_USAGE = """
使用示例（通过 RLlib 命令行）:
    rllib rollout /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl

使用示例（通过可执行文件）:
    ./rollout.py /tmp/ray/checkpoint_dir/checkpoint-0 --run DQN
    --env CartPole-v0 --steps 1000000 --out rollouts.pkl
"""


def create_parser(parser_creator=None):
    """创建命令行参数解析器"""
    parser_creator = parser_creator or argparse.ArgumentParser
    parser = parser_creator(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="加载 checkpoint，让训练好的智能体在环境中执行",
        epilog=EXAMPLE_USAGE)

    parser.add_argument(
        "checkpoint", type=str, help="要加载的 checkpoint 路径")
    required_named = parser.add_argument_group("必需参数")
    required_named.add_argument(
        "--env", type=str, help="使用的 gym 环境")
    parser.add_argument(
        "--no-render",
        default=False,
        action="store_const",
        const=True,
        help="关闭环境渲染（不显示可视化窗口）")
    parser.add_argument(
        "--steps", default=10000, help="回放的步数")
    parser.add_argument(
        "--episodes",
        default=1,
        type=int,
        help="回放的 episode 数量")
    parser.add_argument("--out", default=None, help="输出文件名（保存 rollout 数据）")
    parser.add_argument(
        "--config",
        default="{}",
        type=json.loads,
        help="算法特定配置（如 env, hyperparams），设置后会覆盖 checkpoint 中的配置")
    return parser


def run(args, parser):
    """主流程：加载配置和模型，开始回放"""
    config = {}
    # 从 checkpoint 目录加载训练时的配置
    config_dir = os.path.dirname(args.checkpoint)
    config_path = os.path.join(config_dir, "params.pkl")
    if not os.path.exists(config_path):
        config_path = os.path.join(config_dir, "../params.pkl")
    if not os.path.exists(config_path):
        if not args.config:
            raise ValueError(
                "在 checkpoint 目录或其父目录都找不到 params.pkl")
    else:
        with open(config_path, "rb") as f:
            config = pickle.load(f)
    
    # 限制 worker 数量（回放不需要太多）
    if "num_workers" in config:
        config["num_workers"] = min(2, config["num_workers"])
    config = merge_dicts(config, args.config)
    if not args.env:
        if not config.get("env"):
            parser.error("必须指定 --env 参数")
        args.env = config.get("env")

    # 移除训练时才需要的参数（回放用不上）
    if "num_workers" in config:
        del config["num_workers"]
    if "human_data_dir" in config["optimizer"]:
        del config["optimizer"]["human_data_dir"]
    if "human_demonstration" in config["optimizer"]:
        del config["optimizer"]["human_demonstration"]
    if "multiple_human_data" in config["optimizer"]:
        del config["optimizer"]["multiple_human_data"]
    if "num_replay_buffer_shards" in config["optimizer"]:
        del config["optimizer"]["num_replay_buffer_shards"]
    if "demonstration_zone_percentage" in config["optimizer"]:
        del config["optimizer"]["demonstration_zone_percentage"]
    if "dynamic_experience_replay" in config["optimizer"]:
        del config["optimizer"]["dynamic_experience_replay"]
    if "robot_demo_path" in config["optimizer"]:
        del config["optimizer"]["robot_demo_path"]

    ray.init()

    # 创建 DDPG 智能体（硬编码使用 DDPG，但 rollout 时 Apex-DDPG 也用这个）
    cls = get_agent_class("DDPG")
    agent = cls(env="ROBOTIC_ASSEMBLY", config=config)    

    # 加载 checkpoint 权重
    agent.restore(args.checkpoint)
    num_steps = int(args.steps)
    num_episodes = int(args.episodes)
    rollout(agent, args.env, num_steps, num_episodes, args.out)


class DefaultMapping(collections.defaultdict):
    """default_factory 现在接受缺失的键作为参数"""

    def __missing__(self, key):
        self[key] = value = self.default_factory(key)
        return value


def default_policy_agent_mapping(unused_agent_id):
    """默认的智能体ID到策略ID的映射"""
    return DEFAULT_POLICY_ID


def rollout(agent, env_name, num_steps, num_episodes, out=None):
    """
    在环境中执行训练好的策略
    - agent: 训练好的强化学习智能体
    - env_name: 环境名称
    - num_steps: 最大步数
    - num_episodes: 最大 episode 数
    - out: 如果指定，将 rollout 数据保存到此文件
    """
    policy_agent_mapping = default_policy_agent_mapping

    # 检查是否是多智能体环境
    if hasattr(agent, "workers"):
        env = agent.workers.local_worker().env
        multiagent = isinstance(env, MultiAgentEnv)
        if agent.workers.local_worker().multiagent:
            policy_agent_mapping = agent.config["multiagent"][
                "policy_mapping_fn"]

        policy_map = agent.workers.local_worker().policy_map
        state_init = {p: m.get_initial_state() for p, m in policy_map.items()}
        use_lstm = {p: len(s) > 0 for p, s in state_init.items()}
        action_init = {
            p: _flatten_action(m.action_space.sample())
            for p, m in policy_map.items()
        }
    else:
        env = gym.make(env_name)
        multiagent = False
        use_lstm = {DEFAULT_POLICY_ID: False}

    if out is not None:
        rollouts = []  # 收集所有 rollout 数据
    steps = 0
    episodes = 0
    while steps < (num_steps or steps + 1) and (episodes < num_episodes):
        mapping_cache = {}
        if out is not None:
            rollout = []
        obs = env.reset()
        agent_states = DefaultMapping(
            lambda agent_id: state_init[mapping_cache[agent_id]])
        prev_actions = DefaultMapping(
            lambda agent_id: action_init[mapping_cache[agent_id]])
        prev_rewards = collections.defaultdict(lambda: 0.)
        done = False
        reward_total = 0.0
        while not done and steps < (num_steps or steps + 1):
            # 对于多智能体环境，逐个智能体计算动作
            multi_obs = obs if multiagent else {_DUMMY_AGENT_ID: obs}
            action_dict = {}
            for agent_id, a_obs in multi_obs.items():
                if a_obs is not None:
                    policy_id = mapping_cache.setdefault(
                        agent_id, policy_agent_mapping(agent_id))
                    p_use_lstm = use_lstm[policy_id]
                    if p_use_lstm:
                        a_action, p_state, _ = agent.compute_action(
                            a_obs,
                            state=agent_states[agent_id],
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                        agent_states[agent_id] = p_state
                    else:
                        a_action = agent.compute_action(
                            a_obs,
                            prev_action=prev_actions[agent_id],
                            prev_reward=prev_rewards[agent_id],
                            policy_id=policy_id)
                    a_action = _flatten_action(a_action)
                    action_dict[agent_id] = a_action
                    prev_actions[agent_id] = a_action
            action = action_dict

            action = action if multiagent else action[_DUMMY_AGENT_ID]
            next_obs, reward, done, _ = env.step(action)
            if multiagent:
                for agent_id, r in reward.items():
                    prev_rewards[agent_id] = r
            else:
                prev_rewards[_DUMMY_AGENT_ID] = reward

            if multiagent:
                done = done["__all__"]
                reward_total += sum(reward.values())
            else:
                reward_total += reward

            if out is not None:
                rollout.append([obs, action, next_obs, reward, done])
            steps += 1
            obs = next_obs
        if out is not None:
            rollouts.append(rollout)
        print("Episode reward", reward_total)
        episodes += 1

    if out is not None:
        pickle.dump(rollouts, open(out, "wb"))


if __name__ == "__main__":
    register_env("ROBOTIC_ASSEMBLY", el.env_creator)
    parser = create_parser()
    args = parser.parse_args()
    run(args, parser)
