#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""
train.py - 强化学习训练脚本

使用方法：
  python train.py -f hyper_parameters/assembly_apex_ddpg.yaml

功能：
  - 启动 Ray/RLlib 进行分布式强化学习训练
  - 自动记录成功率（success rate）作为训练指标
  - 支持在训练过程中保存机器人的成功演示轨迹
"""

import argparse
import yaml

import ray
from ray.tests.cluster_utils import Cluster 
from ray.tune.config_parser import make_parser
from ray.tune.result import DEFAULT_RESULTS_DIR
from ray.tune.resources import resources_to_json
from ray.tune.tune import _make_scheduler, run_experiments

EXAMPLE_USAGE = """
训练示例（通过 RLlib 命令行）:
    rllib train --run DQN --env CartPole-v0

网格搜索示例（通过 YAML 配置文件）:
    python train.py -f tuned_examples/cartpole-grid-search-example.yaml

注意：-f 参数会覆盖所有其他命令行参数
"""


# additional libraries for dynamic experience replay 
from ray.tune.registry import register_env, register_trainable
from ray.rllib.agents.registry import get_agent_class
import os
import random
import envs_launcher
import utilities as util
import shutil
from collections import deque
import pickle

#========================================
# 回调函数（Callback Functions）
# 用途：
# 1. 自定义评价指标 -> 记录成功率（success rate）
# 2. 保存机器人成功完成任务的演示轨迹
#========================================

def on_episode_start(info):
    """每个 episode 开始时：初始化成功标志为 0"""
    episode = info["episode"]
    episode.user_data["success"] = 0

def on_episode_step(info):    
    """每个 step 时：暂时不做任何事"""
    pass

def on_episode_end(info):
    """每个 episode 结束时：从环境返回的 info 中提取成功标志，记录到自定义指标"""
    episode = info["episode"]
    if len(episode.last_info_for().values()) > 0:
        episode.user_data["success"] = list(episode.last_info_for().values())[0]
        episode.custom_metrics["successful_rate"] = episode.user_data["success"]

def on_sample_end(info):    
    pass

def on_postprocess_traj(info):
    """后处理轨迹（暂时屏蔽，可解开注释来保存成功的机器人演示）"""
    pass 
    # if list(info["episode"].last_info_for().values())[0] > 0:
    #    save_episode(info["post_batch"])
    
def on_train_result(info):
    """训练结束时：把自定义指标的成功率均值写入主结果"""
    if "successful_rate_mean" in info["result"]["custom_metrics"]:
        info["result"]["successful_rate"] = info["result"]["custom_metrics"]["successful_rate_mean"]

def save_episode(samples):
    """保存一个成功 episode 的所有 transition 数据到文件"""
    memory = deque()
    for row in samples.rows():
        obs = row["obs"]
        action = row["actions"]
        reward = row["rewards"]
        new_obs = row["new_obs"]
        done = row["dones"]
        memory.append((obs, action, reward, new_obs, done))
    # 保存到随机文件名
    file_name = dir_path + str(random.random())
    out_file = open(file_name, 'wb')
    pickle.dump(memory, out_file)
    out_file.close()
    util.prGreen("成功保存了一个成功轨迹，长度 {}".format(len(memory)))

def get_task_path(yaml_file):
    """从 YAML 配置中读取保存路径，创建 robot_demos 目录"""
    with open(yaml_file) as f:
        experiments = yaml.safe_load(f)
        experiment_name = next(iter(experiments))
        dir_path = experiments[experiment_name]["local_dir"]
        dir_path = os.path.expanduser(dir_path)
        dir_path = os.path.join(dir_path, experiment_name) + "/robot_demos/"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            shutil.rmtree(dir_path)  # 清空旧的演示数据
            os.makedirs(dir_path)
    return dir_path   
#======================================
# 回调函数定义结束
#======================================


def create_parser(parser_creator=None):
    """创建命令行参数解析器"""
    parser = make_parser(
        parser_creator=parser_creator,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="训练一个强化学习智能体",
        epilog=EXAMPLE_USAGE)

    # 以下参数继承自 Ray Tune 的基础解析器
    parser.add_argument(
        "--ray-address",
        default=None,
        type=str,
        help="连接到已有的 Ray 集群地址（而非启动新集群）")
    parser.add_argument(
        "--ray-num-cpus",
        default=None,
        type=int,
        help="启动新集群时使用的 CPU 数量")
    parser.add_argument(
        "--ray-num-gpus",
        default=None,
        type=int,
        help="启动新集群时使用的 GPU 数量")
    parser.add_argument(
        "--ray-num-nodes",
        default=None,
        type=int,
        help="模拟多个集群节点（用于调试）")
    parser.add_argument(
        "--ray-redis-max-memory",
        default=None,
        type=int,
        help="启动新集群时的 Redis 最大内存")
    parser.add_argument(
        "--ray-memory",
        default=None,
        type=int,
        help="启动新集群时的最大内存")
    parser.add_argument(
        "--ray-object-store-memory",
        default=None,
        type=int,
        help="启动新集群时的对象存储最大内存")
    parser.add_argument(
        "--experiment-name",
        default="default",
        type=str,
        help="`local_dir` 下的子目录名，存放训练结果")
    parser.add_argument(
        "--local-dir",
        default=DEFAULT_RESULTS_DIR,
        type=str,
        help="存放训练结果的本地目录，默认为 '{}'".format(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--upload-dir",
        default="",
        type=str,
        help="将训练结果同步到的 URI（如 s3://bucket）")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="是否尝试恢复之前的 Tune 实验")
    parser.add_argument(
        "--eager",
        action="store_true",
        help="是否启用 TensorFlow Eager 模式")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="是否在 eager 模式下启用 tracing")
    parser.add_argument(
        "--env", default=None, type=str, help="使用的 gym 环境")
    parser.add_argument(
        "--queue-trials",
        action="store_true",
        help="集群资源不足时是否排队等待（自动化伸缩集群时应设为 True）")
    parser.add_argument(
        "-f",
        "--config-file",
        default=None,
        type=str,
        help="使用此 YAML 配置文件中的配置，会覆盖命令行参数")
    return parser

def run(args, parser):
    """开始训练"""
    if args.config_file:
        # 从 YAML 文件读取配置
        with open(args.config_file) as f:
            experiments = yaml.safe_load(f)

            # 给配置添加自定义回调函数，用于记录成功率和保存演示数据
            experiment_name = next(iter(experiments))
            experiments[experiment_name]["config"]["optimizer"]["robot_demo_path"] = dir_path            
            experiments[experiment_name]["config"]["callbacks"] = {
                    "on_episode_start": on_episode_start,
                    "on_episode_step": on_episode_step,
                    "on_episode_end": on_episode_end,
                    "on_sample_end": on_sample_end,
                    "on_train_result": on_train_result,
                    "on_postprocess_traj": on_postprocess_traj
                    }
    else:
        # 从命令行参数构建实验配置
        experiments = {
            args.experiment_name: {
                "run": args.run,
                "checkpoint_freq": args.checkpoint_freq,
                "keep_checkpoints_num": args.keep_checkpoints_num,
                "checkpoint_score_attr": args.checkpoint_score_attr,
                "local_dir": args.local_dir,
                "resources_per_trial": (
                    args.resources_per_trial and
                    resources_to_json(args.resources_per_trial)),
                "stop": args.stop,
                "config": dict(args.config, env=args.env),
                "restore": args.restore,
                "num_samples": args.num_samples,
                "upload_dir": args.upload_dir,
            }
        }

    # 验证配置有效性
    for exp in experiments.values():
        if not exp.get("run"):
            parser.error("必须指定 --run 参数")
        if not exp.get("env") and not exp.get("config", {}).get("env"):
            parser.error("必须指定 --env 参数")
        if args.eager:
            exp["config"]["eager"] = True
        if args.trace:
            if not exp["config"].get("eager"):
                raise ValueError("启用 tracing 必须先启用 --eager")
            exp["config"]["eager_tracing"] = True

    # 初始化 Ray 集群（支持单机多节点模拟）
    if args.ray_num_nodes:
        cluster = Cluster()
        for _ in range(args.ray_num_nodes):
            cluster.add_node(
                num_cpus=args.ray_num_cpus or 1,
                num_gpus=args.ray_num_gpus or 0,
                object_store_memory=args.ray_object_store_memory,
                memory=args.ray_memory,
                redis_max_memory=args.ray_redis_max_memory)
        ray.init(address=cluster.address)
    else:
        ray.init(
            address=args.ray_address,
            object_store_memory=args.ray_object_store_memory,
            memory=args.ray_memory,
            redis_max_memory=args.ray_redis_max_memory,
            num_cpus=args.ray_num_cpus,
            num_gpus=args.ray_num_gpus)
    
    # 开始跑实验
    run_experiments(
        experiments,
        scheduler=_make_scheduler(args),
        queue_trials=args.queue_trials,
        resume=args.resume)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    random.seed(12345)
    
    # 注册自定义环境到 RLlib
    register_env("ROBOTIC_ASSEMBLY", envs_launcher.env_creator)

    # 获取机器人演示轨迹的保存路径
    dir_path = get_task_path(args.config_file)
    
    run(args, parser)
