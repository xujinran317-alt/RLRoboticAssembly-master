"""
copy_to_rllib.py - RLlib 补丁安装脚本

本项目对 Ray 0.7.5 的 RLlib 做了以下定制修改：
  1. DQN agent: 让所有统计信息在终端完整显示（而不是只显示1/3）
  2. Async replay optimizer: 支持人类示范数据混合训练
  3. Replay buffer: 支持动态经验回放和加载人类示范数据
  4. Evaluation metrics: 自定义单步成功率指标（而非历史滚动平均）

使用方法：运行此脚本将补丁文件复制到 RLlib 目录下
    python copy_to_rllib.py
"""

from shutil import copyfile 
import os

# 让 DQN 训练时所有统计信息都打印到终端
dqn_src = "copy_to_rllib/agents/dqn/dqn.py"
dqn_dst = "../agents/dqn/dqn.py"
copyfile(dqn_src, dqn_dst)

# 替换异步 replay optimizer（支持人类示范数据 + 动态经验回放）
async_src = "copy_to_rllib/optimizers/async_replay_optimizer.py"
async_dst = "../optimizers/async_replay_optimizer.py"
copyfile(async_src, async_dst)

# 替换 replay buffer（支持加载人类演示数据）
buffer_src = "copy_to_rllib/optimizers/replay_buffer.py"
buffer_dst = "../optimizers/replay_buffer.py"
copyfile(buffer_src, buffer_dst)

# 替换评价指标计算（只计算当前 transition 的成功率，而非历史平均）
metrics_src = "copy_to_rllib/evaluation/metrics.py"
metrics_dst = "../evaluation/metrics.py"
copyfile(metrics_src, metrics_dst) 
