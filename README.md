git add -A && git commit -m "xxx" && git push

# RL Robotic Assembly — 机器人装配强化学习

基于 **模仿学习 + SAC 强化学习** 的机器人装配任务训练框架。

任务：PyBullet 仿真搭接关节（lap joint）装配，末端执行器从上方移动并对准目标工件插入。

---

## 环境要求

```
Python 3.10+
pybullet >= 3.2
gymnasium >= 1.0
torch >= 2.0
numpy
transforms3d
```

---

## 快速开始

```bash
# 一键运行完整 pipeline（BC → Reward Model → SAC）
python -m imitation_pipeline.run_pipeline

# 只跑 SAC（BC 和 Reward Model 已训练好时）
python -m imitation_pipeline.run_pipeline --skip-bc --skip-reward
```

---

## 分步执行

### 1. 采集演示数据

```bash
python scripts/collect_demo_buffer.py -n 30 --noise -o demo_buffer.pkl
```

### 2. Behavior Cloning（行为克隆）

```bash
python -m imitation_pipeline.bc.train_bc --demo-path demo_buffer.pkl --epochs 100
```

### 3. Reward Model 训练

```bash
python -m imitation_pipeline.reward_model.train_reward --demo-path demo_buffer.pkl --epochs 100
```

### 4. SAC 训练

```bash
python -m imitation_pipeline.rl.train_sac_with_learned_reward --mode train --total-steps 200000
```

### 5. 评估

```bash
python -m imitation_pipeline.rl.train_sac_with_learned_reward --mode eval --checkpoint imitation_pipeline/rl/checkpoints/sac_final.pt
```

---

## 课程学习（Curriculum Learning）

训练自动从近距离开始，成功率达标后逐步增加难度：

| 阶段 | 起始距离 | 解锁条件 |
|------|----------|----------|
| Phase 0 | 1.0cm | 默认 |
| Phase 1 | 1.5cm | 成功率 > 20%，至少 30 episodes |
| Phase 2 | 2.0cm | 同上 |
| Phase 3 | 2.5cm | 同上 |
| Phase 4 | 3.0cm（原始） | 同上 |

---

## 奖励设计

采用**进度主导**的稠密奖励：

```python
reward = progress * 100      # 靠近目标 = 正奖励（主导信号）
       + advantage * 5       # 距离优势（相对于初始距离）
       - orn_dist * 5        # 姿态对齐惩罚
       - 0.1                 # 时间惩罚（鼓励高效完成）
       + 200 (if success)    # 成功大奖
```

- `progress = prev_dist - curr_dist`：只要比上一步更近就给正奖励
- `advantage = max_dist - curr_dist`：初始 ≈ 0，靠近时增大

---

## 断点续训

```bash
# 从 checkpoint 恢复
python -m imitation_pipeline.rl.train_sac_with_learned_reward \
    --mode train \
    --resume imitation_pipeline/rl/checkpoints/checkpoint \
    --total-steps 300000
```

Checkpoint 结构：
```
imitation_pipeline/rl/checkpoints/
├── checkpoint/
│   ├── policy.pt        # 网络权重
│   ├── optimizers.pt    # 优化器状态
│   └── meta.json        # 训练元数据
└── sac_checkpoint_step_*.pt
```

---

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--reward-alpha` | 1.0 | 环境奖励权重 |
| `--reward-beta` | 0.0 | Learned reward 权重（0 = 纯环境奖励） |
| `--total-steps` | 300000 | 总训练步数 |
| `--warm-start-steps` | 200 | 引导式探索步数 |
| `--learning-starts` | 1000 | 开始 SAC 更新的步数 |
| `--lr` | 3e-4 | 学习率 |
| `--gamma` | 0.99 | 折扣因子 |
| `--tau` | 0.005 | 软更新系数 |
| `--alpha` | 0.2 | SAC 初始温度 |
| alpha_min | 0.1 | SAC 最低温度（防熵崩溃） |

---

## 项目结构

```
RLRoboticAssembly-master/
├── assembly_env/                # Gymnasium 环境
│   ├── base_task.py             # 环境基类（观测/动作空间、距离计算）
│   ├── sim_task.py              # PyBullet 仿真（稠密奖励、课程学习）
│   └── robots/
│       └── sim_robotless.py     # 无机器人模型（搭接关节任务）
├── imitation_pipeline/          # 模仿学习 + RL pipeline
│   ├── bc/                      # Behavior Cloning
│   ├── reward_model/            # 奖励模型（判别器）
│   ├── rl/
│   │   ├── sac.py               # SAC 实现（含 RunningMeanStd）
│   │   └── train_sac_with_learned_reward.py  # SAC 训练脚本
│   └── utils.py                 # 数据加载工具
├── configs/
│   └── assembly_sac.yaml        # SAC 配置文件
├── scripts/                     # CLI 工具
│   ├── collect_demo_buffer.py   # 采集演示
│   ├── augment_demo.py          # 数据增强
│   └── cli.py                   # 命令行接口
└── demo_buffer.pkl              # 演示数据
```

---

## 许可

见 [LICENSE.md](LICENSE.md)。
