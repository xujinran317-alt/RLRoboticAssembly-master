"""
rl/sac.py - SAC (Soft Actor-Critic) 算法实现

核心改进：使用 learned_reward 替代或辅助环境 reward。
修改位置在 `sac.update()` 中的 reward 计算部分。
"""
import os
import json
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


# ========== 网络结构 ==========

class MLPNetwork(nn.Module):
    """通用的 MLP 网络，用于 Q 函数和 Policy"""
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.net(x)


class SquashedGaussianPolicy(nn.Module):
    """
    SAC 策略网络：输出高斯分布，经 tanh 变换到 [-1, 1]
    输入: state
    输出: action
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.fc = MLPNetwork(obs_dim, hidden_dim, hidden_dim)  # 复用 MLP
        # 重新定义：用两个线性层编码
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
    def forward(self, state):
        h = self.shared(state)
        mean = self.mean(h)
        log_std = self.log_std(h)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(self, state):
        """采样动作，返回 action, log_prob"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # reparameterization trick
        action = torch.tanh(x_t)
        # 计算 log prob（考虑 tanh 变换）
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob
    
    def get_action(self, state, device='cpu'):
        """推理接口：输出 deterministic action"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            mean, _ = self.forward(state)
            action = torch.tanh(mean)
        return action.cpu().numpy().flatten()


class TwinQ(nn.Module):
    """
    双 Q 网络（Twin Q-functions），用于 SAC 减少 overestimation
    输入: (state, action)
    输出: Q(s,a) 标量
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q1 = MLPNetwork(obs_dim + action_dim, 1, hidden_dim)
        self.q2 = MLPNetwork(obs_dim + action_dim, 1, hidden_dim)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)
    
    def both(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1(x), self.q2(x)


# ========== Running Statistics (for reward normalization) ==========

class RunningMeanStd:
    """
    Welford 在线算法：追踪运行均值和标准差。
    用于奖励归一化，替代 per-batch z-score，确保训练稳定性。
    """
    def __init__(self, epsilon=1e-8):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon
    
    def update(self, x):
        """用 batch 数据更新统计量"""
        x = np.asarray(x, dtype=np.float64).flatten()
        batch_mean = x.mean()
        batch_count = len(x)
        delta = batch_mean - self.mean

        new_count = self.count + batch_count
        self.mean += delta * batch_count / new_count
        batch_var = x.var() if batch_count > 1 else 0.0
        self.var = (
            (self.var * self.count + batch_var * batch_count
             + delta**2 * self.count * batch_count / new_count)
            / new_count
        )
        self.count = new_count
    
    def normalize(self, x):
        """归一化到 ~N(0,1)"""
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


# ========== Replay Buffer ==========

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity, obs_dim, action_dim, device='cpu'):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0
        
        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
    
    def push(self, state, action, reward, next_state, done):
        idx = self.ptr % self.capacity
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.states[idx]).to(self.device),
            torch.FloatTensor(self.actions[idx]).to(self.device),
            torch.FloatTensor(self.rewards[idx]).to(self.device),
            torch.FloatTensor(self.next_states[idx]).to(self.device),
            torch.FloatTensor(self.dones[idx]).to(self.device),
        )


# ========== SAC Agent ==========

class SACAgent:

    def __init__(
        self,
        obs_dim,
        action_dim,
        reward_model=None,           # 奖励模型（可选）
        reward_alpha=0.0,            # 环境奖励权重
        reward_beta=1.0,             # learned reward 权重
        hidden_dim=256,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,                   # SAC 温度系数
        lr=3e-4,
        batch_size=256,
        buffer_capacity=100000,
        device='cpu',
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.device = device
        self.reward_model = reward_model
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        
        # --- 奖励归一化（运行统计量） ---
        self.reward_rms = RunningMeanStd()
        
        # --- 网络 ---
        self.policy = SquashedGaussianPolicy(obs_dim, action_dim, hidden_dim).to(device)
        self.q = TwinQ(obs_dim, action_dim, hidden_dim).to(device)
        self.q_target = TwinQ(obs_dim, action_dim, hidden_dim).to(device)
        self.q_target.load_state_dict(self.q.state_dict())
        
        # --- 优化器 ---
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optim = optim.Adam(self.q.parameters(), lr=lr)
        
        # --- 自动调节温度 alpha ---
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
        # 目标熵设为 -2 （而不是 -action_dim=-6），让策略更快收敛到确定性
        self.target_entropy = -2.0
        
        # --- Alpha 下限：防止熵崩溃，确保策略始终有最低探索 ---
        self.alpha_min = 0.1
        
        # --- 回放缓冲区 ---
        self.buffer = ReplayBuffer(buffer_capacity, obs_dim, action_dim, device)
        
        # --- 训练步数计数器 ---
        self.step_counter = 0
    
    def select_action(self, state, sample=True):
        """选择动作（训练时用 sample 随机探索，评估时用 mean 确定）"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if sample:
                action, _ = self.policy.sample(state)
            else:
                mean, _ = self.policy.forward(state)
                action = torch.tanh(mean)
        return action.cpu().numpy().flatten()
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储 transition 到 replay buffer"""
        self.buffer.push(state, action, reward, next_state, float(done))
    
    def update(self):
        """
        SAC 更新步骤。
        
        【核心改动位置】
        原本: r = env_reward
        现在: 
          if reward_model is not None:
              learned_r = reward_model(state, action)  # 0~1 的分数
              r = reward_alpha * env_reward + reward_beta * learned_r
          或完全替换：reward_alpha=0, reward_beta=1
        """
        if self.buffer.size < self.batch_size:
            return {}
        
        # --- 采样 ---
        states, actions, env_rewards, next_states, dones = self.buffer.sample(self.batch_size)
        
        # ===== 【关键改动】使用 learned reward + 运行统计归一化 =====
        if self.reward_model is not None:
            with torch.no_grad():
                learned_r_logits = self.reward_model(states, actions)
                learned_r = torch.sigmoid(learned_r_logits)  # [0, 1]
            # 用运行统计量归一化环境奖励（而非 per-batch z-score）
            # 这样归一化后的均值会随训练进展逐渐上升，驱动策略改进
            self.reward_rms.update(env_rewards.cpu().numpy())
            env_rewards_norm = torch.FloatTensor(
                self.reward_rms.normalize(env_rewards.cpu().numpy())
            ).to(self.device)
            rewards = self.reward_alpha * env_rewards_norm + self.reward_beta * learned_r
        
        # --- 更新 Q 函数 ---
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            q1_target, q2_target = self.q_target.both(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target)
            q_target = q_target - self.alpha * next_log_probs
            y = rewards + self.gamma * (1 - dones) * q_target
        
        q1, q2 = self.q.both(states, actions)
        q1_loss = F.mse_loss(q1, y)
        q2_loss = F.mse_loss(q2, y)
        q_loss = q1_loss + q2_loss
        
        self.q_optim.zero_grad()
        q_loss.backward()
        self.q_optim.step()
        
        # --- 更新 policy ---
        new_actions, log_probs = self.policy.sample(states)
        q1_new, q2_new = self.q.both(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_probs - q_new).mean()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        # --- 更新温度 alpha ---
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()
        self.alpha = max(self.log_alpha.exp().item(), self.alpha_min)
        
        # --- 软更新目标网络 ---
        for param, target_param in zip(self.q.parameters(), self.q_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.step_counter += 1
        
        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': float(self.alpha),
            'reward_mean': rewards.mean().item(),
            'learned_reward_mean': learned_r.mean().item() if self.reward_model is not None else 0,
        }
    
    def save(self, path):
        torch.save({
            'policy': self.policy.state_dict(),
            'q': self.q.state_dict(),
            'q_target': self.q_target.state_dict(),
        }, path)
        print(f"[SAC] Model saved to: {path}")
    
    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.q.load_state_dict(checkpoint['q'])
        self.q_target.load_state_dict(checkpoint['q_target'])
        print(f"[SAC] Model loaded from: {path}")
    
    # ====== Checkpoint (full state for resume training) ======

    def save_checkpoint(self, ckpt_dir, step, episode_num, reward_history, recent_successes,
                        best_success_rate, save_buffer=False):
        """
        保存完整训练状态（网络 + 优化器 + 元数据 + 可选经验池）
        结构：
          ckpt_dir/
          ├── policy.pt
          ├── optimizers.pt
          ├── meta.json
          └── replay_buffer.pkl  (可选)
        """
        os.makedirs(ckpt_dir, exist_ok=True)

        # 1. 网络权重
        torch.save({
            'policy': self.policy.state_dict(),
            'q1': self.q.q1.state_dict(),
            'q2': self.q.q2.state_dict(),
            'q_target1': self.q_target.q1.state_dict(),
            'q_target2': self.q_target.q2.state_dict(),
            'log_alpha': self.log_alpha.detach().cpu(),
        }, os.path.join(ckpt_dir, 'policy.pt'))

        # 2. 优化器状态（含 Adam 动量）
        torch.save({
            'policy_optim': self.policy_optim.state_dict(),
            'q_optim': self.q_optim.state_dict(),
            'alpha_optim': self.alpha_optim.state_dict(),
        }, os.path.join(ckpt_dir, 'optimizers.pt'))

        # 3. 元数据
        with open(os.path.join(ckpt_dir, 'meta.json'), 'w') as f:
            json.dump({
                'step': step,
                'episode': episode_num,
                'best_success_rate': best_success_rate,
                'reward_history': reward_history[-500:] if reward_history else [],
                'recent_successes': [int(s) for s in recent_successes],
                'buffer_size': self.buffer.size,
                'buffer_capacity': self.buffer.capacity,
                'alpha': float(self.alpha),
            }, f, indent=2)

        # 4. 经验池（可选）
        if save_buffer:
            self.save_buffer(os.path.join(ckpt_dir, 'replay_buffer.pkl'))

        print(f"[Checkpoint] Saved at step {step}, episode {episode_num} → {ckpt_dir}")

    def load_checkpoint(self, ckpt_dir, load_buffer=False):
        """
        加载完整训练状态。返回 (step, episode_num, reward_history, recent_successes, best_success_rate)
        如果 checkpoint 不存在，返回 (0, 0, [], [], 0.0)
        """
        meta_path = os.path.join(ckpt_dir, 'meta.json')
        if not os.path.exists(meta_path):
            print(f"[Checkpoint] No checkpoint found at {ckpt_dir}, starting from scratch")
            return 0, 0, [], [], 0.0

        # 1. 网络权重
        policy_ckpt = torch.load(os.path.join(ckpt_dir, 'policy.pt'), map_location=self.device)
        self.policy.load_state_dict(policy_ckpt['policy'])
        self.q.q1.load_state_dict(policy_ckpt['q1'])
        self.q.q2.load_state_dict(policy_ckpt['q2'])
        self.q_target.q1.load_state_dict(policy_ckpt['q_target1'])
        self.q_target.q2.load_state_dict(policy_ckpt['q_target2'])
        self.log_alpha = torch.tensor(policy_ckpt['log_alpha'], requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp().item()

        # 2. 优化器
        optim_ckpt = torch.load(os.path.join(ckpt_dir, 'optimizers.pt'), map_location=self.device)
        self.policy_optim.load_state_dict(optim_ckpt['policy_optim'])
        self.q_optim.load_state_dict(optim_ckpt['q_optim'])
        self.alpha_optim.load_state_dict(optim_ckpt['alpha_optim'])

        # 3. 元数据
        with open(meta_path) as f:
            meta = json.load(f)

        # 4. 经验池（可选）
        if load_buffer:
            buffer_path = os.path.join(ckpt_dir, 'replay_buffer.pkl')
            if os.path.exists(buffer_path):
                self.load_buffer(buffer_path)

        print(f"[Checkpoint] Resumed at step {meta['step']}, episode {meta['episode']}")
        return (
            meta['step'],
            meta['episode'],
            meta.get('reward_history', []),
            meta.get('recent_successes', []),
            meta.get('best_success_rate', 0.0),
        )

    def save_buffer(self, path):
        """保存 replay buffer 到文件"""
        buffer_data = {
            'states': self.buffer.states[:self.buffer.size],
            'actions': self.buffer.actions[:self.buffer.size],
            'rewards': self.buffer.rewards[:self.buffer.size],
            'next_states': self.buffer.next_states[:self.buffer.size],
            'dones': self.buffer.dones[:self.buffer.size],
            'ptr': self.buffer.ptr,
            'size': self.buffer.size,
            'capacity': self.buffer.capacity,
        }
        with open(path, 'wb') as f:
            pickle.dump(buffer_data, f)
        print(f"[Buffer] Saved {self.buffer.size} transitions → {path}")

    def load_buffer(self, path):
        """加载 replay buffer"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.buffer.states[:data['size']] = data['states']
        self.buffer.actions[:data['size']] = data['actions']
        self.buffer.rewards[:data['size']] = data['rewards']
        self.buffer.next_states[:data['size']] = data['next_states']
        self.buffer.dones[:data['size']] = data['dones']
        self.buffer.ptr = data['ptr']
        self.buffer.size = data['size']
        self.buffer.capacity = data.get('capacity', self.buffer.capacity)
        print(f"[Buffer] Loaded {data['size']} transitions from {path}")
