"""
reward_model/model.py - 奖励模型网络结构

输入：(state, action) 或 state
输出：标量 reward score (0~1)
训练方式：判别器（正样本=演示，负样本=随机/失败轨迹）
"""
import torch
import torch.nn as nn
import numpy as np


class RewardModel(nn.Module):
    """
    奖励模型（Reward Model / Discriminator）
    
    输入: (state, action) 拼接，或仅 state
    输出: 标量 score (sigmoid 后为 0~1，表示"像演示"的概率)
    
    网络结构: MLP: (obs_dim + action_dim) -> 256 -> 256 -> 1
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = obs_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        
    def forward(self, state, action):
        """
        Args:
            state: (batch, obs_dim) or (obs_dim,)
            action: (batch, action_dim) or (action_dim,)
        Returns:
            logits: (batch, 1) 未经 sigmoid 的分数
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
    
    def predict_reward(self, state, action, device='cpu'):
        """
        预测奖励值（0~1），供 RL 训练时调用
        
        Returns: scalar or numpy array
        """
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(device)
        if isinstance(action, np.ndarray):
            action = torch.FloatTensor(action).to(device)
        with torch.no_grad():
            logits = self.forward(state, action)
            reward = torch.sigmoid(logits)  # 映射到 [0, 1]
        return reward.cpu().numpy().flatten()
