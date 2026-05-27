"""
bc/model.py - Behavior Cloning 网络结构

简单的 MLP：state -> action
Loss: MSE(pred_action, demo_action)
"""
import numpy as np
import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    """
    Behavior Cloning Policy Network.
    
    输入: state (obs_dim,)
    输出: action (action_dim,)
    
    网络结构: 
        MLP: obs_dim -> 256 -> 256 -> action_dim
        激活: ReLU (hidden), Tanh (output, 与动作范围 [-1,1] 匹配)
    """
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # 动作范围 [-1, 1]
        )
        
    def forward(self, state):
        return self.net(state)
    
    def get_action(self, state, device='cpu'):
        """推理接口：输入 state，输出 action (numpy)"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.forward(state).cpu().numpy().flatten()
        return action
