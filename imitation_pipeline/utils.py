"""
utils.py - 数据加载、保存等通用工具
"""
import pickle
import os
from pathlib import Path
import numpy as np
import torch


def load_demo_pkl(pkl_path="demo_buffer.pkl"):
    """
    从 .pkl 文件中加载 human demonstration 数据。
    
    数据格式（本项目）：list of deque，其中 deque 中每个元素是
    (state, action, reward, next_state, done) 的 tuple。
    
    返回：
        states: (N, obs_dim) numpy array
        actions: (N, action_dim) numpy array
        rewards: (N,) numpy array
        next_states: (N, obs_dim) numpy array
        dones: (N,) numpy array
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    # 本项目 demo_buffer.pkl 是 list[deque]
    if isinstance(data, list):
        # 合并所有 deque
        all_transitions = []
        for d in data:
            all_transitions.extend(d)
    elif hasattr(data, '__getitem__'):
        all_transitions = list(data)
    else:
        raise ValueError(f"Unexpected data type: {type(data)}")
    
    states, actions, rewards, next_states, dones = [], [], [], [], []
    for item in all_transitions:
        s, a, r, ns, d_flag = item
        states.append(np.array(s, dtype=np.float32))
        actions.append(np.array(a, dtype=np.float32))
        rewards.append(float(r))
        next_states.append(np.array(ns, dtype=np.float32))
        dones.append(bool(d_flag))
    
    return {
        'states': np.stack(states),
        'actions': np.stack(actions),
        'rewards': np.array(rewards, dtype=np.float32),
        'next_states': np.stack(next_states),
        'dones': np.array(dones, dtype=np.bool_),
    }


def save_model(model, save_path):
    """保存 PyTorch 模型"""
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[Saved] Model -> {save_path}")


def load_model(model, load_path, device='cpu'):
    """加载 PyTorch 模型"""
    model.load_state_dict(torch.load(load_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"[Loaded] Model <- {load_path}")
    return model
