"""
bc/train_bc.py - Behavior Cloning 训练脚本

功能：
  1. 加载 human demonstration (.pkl)
  2. 训练 BC policy（MSE loss）
  3. 保存 policy 模型

用法：
  python -m imitation_pipeline.bc.train_bc
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 添加项目根目录
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from imitation_pipeline.bc.model import BCPolicy
from imitation_pipeline.utils import load_demo_pkl, save_model


def train_bc(args):
    """训练 Behavior Cloning policy"""
    
    # ====== 1. 加载数据 ======
    print(f"[BC] Loading demo data from: {args.demo_path}")
    data = load_demo_pkl(args.demo_path)
    states, actions = data['states'], data['actions']
    
    obs_dim = states.shape[1]
    action_dim = actions.shape[1]
    print(f"[BC] Data: {len(states)} transitions, obs_dim={obs_dim}, action_dim={action_dim}")
    
    # ====== 2. 准备 DataLoader ======
    dataset = TensorDataset(
        torch.FloatTensor(states),
        torch.FloatTensor(actions)
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        drop_last=True
    )
    
    # ====== 3. 初始化模型 ======
    device = torch.device(args.device)
    policy = BCPolicy(obs_dim, action_dim, args.hidden_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    
    print(f"[BC] Model: {policy}")
    print(f"[BC] Device: {device}, lr={args.lr}, batch_size={args.batch_size}")
    
    # ====== 4. 训练循环 ======
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_s, batch_a in dataloader:
            batch_s = batch_s.to(device)
            batch_a = batch_a.to(device)
            
            pred_a = policy(batch_s)
            loss = loss_fn(pred_a, batch_a)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        if (epoch + 1) % args.log_interval == 0:
            print(f"[BC] Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f}")
    
    # ====== 5. 保存模型 ======
    save_model(policy, args.save_path)
    print(f"[BC] Done! Model saved to: {args.save_path}")
    
    return policy


def main():
    parser = argparse.ArgumentParser(description="Behavior Cloning Training")
    parser.add_argument('--demo-path', type=str, 
                        default='demo_buffer.pkl',
                        help='Path to demo .pkl file')
    parser.add_argument('--save-path', type=str, 
                        default='imitation_pipeline/bc/bc_policy.pt',
                        help='Path to save trained policy')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--log-interval', type=int, default=10)
    args = parser.parse_args()
    
    train_bc(args)


if __name__ == '__main__':
    main()
