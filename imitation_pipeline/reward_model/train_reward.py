"""
reward_model/train_reward.py - 奖励模型训练脚本

正样本: demonstration (label=1)
负样本: ① 随机动作（uniform noise） ② replay buffer 中的失败轨迹

用法:
  python -m imitation_pipeline.reward_model.train_reward
"""
import os
import sys
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from imitation_pipeline.reward_model.model import RewardModel
from imitation_pipeline.utils import load_demo_pkl, save_model


def build_negative_samples(
    demo_states, 
    demo_actions, 
    n_neg=None, 
    noise_scale=1.0,
    obs_dim=None, 
    action_dim=None
):
    """
    构建负样本（随机策略采样的 (state, action) 对）。
    
    策略：随机 shuffle 并加噪声，使得负样本不像演示。
    """
    n = len(demo_states) if n_neg is None else int(n_neg)
    
    # 从演示中随机 shuffle state
    neg_states = demo_states[np.random.choice(len(demo_states), n, replace=True)]
    
    # 生成随机动作（主要噪声来源）
    # 动作为均匀随机 [-1, 1] + 演示动作的 shuffle 变体
    rand_actions = np.random.uniform(-noise_scale, noise_scale, size=(n, action_dim))
    shuffle_actions = demo_actions[np.random.choice(len(demo_actions), n, replace=True)]
    
    # 混合：50% 纯随机，50% 随机 shuffle + 噪声
    neg_actions = np.where(
        np.random.rand(n, 1) < 0.5,
        rand_actions,
        shuffle_actions + np.random.randn(n, action_dim) * 0.1
    )
    neg_actions = np.clip(neg_actions, -1, 1)
    
    return neg_states, neg_actions


def train_reward_model(args):
    """训练奖励模型（判别器）"""
    
    # ====== 1. 加载演示数据 ======
    print(f"[RewardModel] Loading demo data from: {args.demo_path}")
    data = load_demo_pkl(args.demo_path)
    demo_states, demo_actions = data['states'], data['actions']
    
    obs_dim = demo_states.shape[1]
    action_dim = demo_actions.shape[1]
    print(f"[RewardModel] Demo: {len(demo_states)} transitions, obs_dim={obs_dim}, action_dim={action_dim}")
    
    # ====== 2. 构建正负样本 ======
    n_demo = len(demo_states)
    n_neg = int(n_demo * args.neg_ratio)  # 负样本数量 = 正样本 * neg_ratio
    
    neg_states, neg_actions = build_negative_samples(
        demo_states, demo_actions,
        n_neg=n_neg,
        noise_scale=args.noise_scale,
        obs_dim=obs_dim,
        action_dim=action_dim
    )
    
    # 合并正负样本
    all_states = np.concatenate([demo_states, neg_states], axis=0)
    all_actions = np.concatenate([demo_actions, neg_actions], axis=0)
    all_labels = np.concatenate([
        np.ones(int(n_demo), dtype=np.float32),
        np.zeros(int(n_neg), dtype=np.float32)
    ], axis=0)
    
    # 打乱
    idx = np.random.permutation(len(all_states))
    all_states = all_states[idx]
    all_actions = all_actions[idx]
    all_labels = all_labels[idx]
    
    print(f"[RewardModel] Total samples: {len(all_states)} (pos={n_demo}, neg={n_neg})")
    
    # ====== 3. 准备 DataLoader ======
    dataset = TensorDataset(
        torch.FloatTensor(all_states),
        torch.FloatTensor(all_actions),
        torch.FloatTensor(all_labels),
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    # ====== 4. 初始化模型 ======
    device = torch.device(args.device)
    reward_model = RewardModel(obs_dim, action_dim, args.hidden_dim).to(device)
    optimizer = optim.Adam(reward_model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()  # 二分类交叉熵
    
    print(f"[RewardModel] Device: {device}, lr={args.lr}, batch_size={args.batch_size}")
    
    # ====== 5. 训练循环 ======
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_s, batch_a, batch_l in dataloader:
            batch_s, batch_a, batch_l = batch_s.to(device), batch_a.to(device), batch_l.to(device)
            
            logits = reward_model(batch_s, batch_a)
            loss = loss_fn(logits, batch_l.unsqueeze(1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        if (epoch + 1) % args.log_interval == 0:
            # 计算正负样本的预测准确率
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                pos_acc = (probs[batch_l == 1] > 0.5).float().mean().item() if (batch_l == 1).sum() > 0 else 0
                neg_acc = (probs[batch_l == 0] < 0.5).float().mean().item() if (batch_l == 0).sum() > 0 else 0
            print(f"[RewardModel] Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | "
                  f"PosAcc: {pos_acc:.3f} | NegAcc: {neg_acc:.3f}")
    
    # ====== 6. 保存模型 ======
    save_model(reward_model, args.save_path)
    print(f"[RewardModel] Done! Model saved to: {args.save_path}")
    
    return reward_model


def main():
    parser = argparse.ArgumentParser(description="Reward Model Training")
    parser.add_argument('--demo-path', type=str, default='demo_buffer.pkl')
    parser.add_argument('--save-path', type=str, 
                        default='imitation_pipeline/reward_model/reward_model.pt')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--neg-ratio', type=float, default=2.0,
                        help='负样本数量 = 正样本 * neg_ratio')
    parser.add_argument('--noise-scale', type=float, default=1.0,
                        help='随机动作的噪声范围')
    parser.add_argument('--device', type=str, 
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--log-interval', type=int, default=10)
    args = parser.parse_args()
    
    train_reward_model(args)


if __name__ == '__main__':
    main()
