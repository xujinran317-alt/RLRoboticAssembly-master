"""
augment_demo.py - 演示数据增强

方法：
  1. 动作扰动 (action noise 0.01~0.03)
  2. 时间裁剪：每条轨迹切 3 段
  3. 合成一个大 buffer

用法：
  python scripts/augment_demo.py -i demo_buffer.pkl -o demo_buffer_augmented.pkl --noise-std 0.02 --n-segments 3
"""
import argparse
import copy
import pickle
import random
import sys
from pathlib import Path
from collections import deque

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def augment_one_episode(memory, noise_std=0.02, n_segments=3):
    """
    对一条 episode（deque of transitions）做数据增强，返回多条 episode。
    """
    augmented = []
    length = len(memory)
    if length == 0:
        return augmented

    transitions = list(memory)

    # --- 方法1: 动作扰动（整条轨迹加小噪声）---
    noisy_traj = []
    for (obs, action, reward, next_obs, done) in transitions:
        noise = np.random.normal(0, noise_std, size=action.shape).astype(np.float32)
        noisy_action = np.clip(action + noise, -1.0, 1.0)
        noisy_traj.append((obs, noisy_action, reward, next_obs, done))
    augmented.append(deque(noisy_traj))

    # --- 方法2: 再加一条不同噪声的扰动 ---
    noisy_traj2 = []
    for (obs, action, reward, next_obs, done) in transitions:
        noise = np.random.normal(0, noise_std * 1.5, size=action.shape).astype(np.float32)
        noisy_action = np.clip(action + noise, -1.0, 1.0)
        noisy_traj2.append((obs, noisy_action, reward, next_obs, done))
    augmented.append(deque(noisy_traj2))

    # --- 方法3: 时间裁剪 ---
    if length >= 100:
        seg_len = length // n_segments
        for i in range(n_segments):
            start = i * seg_len
            end = length if i == n_segments - 1 else (i + 1) * seg_len
            segment = transitions[start:end]
            # 最后一条的 done 标记保持不变
            augmented.append(deque(segment))

    return augmented


def collect_and_save(input_path='demo_buffer.pkl', output_path='demo_buffer_augmented.pkl',
                     noise_std=0.02, n_segments=3, verbose=True):
    """增强演示数据并保存"""

    # 加载原数据
    with open(input_path, 'rb') as f:
        original_data = pickle.load(f)

    if not isinstance(original_data, list):
        raise ValueError(f"Expected list of episodes, got {type(original_data)}")

    original_episodes = len(original_data)
    original_transitions = sum(len(ep) for ep in original_data)

    if verbose:
        print(f"[Augment] 原始数据: {original_episodes} episodes, {original_transitions} transitions")
        print(f"[Augment] 增强参数: noise_std={noise_std}, n_segments={n_segments}")

    # 增强每条 episode
    all_augmented = []
    for ep_idx, episode in enumerate(original_data):
        augmented_episodes = augment_one_episode(episode, noise_std, n_segments)
        all_augmented.extend(augmented_episodes)
        if verbose and (ep_idx + 1) % 10 == 0:
            print(f"[Augment] 处理 {ep_idx+1}/{original_episodes} ...")

    # 也保留原始轨迹
    all_augmented.extend(copy.deepcopy(original_data))

    total_episodes = len(all_augmented)
    total_transitions = sum(len(ep) for ep in all_augmented)

    # 保存
    with open(output_path, 'wb') as f:
        pickle.dump(all_augmented, f)

    if verbose:
        print(f"\n[Augment] 完成!")
        print(f"[Augment] 原始: {original_episodes} episodes, {original_transitions} transitions")
        print(f"[Augment] 增强后: {total_episodes} episodes, {total_transitions} transitions")
        print(f"[Augment] 扩增倍数: {total_transitions / original_transitions:.1f}x")
        print(f"[Augment] 保存到: {output_path}")

    return all_augmented


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='增强演示数据')
    parser.add_argument('-i', '--input', type=str, default='demo_buffer.pkl')
    parser.add_argument('-o', '--output', type=str, default='demo_buffer_augmented.pkl')
    parser.add_argument('--noise-std', type=float, default=0.02,
                        help='动作噪声标准差 (default: 0.02)')
    parser.add_argument('--n-segments', type=int, default=3,
                        help='每条轨迹切几段 (default: 3)')
    args = parser.parse_args()

    collect_and_save(args.input, args.output, args.noise_std, args.n_segments)
