"""
train_baselines.py - 训练论文所需的 baseline 模型

每个 baseline 训练各自的 checkpoint，用于公平对比。

用法:
    # 训练全部 baseline
    python scripts/train_baselines.py --baselines sparse dense --total-steps 200000

    # 只训练 SAC Sparse
    python scripts/train_baselines.py --baselines sparse --total-steps 200000

    # 只训练 SAC Dense
    python scripts/train_baselines.py --baselines dense --total-steps 200000
"""
import sys
import os
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from assembly_env.sim_task import AssemblySimEnv
from imitation_pipeline.rl.sac import SACAgent, ReplayBuffer
from imitation_pipeline.bc.model import BCPolicy
from imitation_pipeline.utils import load_demo_pkl, load_model

import numpy as np
import torch


def create_env(curriculum_phase=2, use_shaped_reward=True):
    """与 train_sac_with_learned_reward.py 的 create_env() 一致"""
    return AssemblySimEnv(
        env_robot=None,
        self_collision_enabled=True,
        renders=False,
        ft_noise=False,
        pose_noise=False,
        action_noise=False,
        physical_noise=False,
        time_step=1 / 250,
        max_steps=400,
        step_limit=True,
        action_dim=6,
        max_vel=0.05,
        max_rad=0.02,
        ft_obs_only=False,
        limit_ft=False,
        max_ft=[1000, 1000, 2500, 100, 100, 100],
        max_position_range=[2] * 3,
        dist_threshold=0.01,
        use_shaped_reward=use_shaped_reward,
        reward_weights={
            "dist_scale": 5.0,
            "progress_scale": 100.0,
            "orn_scale": 5.0,
            "success_bonus": 200.0,
            "time_penalty": 0.1,
            "proximity_bonus": 50.0,
            "proximity_threshold": 0.005,
        },
        curriculum_phase=curriculum_phase,
    )


def guided_exploration(env, agent, n_steps, noise_level=0.1, label=""):
    """引导探索填充 buffer"""
    state, info = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    collected = 0
    successes = 0
    episode_transitions = []

    while collected < n_steps:
        member_pos = np.array(env.member_pose[0])
        target_pos = np.array(env.get_target_pose()[0])
        diff = target_pos - member_pos
        pos_action = np.clip(diff / (np.max(np.abs(diff)) + 1e-8), -1, 1)
        action = np.concatenate([pos_action, np.zeros(3)]).astype(np.float32)
        action = action + np.random.randn(6).astype(np.float32) * noise_level
        action = np.clip(action, -1, 1)

        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_transitions.append((state, action, reward, next_state, done))
        collected += 1
        state = next_state

        if done:
            is_success = bool(info.get('num_success', 0))
            if is_success:
                successes += 1
                for s, a, r, ns, d in episode_transitions:
                    agent.success_buffer.push(s, a, r, ns, float(d))
            else:
                for s, a, r, ns, d in episode_transitions:
                    agent.buffer.push(s, a, r, ns, float(d))
            episode_transitions = []
            state, info = env.reset()
            if isinstance(state, tuple):
                state = state[0]

    print(f"  [{label}] Guided exploration: {n_steps} steps, {successes} successes")
    return successes


def train_baseline(agent, env, total_steps, learning_starts=1000, log_interval=10,
                   save_path=None, label=""):
    """通用 SAC 训练循环"""
    state, info = env.reset()
    if isinstance(state, tuple):
        state = state[0]

    episode_reward = 0
    episode_steps = 0
    episode_num = 0
    recent_successes = []
    best_success_rate = 0.0

    for step in range(total_steps):
        action = agent.select_action(state, sample=True)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        is_success = bool(info.get('num_success', 0)) if isinstance(info, dict) else False
        agent.store_transition(state, action, reward, next_state, done, is_success=is_success)

        episode_reward += reward
        episode_steps += 1
        state = next_state

        if step >= learning_starts:
            agent.update()

        if done:
            episode_num += 1
            recent_successes.append(is_success)
            if len(recent_successes) > 100:
                recent_successes.pop(0)
            success_rate = np.mean(recent_successes) if recent_successes else 0.0

            if episode_num % log_interval == 0:
                print(f"  [{label}] Ep {episode_num:4d} | Step {step:6d} | "
                      f"Reward: {episode_reward:+8.1f} | Len: {episode_steps:3d} | "
                      f"Success: {success_rate:.3f} | SuccBuf: {agent.success_buffer.size}")

            if success_rate > best_success_rate and success_rate > 0:
                best_success_rate = success_rate
                if save_path:
                    agent.save(save_path.replace('.pt', f'_success_{success_rate:.3f}.pt'))

            state, info = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            episode_reward = 0
            episode_steps = 0

    # 保存最终模型
    if save_path:
        agent.save(save_path)
        print(f"  [{label}] Final model saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument('--baselines', nargs='+', default=['sparse', 'dense'],
                        choices=['sparse', 'dense'])
    parser.add_argument('--total-steps', type=int, default=200000)
    parser.add_argument('--curriculum-phase', type=int, default=2)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--demo-path', default='demo_buffer.pkl')
    parser.add_argument('--save-dir', default='imitation_pipeline/rl/checkpoints')
    parser.add_argument('--bc-path', default='imitation_pipeline/bc/bc_policy.pt')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device

    print(f"[Init] Loading demo data...")
    demo_data = load_demo_pkl(args.demo_path)
    obs_dim = demo_data['states'].shape[1]
    action_dim = demo_data['actions'].shape[1]
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- SAC Sparse ----
    if 'sparse' in args.baselines:
        print(f"\n{'='*60}")
        print(f"[SAC Sparse] Training (use_shaped_reward=False)")
        print(f"{'='*60}")
        env = create_env(curriculum_phase=args.curriculum_phase, use_shaped_reward=False)
        agent = SACAgent(obs_dim, action_dim, device=device)
        # 引导探索填充 buffer
        guided_exploration(env, agent, 5000, noise_level=0.1, label="Sparse")
        # 训练
        train_baseline(agent, env, args.total_steps,
                      save_path=os.path.join(args.save_dir, 'sac_sparse_final.pt'),
                      label="Sparse")
        env.close()

    # ---- SAC Dense ----
    if 'dense' in args.baselines:
        print(f"\n{'='*60}")
        print(f"[SAC Dense] Training (env reward only, no learned reward)")
        print(f"{'='*60}")
        env = create_env(curriculum_phase=args.curriculum_phase, use_shaped_reward=True)
        agent = SACAgent(obs_dim, action_dim, device=device)
        # 引导探索填充 buffer
        guided_exploration(env, agent, 5000, noise_level=0.1, label="Dense")
        # 训练
        train_baseline(agent, env, args.total_steps,
                      save_path=os.path.join(args.save_dir, 'sac_dense_final.pt'),
                      label="Dense")
        env.close()

    print(f"\n{'='*60}")
    print(f"[Done] All baselines trained!")
    print(f"  Checkpoints saved in: {args.save_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
