"""
eval_all_baselines.py - 评估所有 baseline

用法:
    # 评估 Ours（默认 1.5cm，你平时成功的距离）
    python scripts/eval_all_baselines.py --baselines ours

    # 评估全部（需要各自训练好的 checkpoint）
    python scripts/eval_all_baselines.py --baselines bc sparse dense ours

    # 指定距离/课程阶段
    python scripts/eval_all_baselines.py --baselines ours --curriculum-phase 3  # 2.0cm
    python scripts/eval_all_baselines.py --baselines ours --curriculum-phase 2  # 1.5cm

    # 指定 checkpoint
    python scripts/eval_all_baselines.py --baselines ours --sac-path path/to/model.pt
"""
import sys
import os
import json
import argparse
import glob

import numpy as np
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from assembly_env.sim_task import AssemblySimEnv
from imitation_pipeline.rl.sac import SACAgent
from imitation_pipeline.bc.model import BCPolicy
from imitation_pipeline.reward_model.model import RewardModel
from imitation_pipeline.utils import load_demo_pkl, load_model


# 课程阶段 → 距离映射
PHASE_DIST = {0: 1.0, 1: 1.2, 2: 1.5, 3: 2.0, 4: 2.5, 5: 3.0}


def create_env(curriculum_phase=2, renders=False):
    """与 train_sac_with_learned_reward.py 的 create_env() 保持一致"""
    return AssemblySimEnv(
        env_robot=None,
        self_collision_enabled=True,
        renders=renders,
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
        use_shaped_reward=True,
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


def evaluate(env, agent=None, bc_policy=None, n_episodes=100, device='cpu', label=""):
    """通用评估函数"""
    results = []
    for ep in range(n_episodes):
        state, info = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        ep_len = 0
        peak_force_z = 0.0
        peak_force_total = 0.0
        done = False

        while not done:
            if bc_policy is not None:
                action = bc_policy.get_action(state, device)
            else:
                action = agent.select_action(state, sample=False)

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_len += 1

            # state[7:13] = Fx, Fy, Fz, Tx, Ty, Tz
            if len(state) >= 13:
                fx, fy, fz = abs(state[7]), abs(state[8]), abs(state[9])
                f_total = np.sqrt(fx**2 + fy**2 + fz**2)
                peak_force_z = max(peak_force_z, fz)
                peak_force_total = max(peak_force_total, f_total)

        success = info.get('num_success', 0) if isinstance(info, dict) else 0
        results.append({
            'success': int(success),
            'episode_length': ep_len,
            'peak_force_z': peak_force_z,
            'peak_force_total': peak_force_total,
        })

        if (ep + 1) % 10 == 0:
            sr = np.mean([r['success'] for r in results])
            print(f"  [{label}] Episode {ep+1}/{n_episodes} | Success rate: {sr:.1%}")

    return results


def compute_stats(results):
    successes = [r['success'] for r in results]
    lengths = [r['episode_length'] for r in results]
    forces_z = [r['peak_force_z'] for r in results]

    success_lengths = [l for s, l in zip(successes, lengths) if s == 1]
    fail_lengths = [l for s, l in zip(successes, lengths) if s == 0]

    return {
        'success_rate': float(np.mean(successes)),
        'avg_episode_length': float(np.mean(lengths)),
        'avg_episode_length_success': float(np.mean(success_lengths)) if success_lengths else None,
        'avg_episode_length_fail': float(np.mean(fail_lengths)) if fail_lengths else None,
        'avg_peak_force_z': float(np.mean(forces_z)),
        'std_peak_force_z': float(np.std(forces_z)),
        'num_success': int(np.sum(successes)),
        'n_episodes': len(results),
    }


def print_results(all_results, curriculum_phase):
    dist = PHASE_DIST.get(curriculum_phase, '?')
    print(f"\n{'='*80}")
    print(f"  Evaluation @ curriculum_phase={curriculum_phase} ({dist}cm)")
    print(f"{'='*80}")
    print(f"{'Method':<25} {'Success%':>10} {'Avg Len':>10} {'Len(Succ)':>10} {'Peak Fz':>10}")
    print(f"{'-'*80}")
    for name, stats in all_results.items():
        sr = f"{stats['success_rate']*100:.1f}%"
        avg_len = f"{stats['avg_episode_length']:.0f}"
        avg_len_s = f"{stats['avg_episode_length_success']:.0f}" if stats['avg_episode_length_success'] else "N/A"
        peak_fz = f"{stats['avg_peak_force_z']:.1f}"
        print(f"{name:<25} {sr:>10} {avg_len:>10} {avg_len_s:>10} {peak_fz:>10}")
    print(f"{'='*80}")


def find_best_checkpoint():
    """自动找最佳 checkpoint：优先 sac_success_*.pt，其次 checkpoint/policy.pt"""
    ckpt_dir = 'imitation_pipeline/rl/checkpoints'

    # 1. 优先找 success 系列（按成功率排序）
    success_ckpts = sorted(
        glob.glob(os.path.join(ckpt_dir, 'sac_success_*.pt')),
        key=lambda x: float(x.split('_')[-1].replace('.pt', '')),
        reverse=True,
    )
    if success_ckpts:
        print(f"  [Auto] Best success checkpoint: {success_ckpts[0]}")
        return success_ckpts[0]

    # 2. 回退到 checkpoint 目录下的 policy.pt
    ckpt_policy = os.path.join(ckpt_dir, 'checkpoint', 'policy.pt')
    if os.path.exists(ckpt_policy):
        print(f"  [Auto] Using checkpoint policy: {ckpt_policy}")
        return ckpt_policy

    # 3. 回退到 final
    final = os.path.join(ckpt_dir, 'sac_final.pt')
    if os.path.exists(final):
        print(f"  [Auto] Using final checkpoint: {final}")
        return final

    raise FileNotFoundError(f"No SAC checkpoint found in {ckpt_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate baselines")
    parser.add_argument('--baselines', nargs='+', default=['ours'],
                        choices=['bc', 'sparse', 'dense', 'ours'])
    parser.add_argument('--n-episodes', type=int, default=100)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='eval_results.json')
    parser.add_argument('--curriculum-phase', type=int, default=2,
                        help='课程阶段 (0=1cm, 1=1.2cm, 2=1.5cm, 3=2cm, 4=2.5cm, 5=3cm)')
    parser.add_argument('--bc-path', default='imitation_pipeline/bc/bc_policy.pt')
    parser.add_argument('--reward-model-path', default='imitation_pipeline/reward_model/reward_model.pt')
    parser.add_argument('--sac-path', default=None, help='Ours 的 SAC checkpoint')
    parser.add_argument('--sparse-sac-path', default=None, help='SAC Sparse 专用 checkpoint (默认: sac_sparse_final.pt)')
    parser.add_argument('--dense-sac-path', default=None, help='SAC Dense 专用 checkpoint (默认: sac_dense_final.pt)')
    parser.add_argument('--demo-path', default='demo_buffer.pkl')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device

    # 加载维度
    print("[Init] Loading demo data...")
    demo_data = load_demo_pkl(args.demo_path)
    obs_dim = demo_data['states'].shape[1]
    action_dim = demo_data['actions'].shape[1]
    print(f"  obs_dim={obs_dim}, action_dim={action_dim}")

    # 自动找 checkpoint
    sac_path = args.sac_path or find_best_checkpoint()
    ckpt_dir = 'imitation_pipeline/rl/checkpoints'
    sparse_path = args.sparse_sac_path or os.path.join(ckpt_dir, 'sac_sparse_final.pt')
    dense_path = args.dense_sac_path or os.path.join(ckpt_dir, 'sac_dense_final.pt')

    all_results = {}

    # ---- BC Only ----
    if 'bc' in args.baselines:
        print(f"\n{'='*50}")
        print(f"[BC Only] Evaluating @ phase {args.curriculum_phase}...")
        print(f"{'='*50}")
        bc_policy = BCPolicy(obs_dim, action_dim).to(device)
        bc_policy = load_model(bc_policy, args.bc_path, device)
        bc_policy.eval()
        env = create_env(curriculum_phase=args.curriculum_phase)
        results = evaluate(env, bc_policy=bc_policy, n_episodes=args.n_episodes,
                          device=device, label="BC")
        all_results['BC_Only'] = compute_stats(results)
        env.close()

    # ---- SAC (Sparse) ----
    if 'sparse' in args.baselines:
        if not os.path.exists(sparse_path):
            print(f"\n[SKIP] SAC Sparse checkpoint 不存在: {sparse_path}")
            print(f"  先运行: python scripts/train_baselines.py --baselines sparse")
        else:
            print(f"\n{'='*50}")
            print(f"[SAC Sparse] Evaluating @ phase {args.curriculum_phase}...")
            print(f"  Checkpoint: {sparse_path}")
            print(f"{'='*50}")
            env = create_env(curriculum_phase=args.curriculum_phase)
            env.use_shaped_reward = False
            agent = SACAgent(obs_dim, action_dim, device=device)
            agent.load(sparse_path)
            results = evaluate(env, agent=agent, n_episodes=args.n_episodes,
                              device=device, label="Sparse")
            all_results['SAC_Sparse'] = compute_stats(results)
            env.close()

    # ---- SAC (Dense, 纯环境奖励) ----
    if 'dense' in args.baselines:
        if not os.path.exists(dense_path):
            print(f"\n[SKIP] SAC Dense checkpoint 不存在: {dense_path}")
            print(f"  先运行: python scripts/train_baselines.py --baselines dense")
        else:
            print(f"\n{'='*50}")
            print(f"[SAC Dense] Evaluating @ phase {args.curriculum_phase}...")
            print(f"  Checkpoint: {dense_path}")
            print(f"{'='*50}")
            env = create_env(curriculum_phase=args.curriculum_phase)
            agent = SACAgent(obs_dim, action_dim, device=device)
            agent.load(dense_path)
            results = evaluate(env, agent=agent, n_episodes=args.n_episodes,
                              device=device, label="Dense")
            all_results['SAC_Dense'] = compute_stats(results)
            env.close()

    # ---- Ours (BC + SAC + Learned Reward) ----
    if 'ours' in args.baselines:
        print(f"\n{'='*50}")
        print(f"[Ours] Evaluating @ phase {args.curriculum_phase}...")
        print(f"{'='*50}")
        # Ours 评估时不带 reward model（和 train 的 eval 模式一致）
        agent = SACAgent(obs_dim, action_dim, device=device)
        agent.load(sac_path)
        env = create_env(curriculum_phase=args.curriculum_phase)
        results = evaluate(env, agent=agent, n_episodes=args.n_episodes,
                          device=device, label="Ours")
        all_results['Ours'] = compute_stats(results)
        env.close()

    # ---- 打印 & 保存 ----
    print_results(all_results, args.curriculum_phase)

    output_data = {}
    for name, stats in all_results.items():
        output_data[name] = {k: v for k, v in stats.items()}
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
