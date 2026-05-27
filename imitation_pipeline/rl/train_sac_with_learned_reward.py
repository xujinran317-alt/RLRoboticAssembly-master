"""
rl/train_sac_with_learned_reward.py - 使用 Learned Reward 的 SAC 训练脚本

三阶段 pipeline 集成:
  1. 加载预训练的 BC policy(可选,用于 warm start)
  2. 加载预训练的 Reward Model(必需)
  3. 使用 learned reward 训练 SAC

用法:
  python -m imitation_pipeline.rl.train_sac_with_learned_reward --mode train
  python -m imitation_pipeline.rl.train_sac_with_learned_reward --mode eval --checkpoint xxx.pt
"""
import os
import sys
import argparse
import time
import warnings
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Suppress gym NumPy 2.x compatibility warning
warnings.filterwarnings("ignore", message="Gym has been unmaintained", category=UserWarning)

import torch

from imitation_pipeline.rl.sac import SACAgent
from imitation_pipeline.reward_model.model import RewardModel
from imitation_pipeline.bc.model import BCPolicy
from imitation_pipeline.utils import load_demo_pkl, load_model


def create_env(renders=False, curriculum_phase=2):
    """
    创建 PyBullet 装配环境(使用新式 AssemblySimEnv,带稠密奖励)。
    """
    # 确保项目根目录在 sys.path 中
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from assembly_env.sim_task import AssemblySimEnv

    env = AssemblySimEnv(
        env_robot=None,                         # 默认使用 RobotSimRobotless
        self_collision_enabled=True,
        renders=renders,
        ft_noise=False,
        pose_noise=False,
        action_noise=False,
        physical_noise=False,
        time_step=1/250,
        max_steps=200,                          # TASK 1: 4000 -> 200
        step_limit=True,
        action_dim=6,
        max_vel=0.05,                           # TASK 4: ~2mm/step (need 200mm range)
        max_rad=0.02,                           # TASK 4: ~0.004 rad/step ~ 0.8 rad/200ep
        ft_obs_only=False,
        limit_ft=False,
        max_ft=[1000, 1000, 2500, 100, 100, 100],
        max_position_range=[2]*3,
        dist_threshold=0.01,
        use_shaped_reward=True,
        reward_weights={
            "dist_scale": 5.0,                  # 距离优势系数(辅助信号,不宜过大)
            "progress_scale": 100.0,            # 进步奖励系数(主导信号)
            "orn_scale": 5.0,                   # 姿态惩罚系数
            "success_bonus": 200.0,             # 成功大奖
            "time_penalty": 0.1,                # 每步时间惩罚(鼓励高效完成)
        },
        curriculum_phase=curriculum_phase,
    )
    return env


def warm_start_with_bc(agent, bc_policy, env, device='cpu', n_steps=5000, curriculum_phase=0):
    """
    用 BC policy + 引导式探索填充 replay buffer。
    在多个课程阶段采样,确保 buffer 包含不同难度的数据。
    """
    print("[SAC] Warm-starting replay buffer with BC + guided exploration...")

    collected = 0
    total_successes = 0

    # ---- 引导式探索:在当前课程阶段 + 下一阶段采样 ----
    phases_to_sample = [curriculum_phase]
    if curriculum_phase < 2:
        phases_to_sample.append(curriculum_phase + 1)  # 也采样下一难度

    for phase in phases_to_sample:
        env.curriculum_phase = phase
        print(f"[SAC] Guided exploration at curriculum phase {phase}...")
        state, info = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        phase_steps = n_steps // (2 * len(phases_to_sample))
        phase_successes = 0
        for _ in range(phase_steps):
            member_pos = np.array(env.member_pose[0])
            target_pos = np.array(env.get_target_pose()[0])
            diff = target_pos - member_pos
            pos_action = np.clip(diff / (np.max(np.abs(diff)) + 1e-8), -1, 1)
            action = np.concatenate([pos_action, np.zeros(3)]).astype(np.float32)
            action = action + np.random.randn(6).astype(np.float32) * 0.1
            action = np.clip(action, -1, 1)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            collected += 1
            state = next_state

            if done:
                if info.get('num_success', 0):
                    phase_successes += 1
                state, info = env.reset()
                if isinstance(state, tuple):
                    state = state[0]

        total_successes += phase_successes
        print(f"[SAC] Phase {phase}: {phase_steps} steps, {phase_successes} successes")

    # ---- BC policy 填充剩余 ----
    env.curriculum_phase = curriculum_phase
    if bc_policy is not None:
        print("[SAC] BC policy exploration...")
        state, info = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        while collected < n_steps:
            action = bc_policy.get_action(state, device)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_transition(state, action, reward, next_state, done)
            collected += 1
            state = next_state

            if done:
                state, info = env.reset()
                if isinstance(state, tuple):
                    state = state[0]

    print(f"[SAC] Warm-start done! Buffer size: {agent.buffer.size}, total successes: {total_successes}")


def train_with_learned_reward(args):
    """主训练循环"""

    device = torch.device(args.device)

    # ====== 1. 加载 Reward Model ======
    print("=" * 60)
    print("Phase 1: Loading Reward Model")
    print("=" * 60)

    # 从 demo 数据获取维度信息
    demo_data = load_demo_pkl(args.demo_path)
    obs_dim = demo_data['states'].shape[1]
    action_dim = demo_data['actions'].shape[1]

    reward_model = RewardModel(obs_dim, action_dim, args.hidden_dim).to(device)
    reward_model = load_model(reward_model, args.reward_model_path, device)
    reward_model.eval()

    # ====== 2. 加载 BC Policy(可选,用于 warm start)======
    bc_policy = None
    if args.bc_policy_path and os.path.exists(args.bc_policy_path):
        print("=" * 60)
        print("Phase 2 (Optional): Loading BC Policy for warm start")
        print("=" * 60)
        bc_policy = BCPolicy(obs_dim, action_dim, args.hidden_dim).to(device)
        bc_policy = load_model(bc_policy, args.bc_policy_path, device)
        bc_policy.eval()

    # ====== 3. 初始化 SAC Agent (混合奖励:环境稠密奖励 + Learned Reward) ======
    print("=" * 60)
    print("Phase 3: SAC Training (Hybrid Reward: env + learned)")
    print("=" * 60)

    # ====== 混合奖励:环境稠密奖励 + Learned Reward ======
    # learned reward 提供"像演示"的信号(0~1),帮助策略理解哪些动作方向是对的
    # env reward 提供"距离"等物理信号,帮助策略精准对齐
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        reward_model=reward_model,          # Learned Reward(用于混合)
        reward_alpha=args.reward_alpha,     # 环境奖励权重
        reward_beta=args.reward_beta,       # Learned reward 权重
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        tau=args.tau,
        alpha=args.alpha,
        lr=args.lr,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        device=device,
    )

    print(f"[SAC] Reward mode: HYBRID (alpha={args.reward_alpha} * env + beta={args.reward_beta} * learned)")
    print(f"[SAC] Alpha (entropy temp): {args.alpha}")

    # ====== 4. 创建环境(课程学习:从近距离开始)======
    # Phase 0=1.0cm → 1=1.2cm → 2=1.5cm → 3=2.0cm → 4=2.5cm → 5=3.0cm
    # 每步只增加0.2~0.5cm，保证策略能平滑泛化
    curriculum_phase = 0
    curriculum_heights = {0: 1.0, 1: 1.2, 2: 1.5, 3: 2.0, 4: 2.5, 5: 3.0}
    max_curriculum = 5
    env = create_env(curriculum_phase=curriculum_phase)
    print(f"[Curriculum] Phase {curriculum_phase}: starting from {curriculum_heights[curriculum_phase]}cm")

    # ====== 5. Warm-start replay buffer ======
    if args.warm_start_steps > 0:
        warm_start_with_bc(agent, bc_policy, env, device, args.warm_start_steps, curriculum_phase)

    # ====== 6. 断点恢复(如果指定 --resume)======
    start_step = 0
    episode_num = 0
    reward_history = []
    recent_successes = []
    best_success_rate = 0.0
    ckpt_dir = os.path.join(args.save_dir, 'checkpoint')

    if args.resume:
        # 使用用户指定的 checkpoint 目录
        resume_dir = args.resume
        load_buffer = args.save_buffer
        start_step, episode_num, reward_history, recent_successes, best_success_rate = \
            agent.load_checkpoint(resume_dir, load_buffer=load_buffer)
        print(f"[Resume] Resuming from step {start_step}, episode {episode_num}")
        # recent_successes 里存的是 int,保证是 list of int
        recent_successes = [int(x) for x in recent_successes]
        ckpt_dir = resume_dir

    # ====== 7. 训练循环 ======
    state, info = env.reset()
    if isinstance(state, tuple):
        state = state[0]

    episode_reward = 0
    episode_steps = 0

    # 如果恢复时不在 episode 边界,补全该 episode 的累计值
    # (恢复后 step 从 start_step 开始,但 episode 内状态是连续的)

    for step in range(start_step, args.total_steps):

        # --- 选择动作(训练模式:采样) ---
        action = agent.select_action(state, sample=True)

        # --- 环境步进(gymnasium 返回 5 个值) ---
        next_state, env_reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- 存储 transition ---
        agent.store_transition(state, action, env_reward, next_state, done)

        episode_reward += env_reward
        episode_steps += 1
        state = next_state

        # --- SAC 更新 ---
        if step >= args.learning_starts:
            train_info = agent.update()

        # --- episode 结束处理 ---
        if done:
            episode_num += 1
            is_success = info.get('num_success', 0) if isinstance(info, dict) else 0
            recent_successes.append(is_success)
            if len(recent_successes) > 100:
                recent_successes.pop(0)
            success_rate = np.mean(recent_successes) if recent_successes else 0.0

            reward_history.append(episode_reward)
            if len(reward_history) > 500:
                reward_history.pop(0)

            if episode_num % args.log_interval == 0:
                print(f"[SAC] Episode {episode_num:4d} | Step {step:6d} | "
                      f"EpReward: {episode_reward:+8.2f} | "
                      f"EpLen: {episode_steps:4d} | "
                      f"Success(100ep): {success_rate:.3f} | "
                      f"Alpha: {agent.alpha:.3f} | "
                      f"Curr: {curriculum_phase}({curriculum_heights[curriculum_phase]}cm)")

            # 保存 best success checkpoint
            if success_rate > best_success_rate and success_rate > 0:
                best_success_rate = success_rate
                agent.save(os.path.join(args.save_dir, f'sac_success_{success_rate:.3f}.pt'))

            # ---- 课程学习：成功率达标后提升难度 ----
            # 需要至少 30 个 episode + 成功率 > 20% 才升级
            if (curriculum_phase < max_curriculum
                    and len(recent_successes) >= 30
                    and success_rate > 0.2):
                curriculum_phase += 1
                env.close()
                env = create_env(curriculum_phase=curriculum_phase)
                h = curriculum_heights[curriculum_phase]
                print(f"\n{'='*60}")
                print(f"[Curriculum] ADVANCED to Phase {curriculum_phase}: starting from {h}cm")
                print(f"{'='*60}\n")
                recent_successes = []  # 重置成功率统计

                # === 清空旧 buffer，用新难度经验重新填充 ===
                # 旧 phase 的经验对新难度是误导性的（1cm 一步到位的策略对 1.5cm 无效）
                from imitation_pipeline.rl.sac import ReplayBuffer
                agent.buffer = ReplayBuffer(args.buffer_capacity, obs_dim, action_dim, device)
                refresh_steps = max(args.warm_start_steps, 2000)
                print(f"[Curriculum] Buffer cleared. Adding {refresh_steps} steps of Phase {curriculum_phase} experience...")
                warm_start_with_bc(agent, bc_policy, env, device, refresh_steps, curriculum_phase)
                print(f"[Curriculum] Buffer size after refresh: {agent.buffer.size}")

            # 重置环境
            state, info = env.reset()
            if isinstance(state, tuple):
                state = state[0]
            episode_reward = 0
            episode_steps = 0

        # --- 定期保存完整 checkpoint(可续训) ---
        if step > 0 and step % args.ckpt_save_interval == 0:
            agent.save_checkpoint(
                ckpt_dir, step, episode_num,
                reward_history, recent_successes,
                best_success_rate,
                save_buffer=args.save_buffer,
            )

        # 定期保存网络权重(轻量,方便 eval)
        if step > 0 and step % args.save_interval == 0:
            agent.save(os.path.join(args.save_dir, f'sac_checkpoint_step_{step}.pt'))

    # ====== 训练结束 ======
    # 最后保存一次完整 checkpoint
    agent.save_checkpoint(
        ckpt_dir, args.total_steps, episode_num,
        reward_history, recent_successes,
        best_success_rate,
        save_buffer=args.save_buffer,
    )
    agent.save(os.path.join(args.save_dir, 'sac_final.pt'))
    env.close()
    print(f"[SAC] Training done! Final model saved to: {args.save_dir}")
    return agent


def eval(args):
    """评估训练好的 SAC policy"""
    device = torch.device(args.device)

    # 加载维度信息
    demo_data = load_demo_pkl(args.demo_path)
    obs_dim = demo_data['states'].shape[1]
    action_dim = demo_data['actions'].shape[1]

    # 初始化 agent 但不带 reward model(评估只用环境)
    agent = SACAgent(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=args.hidden_dim,
        device=device,
    )
    agent.load(args.checkpoint)
    agent.policy.eval()

    env = create_env()
    if hasattr(env, 'renders'):
        # 如果可能,打开渲染
        pass

    successes = []
    for ep in range(args.eval_episodes):
        state, info = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        ep_reward = 0
        ep_steps = 0
        done = False

        while not done:
            action = agent.select_action(state, sample=False)  # deterministic
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1

        is_success = info.get('num_success', 0) if isinstance(info, dict) else 0
        successes.append(is_success)
        print(f"[Eval] Episode {ep+1}: reward={ep_reward:.2f}, steps={ep_steps}, success={bool(is_success)}")

    success_rate = np.mean(successes)
    print(f"\n[Eval] Success rate over {args.eval_episodes} episodes: {success_rate:.3f}")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="SAC with Learned Reward")
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval'])

    # 数据路径
    parser.add_argument('--demo-path', type=str, default='demo_buffer.pkl')
    parser.add_argument('--reward-model-path', type=str,
                        default='imitation_pipeline/reward_model/reward_model.pt')
    parser.add_argument('--bc-policy-path', type=str,
                        default='imitation_pipeline/bc/bc_policy.pt')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default='imitation_pipeline/rl/checkpoints')

    # Reward 混合权重
    parser.add_argument('--reward-alpha', type=float, default=1.0,
                        help='环境奖励权重(默认1.0,纯环境奖励)')
    parser.add_argument('--reward-beta', type=float, default=0.0,
                        help='Learned reward 权重(默认0.0,关闭learned reward)')

    # SAC 超参数
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--buffer-capacity', type=int, default=100000)

    # 训练参数
    parser.add_argument('--total-steps', type=int, default=200000)
    parser.add_argument('--learning-starts', type=int, default=1000)
    parser.add_argument('--warm-start-steps', type=int, default=200)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--save-interval', type=int, default=20000)

    # Checkpoint / 断点续训
    parser.add_argument('--resume', type=str, default=None,
                        help='checkpoint 目录路径,用于断点续训')
    parser.add_argument('--ckpt-save-interval', type=int, default=5000,
                        help='每多少步保存一次完整 checkpoint (含 optimizer)')
    parser.add_argument('--save-buffer', action='store_true',
                        help='在 checkpoint 中同时保存 replay buffer(较占空间)')

    # 评估
    parser.add_argument('--eval-episodes', type=int, default=10)

    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.mode == 'train':
        train_with_learned_reward(args)
    elif args.mode == 'eval':
        if args.checkpoint is None:
            args.checkpoint = os.path.join(args.save_dir, 'sac_final.pt')
        eval(args)


if __name__ == '__main__':
    main()
