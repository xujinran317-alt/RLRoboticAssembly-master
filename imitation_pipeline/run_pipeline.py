"""
run_pipeline.py - 完整三阶段 Pipeline 运行脚本

一键运行：Behavior Cloning → Reward Model → SAC Fine-tuning

用法：
  python -m imitation_pipeline.run_pipeline [--skip-bc] [--skip-reward]
"""
import os
import sys
import argparse
import subprocess


def run_command(cmd, desc):
    """运行命令并打印输出"""
    print("\n" + "=" * 70)
    print(f"[Pipeline] {desc}")
    print("=" * 70)
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"[ERROR] Step failed with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"[Pipeline] Done: {desc}\n")


def main():
    parser = argparse.ArgumentParser(description="Run the full imitation + RL pipeline")
    parser.add_argument('--skip-bc', action='store_true', help='Skip BC training')
    parser.add_argument('--skip-reward', action='store_true', help='Skip reward model training')
    parser.add_argument('--demo-path', type=str, default='demo_buffer.pkl')
    parser.add_argument('--device', type=str, default='cuda' if __import__("torch").cuda.is_available() else 'cpu')
    
    # BC 参数
    parser.add_argument('--bc-epochs', type=int, default=100)
    parser.add_argument('--bc-lr', type=float, default=1e-3)
    
    # Reward 参数
    parser.add_argument('--reward-epochs', type=int, default=100)
    parser.add_argument('--reward-lr', type=float, default=1e-3)
    parser.add_argument('--neg-ratio', type=float, default=2.0)
    
    # SAC 参数
    parser.add_argument('--total-steps', type=int, default=300000)
    parser.add_argument('--reward-alpha', type=float, default=0.5)
    parser.add_argument('--reward-beta', type=float, default=0.5)
    parser.add_argument('--skip-sac', action='store_true', help='Skip SAC training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint 目录路径，用于断点续训 SAC')
    parser.add_argument('--save-buffer', action='store_true',
                        help='在 checkpoint 中同时保存 replay buffer')
    parser.add_argument('--ckpt-save-interval', type=int, default=5000,
                        help='每多少步保存一次完整 checkpoint (含 optimizer)')
    
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)
    os.chdir(project_dir)
    
    print(f"[Pipeline] Working directory: {os.getcwd()}")
    print(f"[Pipeline] Device: {args.device}")
    print(f"[Pipeline] Demo path: {args.demo_path}")
    
    # ====== Phase 1: Behavior Cloning ======
    if not args.skip_bc:
        cmd = (
            f"python -m imitation_pipeline.bc.train_bc "
            f"--demo-path {args.demo_path} "
            f"--epochs {args.bc_epochs} "
            f"--lr {args.bc_lr} "
            f"--device {args.device} "
            f"--save-path {base_dir}/bc/bc_policy.pt"
        )
        run_command(cmd, "Phase 1: Behavior Cloning")
    else:
        print("[Pipeline] Skipping BC training")
    
    # ====== Phase 2: Reward Model ======
    if not args.skip_reward:
        cmd = (
            f"python -m imitation_pipeline.reward_model.train_reward "
            f"--demo-path {args.demo_path} "
            f"--epochs {args.reward_epochs} "
            f"--lr {args.reward_lr} "
            f"--neg-ratio {args.neg_ratio} "
            f"--device {args.device} "
            f"--save-path {base_dir}/reward_model/reward_model.pt"
        )
        run_command(cmd, "Phase 2: Reward Model Training")
    else:
        print("[Pipeline] Skipping Reward Model training")
    
    # ====== Phase 3: SAC Training with Hybrid Reward (env + learned) ======
    if not args.skip_sac:
        # SAC 使用混合奖励：alpha * env_reward + beta * learned_reward
        # 默认 alpha=0.5, beta=0.5，环境信号和模仿信号各占一半
        cmd = (
            f"python -m imitation_pipeline.rl.train_sac_with_learned_reward "
            f"--mode train "
            f"--demo-path {args.demo_path} "
            f"--reward-model-path {base_dir}/reward_model/reward_model.pt "
            f"--bc-policy-path {base_dir}/bc/bc_policy.pt "
            f"--reward-alpha {args.reward_alpha} "
            f"--reward-beta {args.reward_beta} "
            f"--total-steps {args.total_steps} "
            f"--device {args.device} "
            f"--save-dir {base_dir}/rl/checkpoints"
        )
        if args.resume:
            cmd += f" --resume {args.resume}"
        if args.save_buffer:
            cmd += " --save-buffer"
        if args.ckpt_save_interval:
            cmd += f" --ckpt-save-interval {args.ckpt_save_interval}"
        run_command(cmd, "Phase 3: SAC Training (Hybrid Reward)")
    else:
        print("[Pipeline] Skipping SAC training")
    
    print("\n" + "=" * 70)
    print("[Pipeline] All phases complete!")
    print("=" * 70)
    
    # 打印文件结构
    print("\nGenerated files:")
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            fp = os.path.join(root, f)
            print(f"  {fp}")


if __name__ == '__main__':
    main()
