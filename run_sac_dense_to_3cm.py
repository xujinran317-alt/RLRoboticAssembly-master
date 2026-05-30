"""
run_sac_dense_to_3cm.py - SAC Dense baseline 从 1.5cm 训练到 3.0cm

课程: 1.5cm → 1.7cm → 2.0cm → 2.5cm → 3.0cm
用法:
  python run_sac_dense_to_3cm.py           # 训练
  python run_sac_dense_to_3cm.py --eval    # 评估
"""
import os
import sys
import subprocess
import json
import time
import argparse
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'


def run(cmd, desc):
    print(f"\n{'='*70}")
    print(f"[RUN] {desc}")
    print(f"{'='*70}")
    print(f"$ {cmd}")
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, cwd=PROJECT_DIR)
    elapsed = time.time() - t0
    if r.returncode != 0:
        print(f"[FAIL] {desc} (code {r.returncode}, {elapsed:.0f}s)")
        return False
    print(f"[OK] {desc} ({elapsed:.0f}s)")
    return True


def train(args):
    """从 1.5cm 最佳权重开始，走课程到 3.0cm"""
    # 用 sac_final.pt（1.5cm 最终模型）作为起点
    load_weights = os.path.join(PROJECT_DIR, "imitation_pipeline/rl/checkpoints_dense/sac_final.pt")
    if not os.path.exists(load_weights):
        print(f"[ERROR] 找不到 1.5cm 权重: {load_weights}")
        print("请先跑 SAC Dense 1.5cm 训练")
        return False

    save_dir = "imitation_pipeline/rl/checkpoints_dense_3cm"

    cmd = (
        f"python -m imitation_pipeline.rl.train_sac_with_learned_reward "
        f"--mode train "
        f"--demo-path demo_buffer.pkl "
        f"--reward-alpha 1.0 "
        f"--reward-beta 0.0 "
        f"--total-steps {args.total_steps} "
        f"--warm-start-steps {args.warm_start_steps} "
        f"--start-phase 2 "
        f"--load-weights {load_weights} "
        f"--device {DEVICE} "
        f"--save-dir {save_dir} "
        f"--log-interval 20"
    )
    return run(cmd, "SAC Dense 训练 1.5cm→3.0cm (课程自动升级)")


def eval_at_3cm(args):
    """评估 SAC Dense 在 3.0cm 的表现"""
    # 找最佳 checkpoint
    save_dir = os.path.join(PROJECT_DIR, "imitation_pipeline/rl/checkpoints_dense_3cm")
    
    # 优先用 success rate 最高的，否则用 final
    best_ckpt = None
    best_sr = 0.0
    if os.path.exists(save_dir):
        for f in os.listdir(save_dir):
            if f.startswith("sac_success_") and f.endswith(".pt"):
                sr = float(f.replace("sac_success_", "").replace(".pt", ""))
                if sr > best_sr:
                    best_sr = sr
                    best_ckpt = os.path.join(save_dir, f)
    
    if best_ckpt is None:
        best_ckpt = os.path.join(save_dir, "sac_final.pt")
    
    if not os.path.exists(best_ckpt):
        print(f"[ERROR] 找不到 checkpoint: {best_ckpt}")
        return False

    print(f"[Eval] 使用 checkpoint: {best_ckpt}")

    script = f'''
import os, sys, numpy as np, torch
sys.path.insert(0, r"{PROJECT_DIR}")
from assembly_env.sim_task import AssemblySimEnv
from imitation_pipeline.rl.sac import SACAgent
from imitation_pipeline.utils import load_demo_pkl

def make_env(phase):
    return AssemblySimEnv(
        env_robot=None, self_collision_enabled=True, renders=False,
        ft_noise=False, pose_noise=False, action_noise=False, physical_noise=False,
        time_step=1/250, max_steps=500, step_limit=True, action_dim=6,
        max_vel=0.05, max_rad=0.02, ft_obs_only=False, limit_ft=False,
        max_ft=[1000,1000,2500,100,100,100], max_position_range=[2]*3,
        dist_threshold=0.01, use_shaped_reward=True,
        curriculum_phase=phase,
    )

device = "{DEVICE}"
data = load_demo_pkl("demo_buffer.pkl")
obs_dim, action_dim = data['states'].shape[1], data['actions'].shape[1]

agent = SACAgent(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=256, device=device)
agent.load(r"{best_ckpt}")
agent.policy.eval()

# 评估 3.0cm (phase 6)
env = make_env(6)
succs, lens_success, lens_all = [], [], []
for ep in range(100):
    s, info = env.reset()
    if isinstance(s, tuple): s = s[0]
    done, steps = False, 0
    while not done:
        a = agent.select_action(s, sample=False)
        s, r, term, trunc, info = env.step(a)
        done = term or trunc; steps += 1
    is_succ = int(bool(info.get('num_success', 0)))
    succs.append(is_succ)
    lens_all.append(steps)
    if is_succ:
        lens_success.append(steps)
    if (ep+1) % 20 == 0:
        print(f"  [SAC Dense] 3.0cm ep {{ep+1}}: running SR={{np.mean(succs)*100:.1f}}%")

sr = np.mean(succs)
avg_len = np.mean(lens_all)
avg_len_succ = np.mean(lens_success) if lens_success else None
print(f"\\n[SAC Dense @ 3.0cm] Success Rate: {{sr*100:.1f}}%  ({{sum(succs)}}/{{len(succs)}})")
print(f"[SAC Dense @ 3.0cm] Avg Episode Length: {{avg_len:.1f}}")
if avg_len_succ:
    print(f"[SAC Dense @ 3.0cm] Avg Length (success only): {{avg_len_succ:.1f}}")

result = {{
    "method": "SAC_Dense",
    "distance": "3.0cm",
    "success_rate": float(sr),
    "avg_episode_length": float(avg_len),
    "avg_episode_length_success": float(avg_len_succ) if avg_len_succ else None,
    "num_success": int(sum(succs)),
    "n_episodes": len(succs),
    "checkpoint": r"{best_ckpt}",
}}
with open(r"{PROJECT_DIR}\\sac_dense_3cm_results.json", "w") as f:
    import json
    json.dump(result, f, indent=2)
print("[Saved] sac_dense_3cm_results.json")
env.close()
'''

    script_path = os.path.join(PROJECT_DIR, '_tmp_eval_dense_3cm.py')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script)
    ok = run(f"python {script_path}", "评估 SAC Dense @ 3.0cm (100 episodes)")
    os.remove(script_path)
    return ok


def main():
    parser = argparse.ArgumentParser(description="SAC Dense baseline → 3.0cm")
    parser.add_argument('--eval', action='store_true', help='只评估（跳过训练）')
    parser.add_argument('--total-steps', type=int, default=500000)
    parser.add_argument('--warm-start-steps', type=int, default=30000)
    args = parser.parse_args()

    if args.eval:
        eval_at_3cm(args)
    else:
        ok = train(args)
        if ok:
            eval_at_3cm(args)


if __name__ == '__main__':
    main()
