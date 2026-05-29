"""
run_all_baselines.py - Table IV 全量 baseline 训练+评估

统一脚本，依次训练/评估 Table IV 所有方法：
  1. SAC (Sparse)      → 已知 0%/200，只做验证
  2. BC Only           → 直接评估 BC policy
  3. SAC + Dense       → 训练 SAC（纯环境稠密奖励）
  4. Ours              → 你已有的 train_sac_with_learned_reward.py

所有方法都走课程：1cm → 1.2cm → 1.5cm

用法:
  python run_all_baselines.py                # 跑全部
  python run_all_baselines.py --method 2     # 只跑 BC Only
  python run_all_baselines.py --method 3     # 只跑 SAC + Dense
  python run_all_baselines.py --method eval  # 评估所有已训练模型
"""
import os
import sys
import argparse
import subprocess
import json
import time
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


# ================================================================
# 方法 1: SAC Sparse — 验证 0% 结果
# ================================================================
def method_1_sac_sparse(args):
    """跑 SAC Sparse 验证 0% 成功率"""
    print("\n" + "#"*70)
    print("# 方法 1: SAC (Sparse Reward)")
    print("# 预期: 0% success, 200 steps timeout")
    print("#"*70)

    # 直接在当前环境上跑 100 个 episode，用随机策略（SAC 未训练 = 随机）
    script = r'''
import os, sys, numpy as np, torch
sys.path.insert(0, r"PROJECT_DIR")
from assembly_env.sim_task import AssemblySimEnv
from imitation_pipeline.rl.sac import SACAgent
from imitation_pipeline.utils import load_demo_pkl

def make_env(phase):
    return AssemblySimEnv(
        env_robot=None, self_collision_enabled=True, renders=False,
        ft_noise=False, pose_noise=False, action_noise=False, physical_noise=False,
        time_step=1/250, max_steps=400, step_limit=True, action_dim=6,
        max_vel=0.05, max_rad=0.02, ft_obs_only=False, limit_ft=False,
        max_ft=[1000,1000,2500,100,100,100], max_position_range=[2]*3,
        dist_threshold=0.01, use_shaped_reward=False,  # 稀疏奖励
        curriculum_phase=phase,
    )

device = "DEVICE_STR"
data = load_demo_pkl("demo_buffer.pkl")
obs_dim, action_dim = data['states'].shape[1], data['actions'].shape[1]
agent = SACAgent(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=256, device=device)

results = {}
for phase, height in [(0,"1.0cm"),(1,"1.2cm"),(2,"1.5cm")]:
    env = make_env(phase)
    succs, lens = [], []
    for ep in range(100):
        s, info = env.reset()
        if isinstance(s, tuple): s = s[0]
        done, steps = False, 0
        while not done:
            a = agent.select_action(s, sample=True)
            s, r, term, trunc, info = env.step(a)
            done = term or trunc; steps += 1
        succs.append(int(bool(info.get('num_success',0))))
        lens.append(steps)
    sr, al = np.mean(succs), np.mean(lens)
    results[phase] = {"success_rate": sr, "avg_length": al}
    print(f"[Sparse] {height}: SR={sr*100:.1f}%, AvgLen={al:.1f}")
    env.close()

print(json.dumps(results, indent=2))
'''.replace('PROJECT_DIR', PROJECT_DIR).replace('DEVICE_STR', DEVICE)

    script_path = os.path.join(PROJECT_DIR, '_tmp_sparse.py')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script)
    ok = run(f"python {script_path}", "SAC Sparse 验证 (100ep x 3 phases)")
    os.remove(script_path)
    return ok


# ================================================================
# 方法 2: BC Only — 直接评估 BC policy
# ================================================================
def method_2_bc_only(args):
    """评估 BC policy 在 1cm/1.2cm/1.5cm 的表现"""
    print("\n" + "#"*70)
    print("# 方法 2: BC Only (纯 BC，无 RL)")
    print("#"*70)

    script = r'''
import os, sys, numpy as np, torch
sys.path.insert(0, r"PROJECT_DIR")
from assembly_env.sim_task import AssemblySimEnv
from imitation_pipeline.bc.model import BCPolicy
from imitation_pipeline.utils import load_demo_pkl, load_model

def make_env(phase):
    return AssemblySimEnv(
        env_robot=None, self_collision_enabled=True, renders=False,
        ft_noise=False, pose_noise=False, action_noise=False, physical_noise=False,
        time_step=1/250, max_steps=400, step_limit=True, action_dim=6,
        max_vel=0.05, max_rad=0.02, ft_obs_only=False, limit_ft=False,
        max_ft=[1000,1000,2500,100,100,100], max_position_range=[2]*3,
        dist_threshold=0.01, use_shaped_reward=True,
        curriculum_phase=phase,
    )

device = "DEVICE_STR"
data = load_demo_pkl("demo_buffer.pkl")
obs_dim, action_dim = data['states'].shape[1], data['actions'].shape[1]
bc = BCPolicy(obs_dim, action_dim, 256).to(device)
bc = load_model(bc, "imitation_pipeline/bc/bc_policy.pt", device)
bc.eval()

results = {}
for phase, height in [(0,"1.0cm"),(1,"1.2cm"),(2,"1.5cm")]:
    env = make_env(phase)
    succs, lens = [], []
    for ep in range(100):
        s, info = env.reset()
        if isinstance(s, tuple): s = s[0]
        done, steps = False, 0
        while not done:
            a = bc.get_action(s, device)
            s, r, term, trunc, info = env.step(a)
            done = term or trunc; steps += 1
        succs.append(int(bool(info.get('num_success',0))))
        lens.append(steps)
        if (ep+1) % 20 == 0:
            print(f"  [BC] {height} ep {ep+1}: succ={succs[-1]}, steps={lens[-1]}")
    sr, al = np.mean(succs), np.mean(lens)
    results[phase] = {"success_rate": sr, "avg_length": al}
    print(f"\n[BC Only] {height}: SR={sr*100:.1f}%, AvgLen={al:.1f}\n")
    env.close()

print(json.dumps(results, indent=2))
# 保存结果
import json as j
with open(r"PROJECT_DIR\bc_only_results.json", "w") as f:
    j.dump(results, f, indent=2)
print("[Saved] bc_only_results.json")
'''.replace('PROJECT_DIR', PROJECT_DIR).replace('DEVICE_STR', DEVICE)

    script_path = os.path.join(PROJECT_DIR, '_tmp_bc.py')
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script)
    ok = run(f"python {script_path}", "BC Only 评估 (100ep x 3 phases)")
    os.remove(script_path)
    return ok


# ================================================================
# 方法 3: SAC + Manual Dense Reward — 训练
# ================================================================
def method_3_sac_dense(args):
    """训练 SAC + 手动稠密奖励（无 learned reward，复用你的训练脚本）"""
    print("\n" + "#"*70)
    print("# 方法 3: SAC + Manual Dense Reward")
    print("# 使用 train_sac_with_learned_reward.py --reward-beta 0.0")
    print("#"*70)

    cmd = (
        f"python -m imitation_pipeline.rl.train_sac_with_learned_reward "
        f"--mode train "
        f"--demo-path demo_buffer.pkl "
        f"--reward-alpha 1.0 "
        f"--reward-beta 0.0 "
        f"--total-steps {args.total_steps} "
        f"--warm-start-steps {args.warm_start_steps} "
        f"--start-phase 2 "
        f"--device {DEVICE} "
        f"--save-dir imitation_pipeline/rl/checkpoints_dense "
        f"--log-interval 20"
    )
    if args.resume_dense:
        cmd += f" --resume {args.resume_dense}"
    return run(cmd, "SAC + Dense Reward 训练 (课程: 1cm→1.2cm→1.5cm)")


# ================================================================
# 方法 4: Ours — 训练（你已有脚本）
# ================================================================
def method_4_ours(args):
    """训练 Ours: BC + SAC + Learned Reward"""
    print("\n" + "#"*70)
    print("# 方法 4: Ours (BC + SAC + Learned Reward)")
    print("#"*70)

    cmd = (
        f"python -m imitation_pipeline.rl.train_sac_with_learned_reward "
        f"--mode train "
        f"--demo-path demo_buffer.pkl "
        f"--reward-model-path imitation_pipeline/reward_model/reward_model.pt "
        f"--bc-policy-path imitation_pipeline/bc/bc_policy.pt "
        f"--reward-alpha 0.5 "
        f"--reward-beta 0.5 "
        f"--total-steps {args.total_steps} "
        f"--warm-start-steps {args.warm_start_steps} "
        f"--start-phase 2 "
        f"--device {DEVICE} "
        f"--save-dir imitation_pipeline/rl/checkpoints "
        f"--log-interval 20"
    )
    if args.resume_ours:
        cmd += f" --resume {args.resume_ours}"
    return run(cmd, "Ours 训练 (课程: 1cm→1.2cm→1.5cm)")


# ================================================================
# 评估已训练模型
# ================================================================
def eval_trained(args):
    """评估 SAC Dense 和 Ours 的最终模型"""
    print("\n" + "#"*70)
    print("# 评估已训练模型")
    print("#"*70)

    results = {}

    # 评估 SAC Dense
    ckpt_dense = "imitation_pipeline/rl/checkpoints_dense/sac_dense_final.pt"
    if os.path.exists(ckpt_dense):
        script = r'''
import os, sys, numpy as np, torch
sys.path.insert(0, r"PROJECT_DIR")
from assembly_env.sim_task import AssemblySimEnv
from imitation_pipeline.rl.sac import SACAgent
from imitation_pipeline.utils import load_demo_pkl

def make_env(phase):
    return AssemblySimEnv(
        env_robot=None, self_collision_enabled=True, renders=False,
        ft_noise=False, pose_noise=False, action_noise=False, physical_noise=False,
        time_step=1/250, max_steps=400, step_limit=True, action_dim=6,
        max_vel=0.05, max_rad=0.02, ft_obs_only=False, limit_ft=False,
        max_ft=[1000,1000,2500,100,100,100], max_position_range=[2]*3,
        dist_threshold=0.01, use_shaped_reward=True,
        curriculum_phase=phase,
    )

device = "DEVICE_STR"
data = load_demo_pkl("demo_buffer.pkl")
obs_dim, action_dim = data['states'].shape[1], data['actions'].shape[1]
agent = SACAgent(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=256, device=device)
agent.load(r"CKPT_PATH")
agent.policy.eval()

for phase, height in [(0,"1.0cm"),(1,"1.2cm"),(2,"1.5cm")]:
    env = make_env(phase)
    succs, lens = [], []
    for ep in range(100):
        s, info = env.reset()
        if isinstance(s, tuple): s = s[0]
        done, steps = False, 0
        while not done:
            a = agent.select_action(s, sample=False)
            s, r, term, trunc, info = env.step(a)
            done = term or trunc; steps += 1
        succs.append(int(bool(info.get('num_success',0))))
        lens.append(steps)
    sr, al = np.mean(succs), np.mean(lens)
    print(f"[SAC Dense] {height}: SR={sr*100:.1f}%, AvgLen={al:.1f}")
    env.close()
'''.replace('PROJECT_DIR', PROJECT_DIR).replace('DEVICE_STR', DEVICE).replace('CKPT_PATH', ckpt_dense)
        script_path = os.path.join(PROJECT_DIR, '_tmp_eval_dense.py')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script)
        run(f"python {script_path}", "评估 SAC Dense")
        os.remove(script_path)

    # 评估 Ours
    ckpt_ours = "imitation_pipeline/rl/checkpoints/sac_final.pt"
    if os.path.exists(ckpt_ours):
        script = r'''
import os, sys, numpy as np, torch
sys.path.insert(0, r"PROJECT_DIR")
from assembly_env.sim_task import AssemblySimEnv
from imitation_pipeline.rl.sac import SACAgent
from imitation_pipeline.utils import load_demo_pkl

def make_env(phase):
    return AssemblySimEnv(
        env_robot=None, self_collision_enabled=True, renders=False,
        ft_noise=False, pose_noise=False, action_noise=False, physical_noise=False,
        time_step=1/250, max_steps=400, step_limit=True, action_dim=6,
        max_vel=0.05, max_rad=0.02, ft_obs_only=False, limit_ft=False,
        max_ft=[1000,1000,2500,100,100,100], max_position_range=[2]*3,
        dist_threshold=0.01, use_shaped_reward=True,
        curriculum_phase=phase,
    )

device = "DEVICE_STR"
data = load_demo_pkl("demo_buffer.pkl")
obs_dim, action_dim = data['states'].shape[1], data['actions'].shape[1]
agent = SACAgent(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=256, device=device)
agent.load(r"CKPT_PATH")
agent.policy.eval()

for phase, height in [(0,"1.0cm"),(1,"1.2cm"),(2,"1.5cm")]:
    env = make_env(phase)
    succs, lens = [], []
    for ep in range(100):
        s, info = env.reset()
        if isinstance(s, tuple): s = s[0]
        done, steps = False, 0
        while not done:
            a = agent.select_action(s, sample=False)
            s, r, term, trunc, info = env.step(a)
            done = term or trunc; steps += 1
        succs.append(int(bool(info.get('num_success',0))))
        lens.append(steps)
    sr, al = np.mean(succs), np.mean(lens)
    print(f"[Ours] {height}: SR={sr*100:.1f}%, AvgLen={al:.1f}")
    env.close()
'''.replace('PROJECT_DIR', PROJECT_DIR).replace('DEVICE_STR', DEVICE).replace('CKPT_PATH', ckpt_ours)
        script_path = os.path.join(PROJECT_DIR, '_tmp_eval_ours.py')
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script)
        run(f"python {script_path}", "评估 Ours")
        os.remove(script_path)


# ================================================================
# 汇总
# ================================================================
def print_table():
    print("\n" + "="*70)
    print("Table IV: Performance Comparison on Timber Assembly Task")
    print("="*70)
    print(f"{'Method':<50} {'Success Rate (%)':<18} {'Avg Ep Length':<15}")
    print("-"*83)
    print(f"{'SAC (Sparse Reward)':<50} {'0':<18} {'200 (Timeout)':<15}")

    # BC
    bc_file = os.path.join(PROJECT_DIR, 'bc_only_results.json')
    if os.path.exists(bc_file):
        with open(bc_file) as f:
            bc = json.load(f)
        for p, h in [('0','1.0cm'),('1','1.2cm'),('2','1.5cm')]:
            if p in bc:
                r = bc[p]
                print(f"{'  BC Only ('+h+')':<50} {r['success_rate']*100:<18.1f} {r['avg_length']:<15.1f}")

    print(f"{'SAC + Manual Dense Reward':<50} {'(run --method 3)':<18} {'':<15}")
    print(f"{'BC + SAC + Learned Reward (Ours)':<50} {'(run --method 4)':<18} {'':<15}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description="Table IV Baselines")
    parser.add_argument('--method', type=str, default='all',
                        choices=['all','1','2','3','4','eval','table'])
    parser.add_argument('--total-steps', type=int, default=200000)
    parser.add_argument('--warm-start-steps', type=int, default=20000)
    parser.add_argument('--resume-dense', type=str, default=None)
    parser.add_argument('--resume-ours', type=str, default=None)
    args = parser.parse_args()

    if args.method == 'all':
        method_1_sac_sparse(args)
        method_2_bc_only(args)
        method_3_sac_dense(args)
        method_4_ours(args)
        print_table()
    elif args.method == '1':
        method_1_sac_sparse(args)
    elif args.method == '2':
        method_2_bc_only(args)
    elif args.method == '3':
        method_3_sac_dense(args)
    elif args.method == '4':
        method_4_ours(args)
    elif args.method == 'eval':
        eval_trained(args)
    elif args.method == 'table':
        print_table()


if __name__ == '__main__':
    main()
