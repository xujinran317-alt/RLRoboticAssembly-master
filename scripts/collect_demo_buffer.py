"""
collect_demo_buffer.py - 收集演示数据，预热 replay buffer
===========================================================

使用 demo_auto_assembly_simple.py 的比例控制逻辑，
生成 (obs, action, reward, next_obs, done) 五元组序列，
保存到 pickle 文件供训练前预热经验回放池。

用法:
    python scripts/collect_demo_buffer.py
    python scripts/collect_demo_buffer.py -n 5 -o demo_buffer.pkl --noise
"""

import argparse
import math
import os
import pickle
import random
import sys
import time
from collections import deque
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pybullet as p
import transforms3d

from assembly_env.robots.sim_robotless import INITIAL_POS, INITIAL_ORN
from utils.transforms import (
    xyzw_to_wxyz, wxyz_to_xyzw,
    mat44_by_pos_quat, mat44_to_pos_quat,
    get_f1_to_f2_xform, transform_mat,
)
from utils.io_utils import pr_green, pr_red, pr_info, format_urdf_filepath

# ---- 控制参数（与 demo_auto_assembly_simple.py 一致）----
KP_POS = 3.0
KP_ORN = 2.5
MAX_VEL = 0.2
MAX_RAD = 0.2
TIME_STEP = 0.004
DIST_THRESHOLD = 0.01
ANGLE_THRESHOLD = 0.1
MAX_STEPS = 2000
TRAIN_MAX_VEL = 0.01
TRAIN_MAX_RAD = 0.01

def quat_error(q_current, q_target):
    '''计算四元数误差'''
    qc_wxyz = xyzw_to_wxyz(q_current)
    qt_wxyz = xyzw_to_wxyz(q_target)
    q_current_inv = transforms3d.quaternions.qinverse(qc_wxyz)
    q_rel = transforms3d.quaternions.qmult(qt_wxyz, q_current_inv)
    error_rot = 2.0 * np.array(q_rel[1:4])
    if q_rel[0] < 0:
        error_rot = -error_rot
    return error_rot


def build_obs(pos, orn, ft, action_dim):
    '''组装观测向量（与训练环境一致）'''
    if action_dim > 3:
        return np.array(list(pos) + list(orn) + list(ft), dtype=np.float32)
    else:
        return np.array(list(pos) + list(ft), dtype=np.float32)

def run_one_demo(action_dim=6, ox=0.05, oy=0.04, oz=0.0,
                 add_noise=False, headless=False, verbose=True):
    '''跑一次演示，收集 (obs, action, reward, next_obs, done) 序列'''
    memory = deque()
    if headless:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(0.6, 180, -41, [0, 0, 0])
    p.setPhysicsEngineParameter(numSolverIterations=150, enableFileCaching=0)
    p.setTimeStep(TIME_STEP)
    p.setGravity(0, 0, 0)

    target_pos = np.array([0.0, 0.0, 0.0])
    target_orn_xyzw = list(transforms3d.euler.euler2quat(0, 0, math.pi, 'sxyz'))
    target_orn_xyzw = wxyz_to_xyzw(target_orn_xyzw)

    nx = ny = nz = 0.0
    if add_noise:
        nx = random.uniform(-0.02, 0.02)
        ny = random.uniform(-0.02, 0.02)
        nz = random.uniform(-0.01, 0.01)
    start_pos = [ox + nx, oy + ny, 0.24 + oz + nz]

    if verbose:
        pr_info('起始位置: ' + str([round(v, 4) for v in start_pos]))

    tool_uid = p.loadURDF(
        format_urdf_filepath('envs/urdf/robotless_lap_joint/tool'),
        basePosition=start_pos, baseOrientation=INITIAL_ORN, useFixedBase=0)
    p.loadURDF(
        format_urdf_filepath('envs/urdf/robotless_lap_joint/task_lap_90deg'),
        basePosition=[0, 0, 0],
        baseOrientation=wxyz_to_xyzw(transforms3d.euler.euler2quat(0, 0, math.pi, 'sxyz')),
        useFixedBase=1)

    base_pose = p.getBasePositionAndOrientation(tool_uid)
    member_pose = p.getLinkState(tool_uid, 2)[4:6]
    from_base_to_member = get_f1_to_f2_xform(base_pose, member_pose)

    for _ in range(10):
        p.stepSimulation()
    time.sleep(0.05)

    step = 0
    success = False
    prev_dist = float('inf')

    while step < MAX_STEPS:
        base_pos, base_orn = p.getBasePositionAndOrientation(tool_uid)
        member_pose_mat = transform_mat(from_base_to_member, mat44_by_pos_quat(base_pos, base_orn))
        current_pos, current_orn = mat44_to_pos_quat(member_pose_mat)
        ft = np.multiply(-0.1, p.getJointState(tool_uid, 0)[2]).tolist()
        obs = build_obs(current_pos, current_orn, ft, action_dim)

        pos_error = np.array(target_pos) - np.array(current_pos)
        dist = np.linalg.norm(pos_error)
        orn_error = quat_error(current_orn, target_orn_xyzw)
        orn_dist = np.linalg.norm(orn_error)

        vel_cmd = KP_POS * pos_error
        vn = np.linalg.norm(vel_cmd)
        if vn > MAX_VEL:
            vel_cmd = vel_cmd / vn * MAX_VEL
        rot_cmd = KP_ORN * orn_error
        rn = np.linalg.norm(rot_cmd)
        if rn > MAX_RAD:
            rot_cmd = rot_cmd / rn * MAX_RAD

        norm_vel = np.clip(vel_cmd / (TRAIN_MAX_VEL * TIME_STEP), -1.0, 1.0)
        if action_dim > 3:
            norm_rot = np.clip(rot_cmd / (TRAIN_MAX_RAD * TIME_STEP), -1.0, 1.0)
            action = np.concatenate([norm_vel, norm_rot]).astype(np.float32)
        else:
            action = norm_vel.astype(np.float32)

        delta_lin = np.array(action[0:3]) * TRAIN_MAX_VEL * TIME_STEP
        delta_rot = np.array(action[3:6]) * TRAIN_MAX_RAD * TIME_STEP if action_dim > 3 else np.zeros(3)
        new_pos = (np.array(base_pos) + delta_lin).tolist()
        qd = wxyz_to_xyzw(transforms3d.quaternions.qmult(
            [0, delta_rot[0], delta_rot[1], delta_rot[2]], xyzw_to_wxyz(base_orn)))
        new_orn = (np.array(base_orn) + 0.5 * np.array(qd)).tolist()
        no = np.linalg.norm(new_orn)
        new_orn = (np.array(new_orn) / no).tolist()
        p.resetBasePositionAndOrientation(tool_uid, new_pos, new_orn)
        p.stepSimulation()
        step += 1

        bp2, bo2 = p.getBasePositionAndOrientation(tool_uid)
        mpm2 = transform_mat(from_base_to_member, mat44_by_pos_quat(bp2, bo2))
        cp2, co2 = mat44_to_pos_quat(mpm2)
        ft2 = np.multiply(-0.1, p.getJointState(tool_uid, 0)[2]).tolist()
        next_obs = build_obs(cp2, co2, ft2, action_dim)

        progress = prev_dist - dist if prev_dist != float('inf') else 0.0
        prev_dist = dist
        reward = -dist * 1.0
        done = False

        dist2 = np.linalg.norm(np.array(target_pos) - np.array(cp2))
        if dist2 < DIST_THRESHOLD and orn_dist < ANGLE_THRESHOLD:
            reward += 100.0
            done = True
            success = True
            if verbose:
                pr_green(f'  OK 成功 {step}步 | 距离 {dist2*1000:.1f}mm')
        if step >= MAX_STEPS:
            done = True
            if verbose:
                pr_red(f'  FAIL 超时 | 最小距离 {dist*1000:.1f}mm')

        memory.append((obs, action, reward, next_obs, done))
        if verbose and step % 400 == 0:
            pr_info(f'  Step {step:4d} | 距离 {dist*1000:.1f}mm | 角度 {orn_dist*180/math.pi:.1f}deg')
        if done:
            break
        time.sleep(TIME_STEP * 0.1)

    p.disconnect()
    return memory, success

def collect_and_save(num_episodes=1, action_dim=6, output_file='demo_buffer.pkl',
                     headless=False, add_noise=False, verbose=True):
    '''多次演示并保存到 pickle 文件'''
    all_episodes = []
    successes = 0
    total_steps = 0

    offsets = [
        (0.05, 0.04, 0.0), (0.06, 0.03, 0.0), (0.04, 0.05, 0.0),
        (0.055, 0.035, 0.0), (0.045, 0.045, 0.0),
        (0.07, 0.02, 0.005), (0.03, 0.06, -0.005),
    ]

    for ep in range(num_episodes):
        if verbose:
            pr_info(f'\n{"="*60}')
            pr_info(f'Episode {ep+1}/{num_episodes}')
            pr_info(f'{"="*60}')
        ox, oy, oz = offsets[ep % len(offsets)]
        memory, ok = run_one_demo(action_dim=action_dim, ox=ox, oy=oy, oz=oz,
                                  add_noise=add_noise, headless=headless, verbose=verbose)
        all_episodes.append(memory)
        total_steps += len(memory)
        if ok:
            successes += 1
        if verbose:
            pr_info(f'  -> {len(memory)} 步, {"OK" if ok else "FAIL"}')

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(all_episodes, f)

    if verbose:
        pr_green(f'\n{"="*60}')
        pr_green('收集完成!')
        pr_green(f'  Episodes: {num_episodes}')
        pr_green(f'  成功率:   {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)')
        pr_green(f'  总步数:   {total_steps}')
        pr_green(f'  保存到:   {os.path.abspath(output_file)}')
        pr_green(f'{"="*60}')
    return all_episodes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='收集演示数据预热 replay buffer')
    parser.add_argument('-n', '--num-episodes', type=int, default=1)
    parser.add_argument('-a', '--action-dim', type=int, default=6, choices=[3, 6])
    parser.add_argument('-o', '--output', type=str, default='demo_buffer.pkl')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--noise', action='store_true')
    parser.add_argument('-q', '--quiet', action='store_true')
    args = parser.parse_args()
    collect_and_save(args.num_episodes, args.action_dim,
                     args.output, args.headless, args.noise,
                     verbose=not args.quiet)
