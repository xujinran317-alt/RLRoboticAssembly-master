"""
运行仿真环境 / 采集人工演示数据

用法:
    assembly run -i xbc --save-demo --demo-path my_demo.pkl
"""

import time

import numpy as np
import pybullet as p

from assembly_env.sim_task import AssemblySimEnv
from human_demo.devices import REGISTRY
from human_demo.recorder import DemoRecorder


def run_env(
    input_type: str = "pbg",
    input_scaling: tuple = (10, 25),
    action_space: int = 6,
    save_demo: bool = False,
    demo_data_path: str = "human_demo_data/default",
):
    """
    运行仿真环境，可选择采集演示数据

    Args:
        input_type: 输入设备类型
        input_scaling: (位置缩放, 姿态缩放)
        action_space: 动作维度 (3 或 6)
        save_demo: 是否保存演示数据
        demo_data_path: 演示数据保存路径
    """
    # 创建仿真环境
    env = AssemblySimEnv(renders=True, action_dim=action_space)
    obs, info = env.reset()

    # 确保 GUI 已完全初始化（刷新滑块等）
    for _ in range(50):
        p.stepSimulation()
    time.sleep(0.1)

    # 初始化演示数据记录器
    recorder = DemoRecorder() if save_demo else None

    # 初始化输入设备
    device_cls = REGISTRY.get(input_type)
    if device_cls is None:
        available = ", ".join(sorted(set(k for k in REGISTRY.keys() if len(k) <= 4)))
        raise ValueError(
            f"未知输入类型: {input_type}，可用: {available}"
        )

    device = device_cls(*input_scaling)
    device.start()

    if save_demo:
        recorder.start_episode(obs)

    terminated = False
    truncated = False
    step_count = 0
    episode_count = 0

    try:
        while True:
            device.update()
            action = device.pose[:action_space]

            # Gymnasium 返回 5 个值
            new_obs, reward, terminated, truncated, info = env.step(action)

            if save_demo and np.count_nonzero(action) > 0:
                recorder.record_step(obs, action, reward, new_obs, terminated)

            obs = new_obs
            step_count += 1

            if terminated:
                print(f"Episode {episode_count + 1} 结束, 成功率: {info.get('num_success', 0)}", flush=True)
                # 成功后自动开始新 episode
                episode_count += 1
                obs, info = env.reset()
                if save_demo:
                    recorder.start_episode(obs)
                terminated = False
                truncated = False
            elif truncated:
                print(f"Episode {episode_count + 1} 超时, 步数: {step_count}", flush=True)
                # 超时后自动开始新 episode
                episode_count += 1
                obs, info = env.reset()
                if save_demo:
                    recorder.start_episode(obs)
                terminated = False
                truncated = False

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        device.disconnect()
        env.close()

        if save_demo and recorder:
            recorder.save(demo_data_path)
            print(f"   采集步数: {step_count}")
