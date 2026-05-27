"""
run.py - 测试仿真环境 / 采集人工演示数据

使用方法：
  1. 只看环境效果（默认）: python run.py
  2. 用手柄采集演示数据:  python run.py --input-type xbc --save-demo-data=True --demo-data-path=human_demo_data/my_demo
"""

import argparse
import collections
import pickle
import numpy as np

import envs_launcher
import devices
import utilities


def getargs():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input-type',
        type=str,
        default='pbg',
        help='输入设备类型，可选值：' + ', '.join(devices.REGISTRY.keys()))
    parser.add_argument(
        '--input-scaling',
        type=tuple,
        default=(10, 25),
        help='输入设备的缩放系数，格式为 (位置缩放, 姿态缩放)')
    parser.add_argument(
        '--action-space',
        type=int,
        default=6,
        help='动作空间自由度：3（仅平移）或 6（平移+旋转）')
    parser.add_argument(
        '--save-demo-data',
        type=bool,
        default=False,
        help='是否保存演示数据（人工操作数据，用于后续训练）')
    parser.add_argument(
        '--demo-data-path',
        type=str,
        default='human_demo_data/default',
        help='演示数据保存路径')
    args = parser.parse_args()
    assert args.input_type in devices.REGISTRY, \
        '输入类型必须为以下之一：' + ', '.join(devices.REGISTRY.keys())
    assert args.action_space == 3 or args.action_space == 6, \
        '动作空间必须为 3 或 6'
    return args


def main():
    args = getargs()

    # 创建环境（仿真 or 真实）
    environment = envs_launcher.env_creator(args)
    obs = environment.reset()
    # 用双端队列存储演示数据（obs, action, reward, next_obs, done）
    memory = collections.deque()

    # 初始化输入设备（手柄/鼠标/滑块等）
    device_cls = devices.REGISTRY[args.input_type]
    device = device_cls(*args.input_scaling)
    device.start()  # 启动设备监听

    done = False
    while not done:
        device.update()  # 读取设备的最新输入
        action = device.pose[:args.action_space]  # 取前 N 维作为动作

        # 执行动作，环境返回新状态
        new_obs, reward, done, info = environment.step(action)

        # 如果需要保存演示数据，且动作不是全零（排除静止帧）
        if args.save_demo_data and np.count_nonzero(action) > 0:
            memory.append((obs, action, reward, new_obs, done))
            obs = new_obs
    else:
        device.disconnect()  # 断开输入设备

    if args.save_demo_data:
        # 把所有 transitions 保存到文件
        file_name = args.demo_data_path
        out_file = open(file_name, 'wb')
        pickle.dump(memory, out_file)
        out_file.close()
        utilities.prGreen('演示数据已保存')
        utilities.prGreen('步数: {}'.format(len(memory)))


if __name__ == '__main__':
    main()
