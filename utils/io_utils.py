"""
I/O 工具函数
-----------
终端彩色打印、CSV 日志写入、URDF 路径解析、PyBullet 可视化辅助等
"""

import csv
import os
import inspect


# ============================================================
# 终端彩色打印
# ============================================================

def pr_green(text):
    """绿色文字（成功信息）"""
    print(f"\033[92m{text}\033[00m")


def pr_red(text):
    """红色文字（失败信息）"""
    print(f"\033[91m{text}\033[00m")


def pr_info(text):
    """蓝色文字（信息）"""
    print(f"\033[94m{text}\033[00m")


# ============================================================
# CSV 数据记录
# ============================================================

def write_csv(data, csv_file, overwrite=False):
    """追加写入 CSV 文件，overwrite=True 则覆盖"""
    if os.path.isfile(csv_file) and overwrite:
        os.remove(csv_file)
    with open(csv_file, 'a', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)


# ============================================================
# URDF 文件路径处理
# ============================================================

def format_urdf_filepath(name: str) -> str:
    """
    拼接 URDF 文件的完整路径
    自动补上 .urdf 后缀
    """
    dot_urdf = '.urdf'
    if dot_urdf not in name:
        name += dot_urdf
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    # 回到项目根目录（向上两级：utils -> robotic_assembly）
    project_root = os.path.dirname(currentdir)
    return os.path.join(project_root, name)


# ============================================================
# PyBullet 可视化辅助
# ============================================================

def display_frame_axis(body_uid, link_index, line_length=0.05):
    """
    在 PyBullet 中显示坐标轴：
    红色=X, 绿色=Y, 蓝色=Z
    """
    import pybullet as p
    p.addUserDebugLine([0, 0, 0], [line_length, 0, 0], [1, 0, 0],
                       parentObjectUniqueId=body_uid, parentLinkIndex=link_index)
    p.addUserDebugLine([0, 0, 0], [0, line_length, 0], [0, 1, 0],
                       parentObjectUniqueId=body_uid, parentLinkIndex=link_index)
    p.addUserDebugLine([0, 0, 0], [0, 0, line_length], [0, 0, 1],
                       parentObjectUniqueId=body_uid, parentLinkIndex=link_index)
