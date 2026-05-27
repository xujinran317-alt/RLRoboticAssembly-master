"""
utilities.py - 工具函数集

包含功能：
  - 彩色终端打印
  - 坐标系变换矩阵运算（4x4矩阵、四元数、欧拉角）
  - PyBullet 可视化辅助
  - CSV 日志写入
  - URDF 文件路径解析
"""

import csv
import numpy
import os
import inspect
import transforms3d
import pybullet as p


# ---- 终端彩色打印 ----

def prGreen(skk):
    """打印绿色文字（用于成功信息）"""
    print("\033[92m {}\033[00m" .format(skk))


def prRed(skk):
    """打印红色文字（用于失败信息）"""
    print("\033[91m {}\033[00m" .format(skk))


# ---- PyBullet 可视化 ----

def display_frame_axis(body_uid, link_index, line_length=0.05):
    """
    在 PyBullet 中显示坐标轴：
    红色 = X轴, 绿色 = Y轴, 蓝色 = Z轴
    body_uid: 物体ID
    link_index: 连杆索引
    line_length: 轴线长度（米）
    """
    p.addUserDebugLine([0, 0, 0], [line_length, 0, 0], [1, 0, 0],
                       parentObjectUniqueId=body_uid, parentLinkIndex=link_index)
    p.addUserDebugLine([0, 0, 0], [0, line_length, 0], [0, 1, 0],
                       parentObjectUniqueId=body_uid, parentLinkIndex=link_index)
    p.addUserDebugLine([0, 0, 0], [0, 0, line_length], [0, 0, 1],
                       parentObjectUniqueId=body_uid, parentLinkIndex=link_index)


# ---- CSV 数据记录 ----

def write_csv(data, csv_file, overwrite):
    """将数据追加写入 CSV 文件，overwrite=True 则覆盖"""
    if os.path.isfile(csv_file) & overwrite:
        os.remove(csv_file)
    with open(csv_file, 'a') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(data)


# ---- URDF 文件路径处理 ----

def format_urdf_filepath(name):
    """
    拼接 URDF 文件的完整路径
    会自动补上 .urdf 后缀
    """
    dot_urdf = '.urdf'
    if dot_urdf not in name:
        name += dot_urdf
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    os.sys.path.insert(0, currentdir)
    return '{}/{}'.format(currentdir, name)


# ---- 四元数/欧拉角格式转换 ----
# 注意：本项目中四元数有两种格式：
#   xyzw = [x, y, z, w]  —— PyBullet 和部分函数使用
#   wxyz = [w, x, y, z]  —— transforms3d 库使用

def qinverse(q):
    """四元数求逆（输入 xyzw 格式，输出 wxyz 格式）"""
    return transforms3d.quaternions.qinverse(q[3], q[0], q[1], q[2])


def xyzw_by_euler(euler, axes):
    """
    欧拉角 -> 四元数（xyzw 格式）
    euler: [roll, pitch, yaw]
    axes: 旋转顺序，如 'sxyz'
    """
    q = transforms3d.euler.euler2quat(euler[0], euler[1], euler[2], axes)
    return wxyz_to_xyzw(q)


def quat_to_euler(xyzw, axes):
    """四元数（xyzw 格式）-> 欧拉角"""
    return transforms3d.euler.quat2euler(xyzw_to_wxyz(xyzw), axes)


def xyzw_to_wxyz(xyzw):
    """xyzw 格式的四元数 -> wxyz 格式"""
    wxyz = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
    return wxyz


def wxyz_to_xyzw(wxyz):
    """wxyz 格式的四元数 -> xyzw 格式"""
    xyzw = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]
    return xyzw


# ---- 矩阵运算（真实机器人用）----

def mat33_by_abc(abc):
    """
    欧拉角（ZYX内旋）-> 3x3 旋转矩阵
    abc: [绕Z轴, 绕Y', 绕X"]（即 yaw, pitch, roll）
    """
    a, b, c = abc
    return transforms3d.euler.euler2mat(a, b, c, axes='rzyx')


def mat33_by_mat44(mat):
    """从 4x4 齐次矩阵中提取左上角 3x3 旋转矩阵"""
    m = numpy.eye(3, 3)
    for i in range(3):
        for j in range(3):
            m[i][j] = mat[i][j]
    return m


def mat33_by_quat(xyzw):
    """四元数（xyzw）-> 3x3 旋转矩阵"""
    wxyz = [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]
    return transforms3d.quaternions.quat2mat(wxyz)


def mat33_to_quat(mat):
    """3x3 旋转矩阵 -> 四元数（xyzw）"""
    wxyz = transforms3d.quaternions.mat2quat(mat)
    return [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]


def mat33_to_abc(mat):
    """3x3 旋转矩阵 -> 欧拉角（ZYX内旋）"""
    return list(transforms3d.euler.mat2euler(mat, axes='rzyx'))


# ---- 4x4 齐次变换矩阵的构建与分解 ----

def mat44_by_pos_mat33(pos, mat):
    """
    用位置 + 3x3旋转矩阵 合成 4x4 齐次矩阵
    pos: [x, y, z]
    mat: 3x3 矩阵
    返回 4x4 矩阵
    """
    m = numpy.eye(4, 4)
    for i in range(3):
        for j in range(3):
            m[i][j] = mat[i][j]
        m[i][3] = pos[i]
    return m


def mat44_by_pos_quat(pos, quat):
    """用位置 + 四元数合成 4x4 齐次矩阵"""
    quat_mat = mat33_by_quat(quat)
    return mat44_by_pos_mat33(pos, quat_mat)


def mat44_by_pos_abc(pos, abc):
    """用位置 + 欧拉角合成 4x4 齐次矩阵"""
    abc_mat = mat33_by_abc(abc)
    return mat44_by_pos_mat33(pos, abc_mat)


def mat44_to_pos_abc(mat):
    """4x4 矩阵 -> 位置 + 欧拉角"""
    return mat44_to_pos(mat), mat33_to_abc(mat)


def mat44_to_pos_quat(mat):
    """4x4 矩阵 -> 位置 + 四元数"""
    pos = []
    quat_mat = numpy.eye(3, 3)
    for i in range(3):
        for j in range(3):
            quat_mat[i][j] = mat[i][j]
        pos.append(mat[i][3])
    quat = mat33_to_quat(quat_mat)
    return pos, quat


def mat44_to_pos(mat):
    """从 4x4 矩阵中提取位置 [x, y, z]"""
    return [mat[i][3] for i in range(3)]


# ---- 坐标变换运算 ----

def get_relative_xform(mat_from, mat_to):
    """
    计算从 mat_from 到 mat_to 的相对变换矩阵
    即：如果知道 A 的位置和 B 的位置，求 A->B 的相对变换
    返回 4x4 矩阵
    """
    return numpy.matmul(mat_to, numpy.linalg.inv(mat_from))


def transform_mat(xform, mat):
    """用变换矩阵 xform 去变换矩阵 mat（即 xform * mat）"""
    return numpy.matmul(xform, mat)


def transform_mat_from_to(mat, mat_from, mat_to):
    """
    用 mat_from 到 mat_to 的相对变换去变换矩阵 mat
    相当于：先变换到 mat_from 的坐标系，再变换到 mat_to 的坐标系
    """
    xform = get_relative_xform(mat_from, mat_to)
    return transform_mat(xform, mat)


def get_f1_to_f2_xform(pose_from, pose_to):
    """
    计算从 pose_from 到 pose_to 的相对位姿变换
    pose 格式: (位置list, 四元数list)
    """
    from_f1_to_f2 = get_relative_xform(
        mat44_by_pos_quat(pose_from[0], pose_from[1]),
        mat44_by_pos_quat(pose_to[0], pose_to[1])
    )
    return from_f1_to_f2
