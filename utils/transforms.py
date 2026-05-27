"""
坐标系变换工具函数
------------------
四元数格式说明：
  - xyzw = [x, y, z, w]  —— PyBullet 使用
  - wxyz = [w, x, y, z]  —— transforms3d 库使用
"""

import numpy as np
import transforms3d


# ============================================================
# 四元数 / 欧拉角 格式转换
# ============================================================

def qinverse(q):
    """四元数求逆（输入 xyzw 格式，输出 wxyz 格式）"""
    return transforms3d.quaternions.qinverse(q[3], q[0], q[1], q[2])


def xyzw_by_euler(euler, axes):
    """欧拉角 -> 四元数（xyzw 格式）"""
    q = transforms3d.euler.euler2quat(euler[0], euler[1], euler[2], axes)
    return wxyz_to_xyzw(q)


def quat_to_euler(xyzw, axes):
    """四元数（xyzw 格式）-> 欧拉角"""
    return transforms3d.euler.quat2euler(xyzw_to_wxyz(xyzw), axes)


def xyzw_to_wxyz(xyzw):
    """xyzw -> wxyz"""
    return [xyzw[3], xyzw[0], xyzw[1], xyzw[2]]


def wxyz_to_xyzw(wxyz):
    """wxyz -> xyzw"""
    return [wxyz[1], wxyz[2], wxyz[3], wxyz[0]]


# ============================================================
# 矩阵运算
# ============================================================

def mat33_by_abc(abc):
    """欧拉角（ZYX 内旋）-> 3x3 旋转矩阵"""
    a, b, c = abc
    return transforms3d.euler.euler2mat(a, b, c, axes='rzyx')


def mat33_by_mat44(mat):
    """从 4x4 齐次矩阵中提取 3x3 旋转矩阵"""
    return mat[:3, :3].copy()


def mat33_by_quat(xyzw):
    """四元数（xyzw）-> 3x3 旋转矩阵"""
    return transforms3d.quaternions.quat2mat(xyzw_to_wxyz(xyzw))


def mat33_to_quat(mat):
    """3x3 旋转矩阵 -> 四元数（xyzw）"""
    wxyz = transforms3d.quaternions.mat2quat(mat)
    return wxyz_to_xyzw(wxyz)


def mat33_to_abc(mat):
    """3x3 旋转矩阵 -> 欧拉角（ZYX 内旋）"""
    return list(transforms3d.euler.mat2euler(mat, axes='rzyx'))


# ============================================================
# 4x4 齐次变换矩阵构建与分解
# ============================================================

def mat44_by_pos_mat33(pos, mat):
    """位置 + 3x3 旋转矩阵 -> 4x4 齐次矩阵"""
    m = np.eye(4)
    m[:3, :3] = mat
    m[:3, 3] = pos
    return m


def mat44_by_pos_quat(pos, quat):
    """位置 + 四元数 -> 4x4 齐次矩阵"""
    return mat44_by_pos_mat33(pos, mat33_by_quat(quat))


def mat44_by_pos_abc(pos, abc):
    """位置 + 欧拉角 -> 4x4 齐次矩阵"""
    return mat44_by_pos_mat33(pos, mat33_by_abc(abc))


def mat44_to_pos_abc(mat):
    """4x4 矩阵 -> (位置, 欧拉角)"""
    return mat44_to_pos(mat), mat33_to_abc(mat)


def mat44_to_pos_quat(mat):
    """4x4 矩阵 -> (位置, 四元数)"""
    pos = mat[:3, 3].tolist()
    quat = mat33_to_quat(mat[:3, :3])
    return pos, quat


def mat44_to_pos(mat):
    """4x4 矩阵 -> 位置 [x, y, z]"""
    return mat[:3, 3].tolist()


# ============================================================
# 坐标变换运算
# ============================================================

def get_relative_xform(mat_from, mat_to):
    """
    计算从 mat_from 到 mat_to 的相对变换矩阵
    即：已知 A 位姿和 B 位姿，求 A->B 的相对变换
    """
    return mat_to @ np.linalg.inv(mat_from)


def transform_mat(xform, mat):
    """用变换矩阵 xform 去变换矩阵 mat"""
    return xform @ mat


def transform_mat_from_to(mat, mat_from, mat_to):
    """
    将 mat 从 mat_from 坐标系变换到 mat_to 坐标系
    """
    xform = get_relative_xform(mat_from, mat_to)
    return transform_mat(xform, mat)


def get_f1_to_f2_xform(pose_from, pose_to):
    """
    计算从 pose_from 到 pose_to 的相对位姿变换
    pose 格式: (位置列表, 四元数列表)
    """
    return get_relative_xform(
        mat44_by_pos_quat(pose_from[0], pose_from[1]),
        mat44_by_pos_quat(pose_to[0], pose_to[1]),
    )
