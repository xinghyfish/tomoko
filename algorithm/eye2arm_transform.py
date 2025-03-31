from typing import List

from scipy.spatial.transform import Rotation as R
import numpy as np

# hand-eye matrix
H_cb = np.array([
    [1.09081046e-02, 9.99026366e-01, 4.27473136e-02, -0.0626771011],
    [-9.99927051e-01, 1.11197500e-02, -4.71642581e-03,  0.0418794266],
    [-5.18717318e-03, -4.26927479e-02, 9.99074783e-01,  0.0646131336],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
])


def euler2quaternion(euler_angles: List):
    """
    欧拉角转四元组。
    :param euler_angles: [Rx, Ry, Rz] 欧拉角序列
    :return: 四元组序列，用于进行手眼标定
    """
    rx_axis, ry_axis, rz_axis = euler_angles
    R_g = R.from_euler('xyz', euler_angles, degrees=True).as_quat()
    return R_g


def get_target_index(R_g: List, T_g: List, T_c: List):
    """
    手眼标定函数，将深度相机拍摄得到的目标坐标转为机械臂末端坐标。
    :param R_g: [x, y, z, w] 机械臂位姿四元组
    :param T_g: [x, y, z] 机械臂位置向量  单位: m
    :param T_c: [x, y, z, 1] 相机识别目标坐标 单位: m
    :return: [x, y, z] 机械臂目标坐标(不包含抓夹长度) 单位: m
    """
    T_g = np.array(T_g)
    R_g_matrix = R.from_quat(R_g).as_matrix()
    vec_hb = np.array(R_g_matrix)   # vec_hb 机械臂矩阵
    T_g = T_g[:, np.newaxis]
    vec_hb = np.hstack((vec_hb, T_g))
    vec_hb = np.vstack((vec_hb, [0, 0, 0, 1]))

    world_matrix = np.dot(vec_hb, H_cb)
    world_matrix = np.dot(world_matrix, T_c)
    target_index = world_matrix[:3]

    return target_index


def mm2m(data: List[float]) -> List[float]:
    return [x * 0.001 for x in data]


def m2mm(data: List[float]) -> List[float]:
    return [x * 1000 for x in data]
