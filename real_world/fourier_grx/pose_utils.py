import numpy as np
import scipy.spatial.transform as st

def pos_rot_to_mat(pos, rot):
    '''将位置和旋转转换为 4x4 变换矩阵'''
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4,4), dtype=pos.dtype)
    mat[...,:3,3] = pos
    mat[...,:3,:3] = rot.as_matrix()
    mat[...,3,3] = 1
    return mat

def mat_to_pos_rot(mat):
    '''将 4x4 变换矩阵转换为位置和旋转'''
    pos = (mat[...,:3,3].T / mat[...,3,3].T).T
    rot = st.Rotation.from_matrix(mat[...,:3,:3])
    return pos, rot

def pos_rot_to_pose(pos, rot):
    '''将位置和旋转转换为 6D 位姿向量（位置 + 旋转向量）'''
    shape = pos.shape[:-1]
    pose = np.zeros(shape+(6,), dtype=pos.dtype)
    pose[...,:3] = pos
    pose[...,3:] = rot.as_rotvec()
    return pose

def pose_to_pos_rot(pose):
    '''将6D位姿向量转换为位置和旋转'''
    pos = pose[...,:3]
    rot = st.Rotation.from_rotvec(pose[...,3:])
    return pos, rot

def pose_to_mat(pose):
    '''将 6D 位姿向量转换为 4x4 变换矩阵'''
    return pos_rot_to_mat(*pose_to_pos_rot(pose))

def mat_to_pose(mat):
    '''将 4x4 变换矩阵转换为 6D 位姿向量'''
    return pos_rot_to_pose(*mat_to_pos_rot(mat))

def transform_pose(tx, pose):
    """
    将一个变换应用到位姿上
    tx: tx_new_old
    pose: tx_old_obj (6D)
    result: tx_new_obj (6D)
    """
    pose_mat = pose_to_mat(pose)
    tf_pose_mat = tx @ pose_mat
    tf_pose = mat_to_pose(tf_pose_mat)
    return tf_pose

def transform_point(tx, point):
    '''将一个变换应用到点上'''
    return point @ tx[:3,:3].T + tx[:3,3]

def project_point(k, point):
    '''
    将点投影到图像平面
    k 为相机内参矩阵
    '''
    x = point @ k.T
    uv = x[...,:2] / x[...,[2]]
    return uv

def apply_delta_pose(pose, delta_pose):
    '''将一个增量位姿应用到当前位姿上'''
    new_pose = np.zeros_like(pose)

    # simple add for position
    new_pose[:3] = pose[:3] + delta_pose[:3]

    # matrix multiplication for rotation
    rot = st.Rotation.from_rotvec(pose[3:])
    drot = st.Rotation.from_rotvec(delta_pose[3:])
    new_pose[3:] = (drot * rot).as_rotvec()

    return new_pose

def normalize(vec, tol=1e-7):
    '''归一化向量'''
    return vec / np.maximum(np.linalg.norm(vec), tol)

def rot_from_directions(from_vec, to_vec):
    '''
    从两个方向向量计算旋转
    from_vec: 原始方向向量
    to_vec: 目标方向向量
    '''
    from_vec = normalize(from_vec)
    to_vec = normalize(to_vec)
    axis = np.cross(from_vec, to_vec)
    axis = normalize(axis)
    angle = np.arccos(np.dot(from_vec, to_vec))
    rotvec = axis * angle
    rot = st.Rotation.from_rotvec(rotvec)
    return rot

def normalize(vec, eps=1e-12):
    '''归一化向量'''
    norm = np.linalg.norm(vec, axis=-1)
    norm = np.maximum(norm, eps)
    out = (vec.T / norm).T
    return out

def rot6d_to_mat(d6):
    '''将 6D 旋转向量转换为旋转矩阵'''
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-2)
    return out

def mat_to_rot6d(mat):
    '''将旋转矩阵转换为 6D 旋转向量'''
    batch_dim = mat.shape[:-2]
    out = mat[..., :2, :].copy().reshape(batch_dim + (6,))
    return out

def mat_to_pose10d(mat):
    '''将 4x4 变换矩阵转换为 10D 位姿向量'''
    pos = mat[...,:3,3]
    rotmat = mat[...,:3,:3]
    d6 = mat_to_rot6d(rotmat)
    d10 = np.concatenate([pos, d6], axis=-1)
    return d10

def pose10d_to_mat(d10):
    '''将 10D 位姿向量转换为 4x4 变换矩阵'''
    pos = d10[...,:3]
    d6 = d10[...,3:]
    rotmat = rot6d_to_mat(d6)
    out = np.zeros(d10.shape[:-1]+(4,4), dtype=d10.dtype)
    out[...,:3,:3] = rotmat
    out[...,:3,3] = pos
    out[...,3,3] = 1
    return out

if __name__ == "__main__":
    a = np.array([[1,2,3],[3,2,1]])
    # a = np.array([1,2,3])
    print(normalize(a))
    # print(normalize2(a))