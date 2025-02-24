import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def rotation_matrix_to_xyz_quat(rotation_matrix: np.ndarray):
    assert rotation_matrix.shape == (4, 4), "Invalid rotation matrix shape!"

    translation = rotation_matrix[:3, 3]
    R = rotation_matrix[:3, :3]

    qw = np.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2]) / 2
    qx = (R[2, 1] - R[1, 2]) / (4 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4 * qw)

    return translation.tolist() + [qx, qy, qz, qw]

def rotation_matrix_to_xyz_rpy(pose_current_mat: np.ndarray):
    rpy: np.ndarray = R.from_matrix(pose_current_mat[:3,:3]).as_euler('xyz', degrees=True)
    xyz: np.ndarray = pose_current_mat[:3,3]
    return xyz.tolist() + rpy.tolist()

record_file_path = "traj/record_cartesian.json"
with open(record_file_path, "r") as json_file:
    record_traj = json.load(json_file)

replay_file_path = "traj/replay_cartesian.json"
with open(replay_file_path, "r") as json_file:
    replay_traj = json.load(json_file)

left_sum_error = 0
right_sum_error = 0
length = len(record_traj)
for i in range(length):
    record_left_arm = np.array(record_traj[f"{i}"]["left_arm"])
    record_right_arm = np.array(record_traj[f"{i}"]["right_arm"])
    replay_left_arm = np.array(replay_traj[f"{i}"]["left_arm"])
    replay_right_arm = np.array(replay_traj[f"{i}"]["right_arm"])
    record_left_arm = np.array(rotation_matrix_to_xyz_rpy(record_left_arm))
    record_right_arm = np.array(rotation_matrix_to_xyz_rpy(record_right_arm))
    replay_left_arm = np.array(rotation_matrix_to_xyz_rpy(replay_left_arm))
    replay_right_arm = np.array(rotation_matrix_to_xyz_rpy(replay_right_arm))
    left_arm_error = np.abs(record_left_arm - replay_left_arm)
    right_arm_error = np.abs(record_right_arm - replay_right_arm)
    left_sum_error += left_arm_error
    right_sum_error += right_arm_error

print(left_sum_error/length)
print(right_sum_error/length)

cartesian_error_path = "traj/cartesian_error.txt"
with open(cartesian_error_path, "w") as cartesian_error_path:
    print(left_sum_error/length, file=cartesian_error_path)
    print(right_sum_error/length, file=cartesian_error_path)