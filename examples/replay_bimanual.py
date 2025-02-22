import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from diffusion_policy.common.replay_buffer import ReplayBuffer
import zarr
from keystroke_counter import (
    KeystrokeCounter, KeyCode
)
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs

register_codecs()

# tcp pose 格式：[n, 6], 后3是rotvector, 现在的pose都是tcp2tag（桌面上的）

tx_tcp_flange = np.array([[1, 0, 0, 0],
                          [0, 1, 0, 0],
                          [0, 0, 1, 0.24551],
                          [0, 0, 0, 1]])
tx_flange_tcp = np.linalg.inv(tx_tcp_flange)

def get_transformation_matrix(position, rotation_vector):
    # rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    rotation_matrix = R.from_euler('xyz', rotation_vector, degrees=True).as_matrix()
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = position

    return transformation_matrix

def get_robot_pose_list(flange_n_2_flange_0_list: list, width_list: list):
    """获取机械臂位姿列表
    Args:
        input: 数据路径
        arm_id: 机械臂ID ('A' 或 'B')
    Returns:
        relative_pose_list: 相对于初始位姿的位姿列表
        width_list: 对应的夹爪开度列表
    """
    flange_0_2_tag = flange_n_2_flange_0_list[0]  # First pose
    tag_2_flange_0 = np.linalg.inv(flange_0_2_tag)

    relative_pose_list = [tag_2_flange_0 @ flange_n_2_tag for flange_n_2_tag in flange_n_2_flange_0_list]

    return relative_pose_list, width_list

def exec_arm(relative_pose_list_a, width_list_a):
    """执行双臂轨迹
    Args:
        relative_pose_list_a: 机械臂A的相对位姿列表
        width_list_a: 机械臂A的夹爪开度列表
        relative_pose_list_b: 机械臂B的相对位姿列表
        width_list_b: 机械臂B的夹爪开度列表
    """
    robot_a = RokaeInterface(arm_name='A')
    pgi_a = PGIInterface(serial_name='/dev/ttyUSB0', timeout=1)

    curr_pose_a = robot_a.get_obs_replay

    next_pose_list_a = [curr_pose_a @ relative_pose for relative_pose in relative_pose_list_a]
    step_idx = 0
    for (pose_a, width_a) in zip(next_pose_list_a, width_list_a,):
        robot_a.execute_replay(pose_a)
        pgi_a.set_pos(width_a)
        print(f"step {step_idx}")
        step_idx += 1
        time.sleep(0.1)

if __name__ == "__main__":
    input = '/mnt/disk2/data_collection/SingleArm/wash_with_brush_right_slam_pipeline/wash_with_brush_right_slam_pipeline.zarr.zip'
    with zarr.ZipStore(input, mode='r') as zip_store:
        replay_buffer = ReplayBuffer.copy_from_store(
            src_store=zip_store, store=zarr.MemoryStore())
    # 4 70 60 49 30
    replay_episode = 0
    episode_slice = replay_buffer.get_episode_slice(replay_episode)
    start_idx = episode_slice.start
    stop_idx =  episode_slice.stop
    flag = 0
    traj_list = list()
    # width_list = list()
    while True:
        pos = replay_buffer['robot0_eef_pos'][start_idx]
        rot = replay_buffer['robot0_eef_rot_axis_angle'][start_idx]
        traj = get_transformation_matrix(pos, rot)
        # width = replay_buffer['robot0_gripper_width'][start_idx]
        # width_list.append(width)
        traj_list.append(traj)
        if start_idx == stop_idx:
            break
        start_idx += 1

    print("ready to exec!!!!!!!!!!")
    relative_pose_list_a, width_list_a = get_robot_pose_list(traj_list, width_list)

    exec_arm(
        relative_pose_list_a, width_list_a,
    )