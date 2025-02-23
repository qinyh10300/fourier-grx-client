import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from diffusion_policy.common.replay_buffer import ReplayBuffer
import zarr
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from fourier_grx_client import RobotClient, ControlGroup
from loguru import logger

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

# def get_robot_pose_list(flange_n_2_flange_0_list: list, width_list: list):
#     flange_0_2_tag = flange_n_2_flange_0_list[0]  # First pose
#     tag_2_flange_0 = np.linalg.inv(flange_0_2_tag)

#     relative_pose_list = [tag_2_flange_0 @ flange_n_2_tag for flange_n_2_tag in flange_n_2_flange_0_list]

#     return relative_pose_list, width_list

# def exec_arm(relative_pose_list_a, width_list_a):
#     robot_a = RokaeInterface(arm_name='A')
#     pgi_a = PGIInterface(serial_name='/dev/ttyUSB0', timeout=1)

#     curr_pose_a = robot_a.get_obs_replay

#     next_pose_list_a = [curr_pose_a @ relative_pose for relative_pose in relative_pose_list_a]
#     step_idx = 0
#     for (pose_a, width_a) in zip(next_pose_list_a, width_list_a,):
#         robot_a.execute_replay(pose_a)
#         pgi_a.set_pos(width_a)
#         print(f"step {step_idx}")
#         step_idx += 1
#         time.sleep(0.1)

if __name__ == "__main__":
    # Create a RobotClient object and connect to the robot server
    client = RobotClient(namespace="gr/my_awesome_robot", server_ip="192.168.137.252")

    try:
        # Enable the robot motors
        client.enable()
        logger.info("Motors enabled")
        time.sleep(1)

        input = '/home/rvsa/codehub/replay_data/pick_orange_dataset.zarr.zip'
        input = '/home/rvsa/codehub/replay_data/hang_scissors_slam_pipeline.zarr.zip'
        with zarr.ZipStore(input, mode='r') as zip_store:
            replay_buffer = ReplayBuffer.copy_from_store(
                src_store=zip_store, store=zarr.MemoryStore())
        # 4 70 60 49 30
        replay_episode = 0
        episode_slice = replay_buffer.get_episode_slice(replay_episode)
        start_idx:int = episode_slice.start
        stop_idx:int =  episode_slice.stop
        start_pose = get_transformation_matrix(replay_buffer['robot0_eef_pos'][start_idx], replay_buffer['robot0_eef_rot_axis_angle'][start_idx])
        start_pose_inv = np.linalg.inv(start_pose)

        fk_output = client.forward_kinematics(["left_arm"])
        first_pose_current_mat = fk_output[0].copy()

        logger.success(f"Begin replay! replay_episode:{replay_episode} start_idx:{start_idx} stop_idx:{stop_idx}")

        with KeystrokeCounter() as key_counter:
            flag = 0
            index = start_idx
            while True:
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='w'):
                        index = min(stop_idx - 1, index + 1)
                        flag = 1
                    elif key_stroke == KeyCode(char='s'):
                        index = max(start_idx, index - 1)
                        flag = 1

                if flag == 1:
                    next_pos = get_transformation_matrix(replay_buffer['robot0_eef_pos'][index], replay_buffer['robot0_eef_rot_axis_angle'][index])
                    relative_pos = start_pose_inv @ next_pos
                    # fk_output = client.forward_kinematics(["left_arm"])
                    # pose_current_mat = fk_output[0].copy()
                    target_pos = first_pose_current_mat @ relative_pos
                    # print(index, next_pos, relative_pos, target_pos)
                    print(index)
                    # client.movel(sides=["left"], target_poses=[target_pos])
                    pos = client.inverse_kinematics(chain_names=['left_arm'], targets=[target_pos], move=True)
                    print(pos)
                    time.sleep(0.1)
                    flag = 0

    except Exception as ex:
        print("出现以下异常%s"%ex)

    finally:
        # Disable the robot motors
        client.disable()
        logger.info("Motors disabled")

        # Close the connection to the robot server
        client.close()