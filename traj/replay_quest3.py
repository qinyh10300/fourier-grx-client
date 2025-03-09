from fourier_grx_client import RobotClient, ControlGroup
from loguru import logger
import time
import sys
from keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
import json
import numpy as np
import os
from scipy.spatial.transform import Rotation as R

def read_poses(filepath):
    poses = []
    with open(filepath, 'r') as file:
        for line in file:
            poses.append(list(map(float, line.strip().split())))
    return poses

def pose_to_matrix(pose):
    x, y, z, roll, pitch, yaw = pose
    translation = np.array([x, y, z])
    rotation = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix

def matrix_to_pose(matrix):
    translation = matrix[:3, 3]
    rotation = R.from_matrix(matrix[:3, :3]).as_euler('xyz')
    return np.concatenate((translation, rotation))

def compute_relative_pose(pose, reference_pose):
    pose_matrix = pose_to_matrix(pose)
    reference_matrix = pose_to_matrix(reference_pose)
    relative_matrix = np.linalg.inv(reference_matrix) @ pose_matrix
    return matrix_to_pose(relative_matrix)

def main():
    # Create a RobotClient object and connect to the robot server
    client = RobotClient(namespace="gr/my_awesome_robot", server_ip="192.168.137.252")

    try:
        # Enable the robot motors
        client.enable()
        logger.info("Motors enabled")
        time.sleep(1)

        chain = ["left_arm"]
        fk_output = client.forward_kinematics(chain)
        start_pose = fk_output[0].copy()

        # 从 converted_poses.txt 文件读取位姿数据
        file_path = "traj/converted_poses.txt"
        poses = read_poses(file_path)
        reference_pose = poses[0]

        with KeystrokeCounter() as key_counter:
            flag = 0
            index = 0

            while True:
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='w'):
                        index = min(len(poses) - 1, index + 1)
                        flag = 1
                        print(index)
                        break
                    elif key_stroke == KeyCode(char='s'):
                        index = max(0, index - 1)
                        flag = 1
                        print(index)
                        break
                    elif key_stroke == KeyCode(char='q'):
                        flag = -1
                        break

                if flag == 1:
                    relative_pose = compute_relative_pose(poses[index], reference_pose)
                    target_pose_matrix = pose_to_matrix(start_pose) @ pose_to_matrix(relative_pose)
                    target_pose = matrix_to_pose(target_pose_matrix)
                    pos = client.inverse_kinematics(chain_names=['left_arm'], targets=[target_pose.tolist()], move=True)
                    time.sleep(0.02)

                    flag = 0
                elif flag == -1:
                    break

        # Disable the robot motors
        client.disable()
        logger.info("Motors disabled")

        # Close the connection to the robot server
        client.close()
        return True
    except Exception as e:
        logger.error(f"Error occured: {e}")
        return False

if __name__ == '__main__':
    if not main():
        sys.exit(1)