from fourier_grx_client import RobotClient, ControlGroup
from loguru import logger
import time
import sys
from keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
import json
import numpy as np

def main():
    # Create a RobotClient object and connect to the robot server
    client = RobotClient(namespace="gr/my_awesome_robot", server_ip="192.168.137.252")

    try:
        # Enable the robot motors
        client.enable()
        logger.info("Motors enabled")
        time.sleep(1)

        # 从 JSON 文件读取字典
        file_path = "traj/record_cartesian.json"
        with open(file_path, "r") as json_file:
            traj = json.load(json_file)

        logger.info("Moving to start pose")
        left_arm_pose = np.array(traj['0']["left_arm"])
        right_arm_pose = np.array(traj['0']["right_arm"])
        pos = client.inverse_kinematics(chain_names=['left_arm', 'right_arm'], targets=[left_arm_pose, right_arm_pose], move=True)
        time.sleep(3)

        with KeystrokeCounter() as key_counter:
            flag = 0
            index = 0
            while True:
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='w'):
                        index = min(len(traj) - 1, index + 1)
                        flag = 1
                        print(index)
                    elif key_stroke == KeyCode(char='s'):
                        index = max(0, index - 1)
                        flag = 1
                        print(index)
                    elif key_stroke == KeyCode(char='q'):
                        flag = -1
                        break

                if flag == 1:
                    left_arm_pose = np.array(traj[f'{index}']["left_arm"])
                    right_arm_pose = np.array(traj[f'{index}']["right_arm"])
                    pos = client.inverse_kinematics(chain_names=['left_arm', 'right_arm'], targets=[left_arm_pose, right_arm_pose], move=True)
                    # print(pos)
                    time.sleep(0.1)
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