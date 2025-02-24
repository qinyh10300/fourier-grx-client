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

def main():
    # Create a RobotClient object and connect to the robot server
    client = RobotClient(namespace="gr/my_awesome_robot", server_ip="192.168.137.252")

    try:
        # Enable the robot motors
        client.enable()
        logger.info("Motors enabled")
        time.sleep(1)

        replay_file_path = "./traj/replay_joint.json"
        replay_dir_path = os.path.dirname(replay_file_path)
        if not os.path.exists(replay_dir_path):
            os.makedirs(replay_dir_path)

        # 从 JSON 文件读取字典
        file_path = "traj/record_joint.json"
        with open(file_path, "r") as json_file:
            traj = json.load(json_file)

        logger.info("Moving to start pose", np.array(traj['0']["joint_pos"]))
        client.move_joints(ControlGroup.ALL, np.array(traj['0']["joint_pos"]), duration=3, degrees=False, blocking=False)
        time.sleep(3)

        with KeystrokeCounter() as key_counter:
            flag = 0
            index = 0
            replay = {}
            joint_positions = client.joint_positions.copy()
            joint_positions = joint_positions.tolist() if isinstance(joint_positions, np.ndarray) else joint_positions
            replay[0] = {  
                "joint_pos": joint_positions
            } 
            while True:
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='w'):
                        index = min(len(traj) - 1, index + 1)
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
                    client.move_joints(ControlGroup.ALL, np.array(traj[f"{index}"]["joint_pos"]), duration=0.0, degrees=False, blocking=False)
                    time.sleep(0.15)
                    joint_positions = client.joint_positions.copy()
                    joint_positions = joint_positions.tolist() if isinstance(joint_positions, np.ndarray) else joint_positions
                    replay[index] = {  
                        "joint_pos": joint_positions
                    } 
                    flag = 0
                elif flag == -1:
                    break

        with open(replay_dir_path, "w") as json_file:
            json.dump(replay, json_file, indent=4)

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