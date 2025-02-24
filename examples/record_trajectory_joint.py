from fourier_grx_client import RobotClient, ControlGroup
from loguru import logger
import time
import sys
from keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
import json
import os
import numpy as np

def main():
    # Create a RobotClient object and connect to the robot server
    client = RobotClient(namespace="gr/my_awesome_robot", server_ip="192.168.137.252")

    try:
        # Enable the robot motors
        client.enable()
        logger.info("Motors enabled")
        time.sleep(1)

        file_path = "./traj/record_joint.json"
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        with KeystrokeCounter() as key_counter:
            flag = 0
            traj = {}
            cnt = 0
            while True:
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='s'):
                        flag = 1   
                        client.set_enable(False)
                        logger.success("Start Recording!")
                    elif key_stroke == KeyCode(char='q'):
                        flag = -1   
                        client.set_enable(True)
                        logger.success("Stop Recording!")
                        break
                    
                if flag == -1:
                    break
                elif flag == 1:
                    joint_positions = client.joint_positions.copy()
                    joint_positions = joint_positions.tolist() if isinstance(joint_positions, np.ndarray) else joint_positions
                    traj[cnt] = {  
                        "joint_pos": joint_positions
                    } 
                    cnt += 1
                    print(cnt)
                    time.sleep(0.01)

        with open(file_path, "w") as json_file:
            json.dump(traj, json_file, indent=4)
        # print(traj)

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