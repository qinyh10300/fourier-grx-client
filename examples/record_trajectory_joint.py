from fourier_grx_client import RobotClient, ControlGroup
from loguru import logger
import time
import sys
from keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
import json

def main():
    # Create a RobotClient object and connect to the robot server
    client = RobotClient(namespace="gr/my_awesome_robot", server_ip="192.168.137.252")

    try:
        # Enable the robot motors
        client.enable()
        logger.info("Motors enabled")
        time.sleep(1)

        with KeystrokeCounter() as key_counter:
            flag = 0
            traj = {}
            cnt = 0
            while True:
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        flag = 1
                        break
                if flag:
                    break
                
                joint_positions = client.joint_positions.copy()
                traj[cnt] = {  
                    "joint_pos": joint_positions
                } 
                cnt += 1

        # 保存字典为 JSON 文件
        with open("./traj/record_joint.json", "w") as json_file:
            json.dump(traj, json_file, indent=4)

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