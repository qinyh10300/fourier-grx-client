import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
print(ROOT_DIR)
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import time
import numpy as np
from typing import List, Optional
from loguru import logger
from fourier_grx_client import RobotClient, ControlGroup
# from src.fourier_grx_client import RobotClient, ControlGroup
import math

    
if __name__ == "__main__":
    # robot = FourierInterface(which_arm='left_arm')
    # pose = robot.get_obs
    # print(pose)
    # pose[0] += 0.1
    # print(pose)
    # robot.execute(pose)
    # # robot.execute(np.array([0.5, 0.5, 0.5, 0, 0, 0]))
    # robot.stop()
    client = RobotClient(namespace="gr/my_awesome_robot", server_ip="192.168.137.252")
# try:
    client.enable()
    logger.info("Motors enabled")
    time.sleep(1)

    for i in range(10):
        input("Press Enter to continue...")
        joint_pos = client.joint_positions
        curr_pose: List[float] = list(joint_pos)
        print(curr_pose[18])
        curr_pose[18] -= math.radians(3)
        print(curr_pose[18])
        # client.move_joints([ctrl_joint], [curr_pose[ctrl_joint]], degrees=False)
        client.move_joints(ControlGroup.ALL, curr_pose, degrees=False)
        # client.move_joints(ControlGroup.ALL, curr_pose, degrees=False, blocking=False, duration=0)
        time.sleep(1)
# finally:
    client.disable()
    logger.info("Motors disabled")
    # Close the connection to the robot server
    client.close()