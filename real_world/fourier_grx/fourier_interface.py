import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import time
import numpy as np
from typing import List, Optional
from loguru import logger
from scipy.spatial.transform import Rotation as R
from real_world.fourier_grx.pose_utils import pose_to_mat, mat_to_pose
from src.fourier_grx_client import RobotClient, ControlGroup

class FourierInterface:
    def __init__(self, 
                 namespace: str = "gr/my_awesome_robot", 
                 server_ip: str = "192.168.137.252",
                 which_arm: str = 'left_arm',
                 is_bimanual: bool = False,
                 duration: float = 0.2):
        assert which_arm in ['left_arm', 'right_arm'], "which_arm must be 'left_arm' or 'right_arm'"
        self.which_arm = which_arm
        self.is_bimanual = is_bimanual
        self.duration = duration

        self.client = RobotClient(namespace=namespace, server_ip=server_ip)
        self.client.enable()
        logger.info("Motors enabled")
        time.sleep(1)

    def __del__(self):
        self.client.disable()
        logger.info("Motors disabled")
        # Close the connection to the robot server
        self.client.close()
    
    def execute(self, xyz_rpy: np.ndarray, degrees: bool = False, blocking: bool = False):
        pose_mat = pose_to_mat(xyz_rpy)
        ik_joint_pos = self.client.inverse_kinematics(chain_names=[self.which_arm], targets=[pose_mat], move=False)
        
        joint_pos = self.client.joint_positions
        curr_pose: List[float] = list(joint_pos)
        start_index, end_index = ControlGroup.from_string(self.which_arm).slice
        curr_pose[start_index:end_index] = ik_joint_pos

        self.client.move_joints(ControlGroup.ALL, curr_pose, degrees=degrees, blocking=blocking, duration=self.duration)

    def stop(self, reason: Optional[str] = None):
        self.client.disable()
        logger.info("Motors disabled")
        # Close the connection to the robot server
        self.client.close()

        if reason == None:
            logger.info("Terminating node")
        else:
            logger.info(reason)

    def check(self):
        pass

    @property
    def get_obs(self) -> np.ndarray:
        chain = [self.which_arm]
        fk_output = self.client.forward_kinematics(chain)
        current_mat = fk_output[0].copy()
        return mat_to_pose(current_mat)