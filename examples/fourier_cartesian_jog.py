import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from fourier_grx_client import RobotClient, ControlGroup
from loguru import logger
import sys
import math
from typing import List

TRANSLATION_MODES = ("x", "y", "z")
ROTATION_MODES = ("r", "p", "yy")
AVAILABLE_MODES = TRANSLATION_MODES + ROTATION_MODES

def askForCommand():
    strAskForCommand  = '\n可用指令\n\n'
    strAskForCommand += 'x/y/z: 微调x/y/z轴位置\n'
    strAskForCommand += 'r/p/yy: 微调r/p/y欧拉角\n'
    strAskForCommand += 'w: 正方向平移/旋转一个步长\n'
    strAskForCommand += 's: 向负方向平移/旋转一个步长\n'
    strAskForCommand += 'tstep: 设置平移的步长（单位m）\n'
    strAskForCommand += 'rstep: 设置微调转角的步长（单位度）\n'
    strAskForCommand += 'h: 显示此提示，也可用于刷新显示的机械臂状态\n'
    strAskForCommand += '直接回车可以复用上一次的指令\n'
    strAskForCommand += 'q: 退出joint_jog\n'

    print(strAskForCommand)

def main():
    # Create a RobotClient object and connect to the robot server
    client = RobotClient(namespace="gr/my_awesome_robot", server_ip="192.168.137.252")

    try:
        # Enable the robot motors
        client.enable()
        logger.info("Motors enabled")
        time.sleep(1)

        np.set_printoptions(precision=4, suppress=True)
        current_mode: str = "x"
        translation_step: float = 0.02
        rotation_step: float = 2 # 单位°
        last_command: str = ""

        askForCommand()
        try:
            while True:
                # Get the forward kinematics of the left arm using current joint positions
                chain = ["left_arm"]
                fk_output = client.forward_kinematics(chain)
                pose_current_mat = fk_output[0]

                rpy: np.ndarray = R.from_matrix(pose_current_mat[:3,:3]).as_euler('xyz', degrees=True)
                xyz: np.ndarray = pose_current_mat[:3,3]

                translation_mode: bool = current_mode in TRANSLATION_MODES
                current_step: float = translation_step if translation_mode else rotation_step
                mode_str = "当前模式：%s%s, 步长：%.2f%s" % (current_mode[0].upper(), "轴平移" if translation_mode else "角旋转",
                                                            current_step, "m" if translation_mode else "°")
                inp = input("%s, 当前xyz位移：%s, 当前rpy转角: [%.2f°, %.2f°, %.2f°], 输入指令:" % (mode_str, np.array2string(xyz), rpy[0], rpy[1], rpy[2])).lower()

                # 保存/调出历史指令
                if len(inp) == 0:
                    inp = last_command
                else:
                    last_command = inp

                # 解析指令
                if inp in AVAILABLE_MODES:
                    current_mode = inp
                    print("切换到{} (y为position，yy为欧拉角)".format(current_mode))
                elif inp == 'rstep':
                    inp = input("输入rotation新步长(°):")
                    try:
                        rotation_step = float(inp)
                    except ValueError:
                        print("无效步长, 设置失败")
                elif inp == 'tstep':
                    inp = input("输入translation新步长(m):")
                    try:
                        translation_step = float(inp)
                    except ValueError:
                        print("无效步长, 设置失败")
                elif inp == 'w' or inp == 's':
                    sign: int = 1 if inp == 'w' else -1

                    if translation_mode:
                        xyz[TRANSLATION_MODES.index(current_mode)] += current_step * sign
                    else:
                        rpy[ROTATION_MODES.index(current_mode)] += current_step * sign

                    # 计算目标姿态
                    pose_current_mat[:3,:3] = R.from_euler('xyz', rpy, degrees=True).as_matrix()
                    pose_current_mat[:3,3] = xyz

                    # 移动机械臂
                    print(pose_current_mat, type(pose_current_mat))
                    # client.inverse_kinematics(chain_names=['left_arm'], targets=[pose_current_mat], move=True)
                else:
                    askForCommand()

                time.sleep(0.1)

        finally:
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