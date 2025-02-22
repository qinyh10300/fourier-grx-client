from fourier_grx_client import RobotClient, ControlGroup
from loguru import logger
import time
import sys
import math
import numpy as np
from typing import List

def askForCommand():
    strAskForCommand  = '\n可用指令\n\n'
    strAskForCommand += '(0-31): 切换至指定的joint(15-17: head, 18-24: left_arm, 25-31: right_arm)\n'
    strAskForCommand += 'j: 调节运动步长\n'
    strAskForCommand += 'w: 当前joint向正方向旋转一个步长\n'
    strAskForCommand += 's: 当前joint向负方向旋转一个步长\n'
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
        jog_step: float = 3 # in degree
        ctrl_joint: int = 1
        last_command: str = "1"

        askForCommand()
        try:
            while True:
                joint_pos = client.joint_positions
                inp = input("当前joint:%d, 步长:%.1f°, 当前joint pos:%s, 输入指令:" % (ctrl_joint, jog_step, str(np.asanyarray(joint_pos)))).lower()
                
                if len(inp) == 0:
                    inp = last_command
                else:
                    last_command = inp

                try:
                    joint_num = int(inp)
                    if joint_num < 0 or joint_num > 31:
                        raise ValueError()
                    ctrl_joint = joint_num
                    print("切换到joint %d" % ctrl_joint)
                    continue
                except ValueError:
                    pass

                if inp == 'j':
                    inp = input("输入新步长(°):")
                    try:
                        jog_step = float(inp)
                    except ValueError:
                        print("无效步长, 设置失败")
                elif inp == 'w' or inp == 's':
                    sign: int = 1 if inp == 'w' else -1
                    curr_pose: List[float] = list(joint_pos)

                    raw_pose = curr_pose[ctrl_joint]
                    curr_pose[ctrl_joint] += math.radians(jog_step) * sign
                    # client.move_joints([ctrl_joint], [curr_pose[ctrl_joint]], degrees=False)
                    print("Joint %d pos %.4f -> %.4f," % (ctrl_joint, raw_pose, curr_pose[ctrl_joint]), "Sending:", np.asanyarray(curr_pose))
                elif inp == 'q':
                    break
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