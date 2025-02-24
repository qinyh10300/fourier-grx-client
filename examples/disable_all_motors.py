import time
from fourier_grx_client import RobotClient, ControlGroup
from loguru import logger

if __name__ == "__main__":
    # Enable the robot motors
    client = RobotClient(namespace="gr/my_awesome_robot", server_ip="192.168.137.252")
    try:
        client.enable()
        logger.info("Motors enabled")
        time.sleep(10)
    finally:
        # Disable the robot motors
        client.disable()
        logger.info("Motors disabled")

        # Close the connection to the robot server
        client.close()