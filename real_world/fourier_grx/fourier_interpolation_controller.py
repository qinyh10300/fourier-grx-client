import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import time
import enum
import numpy as np
import multiprocessing as mp
import scipy.spatial.transform as st
from multiprocessing.managers import SharedMemoryManager
from real_world.fourier_grx.fourier_interface import FourierInterface

from real_world.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from real_world.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from real_world.fourier_grx.pose_trajectory_interpolator import PoseTrajectoryInterpolator

from diffusion_policy.common.precise_sleep import precise_wait
from real_world.pose_utils import pose_to_mat, mat_to_pose

class CustomError(Exception):
    def __init__(self, message):
        self.message = message

class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2

tx_cam_flange = np.identity(4)
tx_cam_flange[:3, 3] = np.array([0, 0.11, 0.065])
tx_flange_cam = np.linalg.inv(tx_cam_flange)

class FourierInterpolationController(mp.Process):
    def __init__(self,
        shm_manager: SharedMemoryManager, 
        launch_timeout=3,
        verbose = False,
        receive_latency=0.0,
        max_pos_speed=0.25, # 5% of max speed
        max_rot_speed=0.16, # 5% of max speed
        frequency = 100,   # TODO: check if this is the right frequency
        soft_real_time=False,
        get_max_k=None,
        namespace: str = "gr/my_awesome_robot", 
        server_ip: str = "192.168.137.252",
        which_arm: str = 'left_arm',
        is_bimanual: bool = False,
        duration: float = 0.2): 
        assert which_arm in ['left_arm', 'right_arm'], "which_arm must be 'left_arm' or 'right_arm'"
        
        super().__init__(name="FourierInterpolationController")

        self.launch_timeout = launch_timeout
        self.verbose = verbose
        self.receive_latency = receive_latency
        self.frequency = frequency
        self.soft_real_time = soft_real_time

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=512
        )

        example = dict()

        example['tip_pose'] = np.zeros(6)
        example['robot_receive_timestamp'] = time.time()
        example['robot_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.which_arm = which_arm
        self.is_bimanual = is_bimanual
        self.duration = duration
        self.namespace = namespace
        self.server_ip = server_ip

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[FourierInterpolationController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)    # 阻塞直到 ready_event 事件被设置或超时
        assert self.is_alive()                        # 确保控制器进程或线程已经启动并处于活动状态
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def servoL(self, pose, duration=0.1):
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    # ========= main loop in process ============
    def run(self):
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))
        robot = FourierInterface(which_arm=self.which_arm,
                                 namespace = self.namespace, 
                                 server_ip  = self.server_ip,
                                 which_arm = self.which_arm,
                                 is_bimanual = self.is_bimanual,
                                 duration = self.duration)

        try:
            if self.verbose:
                print(f"[FourierInterpolationController] Connect to robot")

            dt = 1. / self.frequency
            curr_pose = mat_to_pose(pose_to_mat(robot.get_obs) @ tx_cam_flange)
            
            # use monotonic time to make sure the control loop never go backward
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[curr_pose]
            )

            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            while keep_running:
                t_now = time.monotonic()
                cam_pose = pose_interp(t_now)
                flange_pose = mat_to_pose(pose_to_mat(cam_pose) @ tx_flange_cam)

                print(robot.check())
                if not robot.check():
                    raise CustomError("Rokae Check Error")

                robot.execute(flange_pose, Delta=False)

                # update robot state
                state = dict()
                state['tip_pose'] = mat_to_pose(pose_to_mat(robot.get_obs) @ tx_cam_flange)
                t_recv = time.time()
                state['robot_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                # fetch command from queue
                try:
                    # process at most 1 command per cycle to maintain frequency（和gripper不同）
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        # stop immediately, ignore later commands
                        break
                    elif cmd == Command.SERVOL.value:
                        # since curr_pose always lag behind curr_target_pose
                        # if we start the next interpolation with curr_pose
                        # the command robot receive will have discontinouity 
                        # and cause jittery robot behavior.
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[FourierInterpolationController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            max_pos_speed=self.max_pos_speed,
                            max_rot_speed=self.max_rot_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break
                    

                # regulate frequency
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # first loop successful, ready to receive command
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[FourierInterpolationController] Actual frequency {1/(time.monotonic() - t_now)}")
        except Exception as e:
            print(f"Exception occurred: {e}")
            raise
        finally:
            robot.stop()
            del robot
            self.ready_event.set()

            if self.verbose:
                print(f"[FourierInterpolationController] Disconnected from robot")