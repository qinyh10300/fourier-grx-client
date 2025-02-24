import json
import numpy as np

record_file_path = "traj/record_joint.json"
with open(record_file_path, "r") as json_file:
    record_traj = json.load(json_file)

replay_file_path = "traj/replay_joint.json"
with open(replay_file_path, "r") as json_file:
    replay_traj = json.load(json_file)

sum_error = 0
length = len(record_traj)
for i in range(length):
    record_joint_pos = np.array(record_traj[f"{i}"]["joint_pos"])
    replay_joint_pos = np.array(replay_traj[f"{i}"]["joint_pos"])
    error = np.abs(record_joint_pos - replay_joint_pos)
    sum_error += error

print(sum_error/length)