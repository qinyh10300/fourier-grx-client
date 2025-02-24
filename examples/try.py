import os
import json

replay_file_path = "./traj/replay_joint.json"
replay_dir_path = os.path.dirname(replay_file_path)
print(replay_dir_path, replay_file_path)
if not os.path.exists(replay_dir_path):
    os.makedirs(replay_dir_path)

replay = {}

with open(replay_file_path, "w") as json_file:
        json.dump(replay, json_file, indent=4)