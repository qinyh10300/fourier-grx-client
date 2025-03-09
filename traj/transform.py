import csv
import numpy as np

def quaternion_to_euler(q_x, q_y, q_z, q_w):
    # 四元数转欧拉角
    t0 = +2.0 * (q_w * q_x + q_y * q_z)
    t1 = +1.0 - 2.0 * (q_x * q_x + q_y * q_y)
    roll_x = np.arctan2(t0, t1)

    t2 = +2.0 * (q_w * q_y - q_z * q_x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = np.arcsin(t2)

    t3 = +2.0 * (q_w * q_z + q_x * q_y)
    t4 = +1.0 - 2.0 * (q_y * q_y + q_z * q_z)
    yaw_z = np.arctan2(t3, t4)

    return roll_x, pitch_y, yaw_z

def read_and_convert_csv(filepath):
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        data = []
        for row in reader:
            x, y, z = float(row[1]), float(row[2]), float(row[3])
            q_x, q_y, q_z, q_w = float(row[4]), float(row[5]), float(row[6]), float(row[7])
            roll, pitch, yaw = quaternion_to_euler(q_x, q_y, q_z, q_w)
            data.append([x, y, z, roll, pitch, yaw])
    return data

def save_to_txt(data, output_filepath):
    with open(output_filepath, 'w') as file:
        for row in data:
            file.write(' '.join(map(str, row)) + '\n')

# 使用示例
input_filepath = '/home/qinyh/codebase/fourier-grx-client/traj/demo_C3464280101035_2025.03.07_10.53.23.088714/camera_trajectory.csv'
output_filepath = '/home/qinyh/codebase/fourier-grx-client/traj/demo_C3464280101035_2025.03.07_10.53.23.088714/converted_poses.txt'

converted_data = read_and_convert_csv(input_filepath)
save_to_txt(converted_data, output_filepath)