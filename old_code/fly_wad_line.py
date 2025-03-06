# 沿着无人机的z轴从-10m到-1米, x y不变

import airsim
import numpy as np
import time

# 连接到AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# 无人机起飞
client.takeoffAsync().join()

# 起始和目标Z坐标
start_z = -10  # 起始Z坐标
end_z = -1      # 目标Z坐标

# X和Y坐标保持不变
x, y = 0, 0
num_steps = 1000  # 逐步移动的步数

# 逐步移动到目标高度
for i in range(num_steps + 1):
    z = start_z + (end_z - start_z) * (i / num_steps)  # 计算当前Z坐标
    pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(pitch=0, roll=0, yaw=0))
    client.simSetVehiclePose(pose, True)
    time.sleep(0.1)  # 等待以便平滑移动

# 完成后，确保无人机稳定
client.hoverAsync().join()

print("运行完成")