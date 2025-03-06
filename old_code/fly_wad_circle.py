import airsim
import numpy as np
import math
import time

# 连接到AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# 无人机起飞
client.takeoffAsync().join()

# 椭圆中心点坐标和长短轴
cx, cy, cz = 0, 0, 0  # 中心点坐标，此处cy实际不变
a, b = 5, 5  # 椭圆在X和Z方向的长短轴长度

# 椭圆绕飞的点数
points = 4000

for i in range(points):
    angle = 2 * math.pi * (i / points)  # 计算角度
    x = cx - a * math.cos(angle)  # 计算椭圆上的x坐标
    y = cy
    z = cz - b * math.sin(angle)  # 计算椭圆上的z坐标
    pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(pitch=np.radians(90)-angle, roll=0, yaw=0))
    client.simSetVehiclePose(pose, True)
    time.sleep(0.02)

# 完成后，确保无人机稳定
client.hoverAsync().join()

print("运行完成")