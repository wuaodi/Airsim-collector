# 飞行控制、位姿真值、传感器数据采集

import airsim
import numpy as np
import time
import os
import datetime
import cv2
from pathlib import Path

class AirSimDataCollector:
    def __init__(self):
        # 连接到AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # 创建基于时间戳的数据存储目录
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.base_path = Path(f"D:/dataset/{timestamp}")
        
        # 创建存储不同类型数据的目录
        self.folders = {
            "gt": self.base_path / "groundtruth/data",
            "lidar": self.base_path / "lidar0/data",
            "cam0_scene": self.base_path / "cam0_Scene/data",
            "cam0_seg": self.base_path / "cam0_Seg/data"
        }
        
        # 创建目录
        for folder in self.folders.values():
            folder.mkdir(parents=True, exist_ok=True)
        
        # 打开CSV文件用于记录元数据
        self.csv_files = {}
        for name, folder in self.folders.items():
            parent_folder = folder.parent
            csv_path = parent_folder / "data.csv"
            self.csv_files[name] = open(csv_path, "w")
            
            # 根据数据类型写入表头
            if name == "gt":
                self.csv_files[name].write("#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []\n")
            elif name == "lidar":
                self.csv_files[name].write("#timestamp,x,y,z,qw,qx,qy,qz\n")
            elif name == "cam0_scene" or name == "cam0_seg":
                self.csv_files[name].write("#timestamp[ns],filename\n")
        
        # 定义相机请求
        self.image_requests = [
            airsim.ImageRequest("cam0", airsim.ImageType.Scene, False, False),
            airsim.ImageRequest("cam0", airsim.ImageType.Segmentation, False, False)
        ]
    
    def collect_data(self, timestamp):
        """在当前位置收集所有传感器数据"""
        # 获取位姿真值
        state = self.client.getMultirotorState()
        self.csv_files["gt"].write(f"{timestamp},"
                                 f"{state.kinematics_estimated.position.x_val},"
                                 f"{state.kinematics_estimated.position.y_val},"
                                 f"{state.kinematics_estimated.position.z_val},"
                                 f"{state.kinematics_estimated.orientation.w_val},"
                                 f"{state.kinematics_estimated.orientation.x_val},"
                                 f"{state.kinematics_estimated.orientation.y_val},"
                                 f"{state.kinematics_estimated.orientation.z_val}\n")
        
        # 获取激光雷达数据
        lidar_data = self.client.getLidarData()
        points = lidar_data.point_cloud
        
        if points:
            # 保存激光雷达点云到文件
            lidar_filename = f"{timestamp}.asc"
            lidar_path = self.folders["lidar"] / lidar_filename

            with open(lidar_path, 'w') as f:
                # 确保我们只处理完整的点云数据（每3个值为一组）
                for i in range(0, len(points) - 2, 3):
                    x, y, z = points[i], points[i+1], points[i+2]
                    f.write(f"{x},{y},{z}\n")
            
            # 记录激光雷达位姿
            # 是相对于世界原点的，已包含无人机运动和安装偏移
            pose = lidar_data.pose
            self.csv_files["lidar"].write(f"{timestamp},"
                                       f"{pose.position.x_val},"
                                       f"{pose.position.y_val},"
                                       f"{pose.position.z_val},"
                                       f"{pose.orientation.w_val},"
                                       f"{pose.orientation.x_val},"
                                       f"{pose.orientation.y_val},"
                                       f"{pose.orientation.z_val}\n")
        
        # 获取相机图像
        responses = self.client.simGetImages(self.image_requests)
        if responses:
            for i, response in enumerate(responses):
                img_type = response.image_type
                img_type = response.image_type
                cam_name = response.camera_name
                cam_id = cam_name[-1]
                
                img_type = response.image_type                
                cam_name = response.camera_name
                cam_id = cam_name[-1]
                
                filename = f"{timestamp}.png"
                
                if img_type == airsim.ImageType.Scene:
                    folder_key = "cam0_scene"
                    img_path = self.folders[folder_key] / filename
                    # 保存RGB图像
                    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                    img_rgb = img1d.reshape(response.height, response.width, 3)
                    cv2.imwrite(str(img_path), img_rgb)
                    self.csv_files[folder_key].write(f"{timestamp},{filename}\n")
                
                elif img_type == airsim.ImageType.Segmentation:
                    folder_key = "cam0_seg"
                    img_path = self.folders[folder_key] / filename
                    # 保存分割图像
                    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
                    img_seg = img1d.reshape(response.height, response.width, 3)
                    cv2.imwrite(str(img_path), img_seg)
                    self.csv_files[folder_key].write(f"{timestamp},{filename}\n")
    
    def fly_circle_z(self, start_z=-10, end_z=-1, num_steps=1000, sleep_time=0.1):
        """让无人机在z轴上从start_z飞行到end_z"""
        print("开始无人机飞行和数据采集")
        
        # 起飞
        self.client.takeoffAsync().join()
        
        # 固定的X和Y坐标
        x, y = 0, 0
        
        # 逐步从start_z移动到end_z
        for i in range(num_steps + 1):
            # 计算当前Z坐标
            z = start_z + (end_z - start_z) * (i / num_steps)
            
            # 设置无人机位姿
            pose = airsim.Pose(airsim.Vector3r(x, y, z), airsim.to_quaternion(pitch=0, roll=0, yaw=0))
            self.client.simSetVehiclePose(pose, True)
            
            # 生成时间戳（以纳秒为单位）
            timestamp = str(int(time.time() * 1e9))
            
            # 在当前位置收集所有数据
            self.collect_data(timestamp)
            
            # 等待以保持频率
            time.sleep(sleep_time)
        
        # 在终点位置悬停
        self.client.hoverAsync().join()
        
        # 关闭CSV文件
        for file in self.csv_files.values():
            file.close()
        
        print("飞行和数据采集完成")
    
    def cleanup(self):
        """清理资源"""
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        
        # 关闭任何打开的文件
        for file in self.csv_files.values():
            if not file.closed:
                file.close()

if __name__ == "__main__":
    try:
        collector = AirSimDataCollector()
        # 从-10m飞到-1m，1000步，10Hz频率(sleep_time=0.1)
        collector.fly_circle_z(start_z=-10, end_z=-1, num_steps=1000, sleep_time=0.1)
    except KeyboardInterrupt:
        print("\n用户中断了程序。")
    except Exception as e:        
        print(f"发生错误: {e}")
    finally:
        if 'collector' in locals():
            collector.cleanup()
        print("程序终止。")