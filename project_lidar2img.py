import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pointcloud_process import *

def visualize_projection(points_cloud, image_path):
    """可视化点云在图像上的投影"""
    # 读取RGB图像
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色通道
    h, w = img.shape[:2]

    # 创建画布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # 显示原始图像
    ax1.imshow(img)
    ax1.set_title('Original Image')

    # 显示带投影点的图像
    projected_img = img.copy()

    # 投影所有有效点
    valid_points = []
    for point in points_cloud:
        # 坐标系转换
        transformed_point = transform_lidar_to_camera_frame(point)
        # 投影到图像
        uv = project_point_to_image(transformed_point, w, h)
        
        if uv:
            valid_points.append(uv)
            # 在图像上画点（红色，5px大小）
            cv2.circle(projected_img, uv, radius=3, color=(255, 0, 0), thickness=5)

    # 显示投影结果
    ax2.imshow(projected_img)
    ax2.set_title(f'Projected Points ({len(valid_points)} valid)')
    plt.show()

def process_files_in_folder(img_path, lidar_path):
    """处理文件夹内所有.asc和.png文件"""
    # 获取所有 .asc 文件
    asc_files = [f for f in os.listdir(lidar_path) if f.endswith('.asc')]
    # 获取所有 .png 文件
    png_files = [f for f in os.listdir(img_path) if f.endswith('.png')]

    # 确保 .asc 和 .png 文件的数量一致
    for asc_file in asc_files:
        # 找到对应的 .png 文件
        corresponding_png = asc_file.replace('.asc', '.png')
        if corresponding_png in png_files:
            # 构造完整的文件路径
            pointcloud_path = os.path.join(lidar_path, asc_file)
            image_path = os.path.join(img_path, corresponding_png)
            print(f"Processing {image_path}...")

            # 读取点云数据
            points = read_asc_pointcloud(pointcloud_path)
            
            # 执行可视化
            print(f"Processing {asc_file} and {corresponding_png}...")
            visualize_projection(points, image_path)

# 使用示例 -----------------------------------------------------------------
if __name__ == "__main__":
    # 输入文件夹路径（需要修改为实际路径）
    img_path = "imgdir"
    lidar_path = "lidardir"

    # 处理文件夹内的所有文件
    process_files_in_folder(img_path, lidar_path)
