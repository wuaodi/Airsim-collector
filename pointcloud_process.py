import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import math

def read_asc_pointcloud(filepath):
    """读取.asc格式的点云文件"""
    points = []
    
    with open(filepath, 'r') as f:
        for line in f:
            # 使用逗号分割，并去除首尾空格/空值
            values = [x.strip() for x in line.strip().split(',') if x.strip()]
            
            if len(values) >= 3:  # x, y, z
                try:
                    x, y, z = map(float, values[:3])
                    points.append([x, y, z])
                except ValueError as e:
                    print(f"格式错误: {values} -> {e}")
    
    points = np.array(points, dtype=np.float32)
    print(f"总读取点数: {len(points)}")
    return points

def transform_lidar_to_camera_frame(point):
    """
    将激光雷达坐标系下的点转换到相机坐标系
    """
    # 激光雷达相对于相机的平移向量
    translation = np.array([0, 0, -0.1])
    # 应用平移
    transformed_point = point + translation
    return transformed_point

def project_point_to_image(point, image_width, image_height):
    """
    将3D点投影到图像平面上
    注意：此函数接收的点应该已经在相机坐标系下，前右下为XYZ
    """
    # 根据提供的配置参数:
    # FOV_Degrees = 60
    # 图像尺寸为 2048x2048
    
    # 计算焦距 (像素)
    fov_rad = math.radians(50)
    focal_length = (image_width / 2) / math.tan(fov_rad / 2)
    
    # 确保点在相机前方 (X轴正方向)
    if point[0] <= 0:
        return None
    
    cx = image_width / 2
    cy = image_height / 2
    
    # 投影
    u = int(focal_length * point[1] / point[0] + cx) # 右
    v = int(focal_length * point[2] / point[0] + cy) # 下
    
    # 检查是否在图像范围内
    if 0 <= u < image_width and 0 <= v < image_height:
        return (u, v)
    return None

def get_label_from_segmentation(seg_image, u, v):
    """从语义分割图像中获取标签"""
    # 读取像素值
    pixel_value = seg_image[v, u]
    
    # Airsim语义分割图像是单通道的，每个像素值对应一个类别
    # 如果您的语义图像是RGB格式的，需要将RGB值映射到标签ID
    
    if len(seg_image.shape) == 3:  # RGB图像
        # 这里需要根据您的Airsim语义设置定义颜色到标签的映射
        # 示例映射:
        color_to_label = {
            (11, 236, 9): 0,     # 主体
            (146, 52, 70): 1,    # 左帆板
            (29, 26, 199): 2     # 右帆板
        }
        
        # 找到最接近的颜色
        min_dist = float('inf')
        label = 0
        pixel_tuple = tuple(pixel_value)
        
        for color, lbl in color_to_label.items():
            dist = sum((pixel_tuple[i] - color[i])**2 for i in range(3))
            if dist < min_dist:
                min_dist = dist
                label = lbl
        return label
    else:  # 单通道图像
        return int(pixel_value)

def convert_airsim_to_kitti(pointcloud_dir, seg_dir, output_bin_dir, output_label_dir):
    """将Airsim点云数据转换为Semantic KITTI格式"""
    
    # 创建输出目录
    os.makedirs(output_bin_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 获取所有点云文件
    pointcloud_files = sorted(glob(os.path.join(pointcloud_dir, "*.asc")))
    
    for pc_file in tqdm(pointcloud_files, desc="Converting files"):
        # 从文件名中提取ID
        file_id = os.path.splitext(os.path.basename(pc_file))[0]
        
        # 查找对应的语义分割图像
        seg_file = os.path.join(seg_dir, f"{file_id}.png")
        
        if not (os.path.exists(seg_file)):
            print(f"Warning: Could not find corresponding images for {file_id}")
            continue
        
        # 读取点云
        points = read_asc_pointcloud(pc_file)
        
        # 读取语义分割图像
        seg_image = cv2.imread(seg_file)
        height, width = seg_image.shape[:2]
        
        # 初始化标签数组
        labels = np.zeros(len(points), dtype=np.uint32)
        
        # 将点投影到图像平面并获取标签
        for i, point in enumerate(points):
            # 将激光雷达点转换到相机坐标系
            camera_point = transform_lidar_to_camera_frame(point)
            
            # 投影点到图像平面
            projection = project_point_to_image(camera_point, width, height)
            if projection:
                u, v = projection
                # 获取标签
                labels[i] = get_label_from_segmentation(seg_image, u, v)
        
        # 创建XYZ点云数据，增加强度值（用0填充）
        xyz = np.zeros((len(points), 4), dtype=np.float32)
        xyz[:, :3] = points  # 复制XYZ坐标
        xyz[:, 3] = 0.0      # 强度值设为0
        
        # 保存为bin文件
        output_bin_file = os.path.join(output_bin_dir, f"{file_id}.bin")
        xyz.tofile(output_bin_file)
        
        # 保存标签为label文件
        output_label_file = os.path.join(output_label_dir, f"{file_id}.label")
        labels.tofile(output_label_file)
        
        print(f"Processed {file_id}")

if __name__ == "__main__":
    # 配置路径
    pointcloud_dir = "D:\project\Airsim-collector\\airsim_data\lidar0\data"  # .asc点云文件夹
    seg_dir = "D:\project\Airsim-collector\\airsim_data\cam0_Seg\data"  # 语义分割图像文件夹
    output_bin_dir = "airsim_data_convert/00/velodyne"    # 输出的bin文件夹，仿照semantic_example结构
    output_label_dir = "airsim_data_convert/00/labels"  # 输出的label文件夹，仿照semantic_example结构

    # 执行转换
    convert_airsim_to_kitti(pointcloud_dir, seg_dir, output_bin_dir, output_label_dir)
    
    print("Conversion completed!")