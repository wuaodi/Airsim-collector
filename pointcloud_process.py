import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm

def read_asc_pointcloud(filepath):
    """读取.asc格式的点云文件"""
    points = []
    intensities = []
    
    with open(filepath, 'r') as f:
        for line in f:
            values = line.strip().split()
            if len(values) >= 4:  # x, y, z, intensity
                x, y, z, intensity = map(float, values[:4])
                points.append([x, y, z])
                intensities.append(intensity)
    
    points = np.array(points, dtype=np.float32)
    intensities = np.array(intensities, dtype=np.float32)
    
    return points, intensities

def project_point_to_image(point, image_width, image_height):
    """
    将3D点投影到图像平面上
    这里假设相机和雷达没有外参差异，使用简单的投影
    实际应用中可能需要更复杂的相机模型和外参
    """
    # 这是一个简化的针孔相机模型
    # 假设视场角为90度，焦距为image_width/2
    focal_length = image_width / 2
    
    # 确保点在相机前方
    if point[0] <= 0:
        return None
    
    # 简单投影
    u = int(focal_length * point[1] / point[0] + image_width / 2)
    v = int(focal_length * point[2] / point[0] + image_height / 2)
    
    # 检查是否在图像范围内
    if 0 <= u < image_width and 0 <= v < image_height:
        return (u, v)
    return None

def get_label_from_segmentation(seg_image, u, v):
    """从语义分割图像中获取标签"""
    # 读取像素值
    pixel_value = seg_image[v, u]
    
    # 如果是彩色图像，需要转换为标签
    # 这里需要根据Airsim的语义图像映射规则进行转换
    # 例如，可以使用颜色映射表将RGB值映射到语义类别
    
    # 简单示例: 假设语义图像中的像素值已经是标签ID
    # 实际应用中可能需要更复杂的映射逻辑
    if len(pixel_value.shape) > 0:  # RGB图像
        # 使用简单的颜色到类别的映射
        # 这里需要根据您的实际数据进行调整
        color_to_label = {
            (0, 0, 0): 0,      # 背景
            (128, 64, 128): 1, # 道路
            (244, 35, 232): 2, # 人行道
            (70, 70, 70): 3,   # 建筑
            # 添加更多映射...
        }
        
        # 找到最接近的颜色
        min_dist = float('inf')
        label = 0
        for color, lbl in color_to_label.items():
            dist = sum((pixel_value[i] - color[i])**2 for i in range(3))
            if dist < min_dist:
                min_dist = dist
                label = lbl
        return label
    else:  # 灰度图像
        return int(pixel_value)

def normalize_intensity(intensity_array):
    """归一化强度值到0-1范围"""
    min_val = np.min(intensity_array)
    max_val = np.max(intensity_array)
    
    if max_val == min_val:
        return np.zeros_like(intensity_array)
    
    normalized = (intensity_array - min_val) / (max_val - min_val)
    # 确保值在0.0-0.99范围内
    normalized = np.round(normalized, decimals=2)
    normalized = np.minimum(normalized, 0.99)
    
    return normalized

def convert_airsim_to_kitti(pointcloud_dir, rgb_dir, seg_dir, output_bin_dir, output_label_dir):
    """将Airsim点云数据转换为Semantic KITTI格式"""
    
    # 创建输出目录
    os.makedirs(output_bin_dir, exist_ok=True)
    os.makedirs(output_label_dir, exist_ok=True)
    
    # 获取所有点云文件
    pointcloud_files = sorted(glob(os.path.join(pointcloud_dir, "*.asc")))
    
    for pc_file in tqdm(pointcloud_files, desc="Converting files"):
        # 从文件名中提取ID
        file_id = os.path.splitext(os.path.basename(pc_file))[0]
        
        # 查找对应的RGB和语义分割图像
        rgb_file = os.path.join(rgb_dir, f"{file_id}.jpg")
        seg_file = os.path.join(seg_dir, f"{file_id}.jpg")
        
        if not (os.path.exists(rgb_file) and os.path.exists(seg_file)):
            print(f"Warning: Could not find corresponding images for {file_id}")
            continue
        
        # 读取点云
        points, intensities = read_asc_pointcloud(pc_file)
        
        # 读取语义分割图像
        seg_image = cv2.imread(seg_file)
        height, width = seg_image.shape[:2]
        
        # 初始化标签数组
        labels = np.zeros(len(points), dtype=np.int32)
        
        # 将点投影到图像平面并获取标签
        for i, point in enumerate(points):
            # 投影点到图像平面
            projection = project_point_to_image(point, width, height)
            if projection:
                u, v = projection
                # 获取标签
                labels[i] = get_label_from_segmentation(seg_image, u, v)
        
        # 归一化强度值
        normalized_intensities = normalize_intensity(intensities)
        
        # 创建XYZI点云数据
        xyzi = np.column_stack((points, normalized_intensities)).astype(np.float32)
        
        # 保存为bin文件
        output_bin_file = os.path.join(output_bin_dir, f"{file_id}.bin")
        xyzi.tofile(output_bin_file)
        
        # 保存标签为label文件
        output_label_file = os.path.join(output_label_dir, f"{file_id}.label")
        labels.tofile(output_label_file)
        
        print(f"Processed {file_id}")

if __name__ == "__main__":
    # 配置路径
    pointcloud_dir = "path/to/asc/files"  # .asc点云文件夹
    rgb_dir = "path/to/rgb/images"        # RGB图像文件夹
    seg_dir = "path/to/segmentation/images"  # 语义分割图像文件夹
    output_bin_dir = "path/to/output/bin"    # 输出的bin文件夹
    output_label_dir = "path/to/output/label"  # 输出的label文件夹
    
    # 执行转换
    convert_airsim_to_kitti(pointcloud_dir, rgb_dir, seg_dir, output_bin_dir, output_label_dir)
    
    print("Conversion completed!")