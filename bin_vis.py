import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 文件路径
DEFAULT_BIN_PATH = 'airsim_data_convert/00/velodyne/1747898805774834432.bin'
DEFAULT_LABEL_PATH = 'airsim_data_convert/00/labels/1747898805774834432.label'

def read_bin_pointcloud(bin_path):
    """
    读取.bin格式的点云文件（每个点为float32的x,y,z,density）
    """
    points = np.fromfile(bin_path, dtype=np.float32)
    if points.size % 4 != 0:
        raise ValueError(f"点云数据长度不是4的倍数: {points.size}")
    points = points.reshape(-1, 4)
    # 打印前10个点的数据shape
    print(f"所有点云数据shape: {points[:10].shape}")
    print("前10个点的原始数据:")
    print(points[:10])
    return points
def read_label_file(label_path):
    """
    读取.label格式的标签文件（每个点为uint32的标签值）
    """
    labels = np.fromfile(label_path, dtype=np.uint32)
    print(f"标签数据shape: {labels.shape}")
    print("前10个标签值:")
    print(labels[:10])
    return labels

def visualize_pointcloud_with_labels(points, labels, title=None):
    """
    可视化带标签的点云
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 使用标签值作为颜色
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        s=1, c=labels, cmap='tab20')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)
    set_axes_equal(ax)  # 保证比例
    
    # 添加颜色条
    plt.colorbar(scatter)
    plt.show()


def visualize_pointcloud(points, title=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c=points[:, 2], cmap='jet')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if title:
        ax.set_title(title)
    plt.show()

def set_axes_equal(ax):
    '''Set 3D plot axes to equal scale.'''
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def main():
    bin_path = DEFAULT_BIN_PATH
    if not os.path.exists(bin_path):
        print(f"文件不存在: {bin_path}")
        return
    points = read_bin_pointcloud(bin_path)
    print(f"读取点数: {points.shape[0]}")
    label_path = DEFAULT_LABEL_PATH
    labels = read_label_file(label_path)
    visualize_pointcloud_with_labels(points, labels, title=os.path.basename(bin_path))

if __name__ == '__main__':
    main() 