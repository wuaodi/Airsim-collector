import redis
import json
import time
import math

def simulate_vertical_path():
    """模拟发送垂直路径的位姿命令，与fly_gt_sensor.py中的路径一致"""
    # 连接到Redis
    try:
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
        redis_client.ping()  # 测试连接
    except redis.ConnectionError:
        print("无法连接到Redis服务器")
        return
    except Exception as e:
        print(f"Redis连接错误: {e}")
        return
        
    pose_channel = 'drone_pose'
    
    # 垂直路径参数（与fly_gt_sensor.py一致）
    x, y = 0, 0  # 固定的X和Y坐标
    start_z = -10
    end_z = -2
    num_steps = 30  # 总步数
    
    print("开始发送垂直路径位姿命令...")
    print(f"路径: z从{start_z}到{end_z}，共{num_steps+1}个点")
    
    try:
        # 逐步从start_z移动到end_z
        for i in range(num_steps + 1):
            # 计算当前Z坐标（与fly_gt_sensor.py中的计算方式一致）
            z = start_z + (end_z - start_z) * (i / num_steps)
            
            # 创建位姿数据
            pose_data = {
                'x': round(x, 3),  # 保留3位小数
                'y': round(y, 3),
                'z': round(z, 3),
                'pitch': 0.0,
                'roll': 0.0,
                'yaw': 0.0
            }
            
            try:
                # 发送到Redis
                redis_client.set(pose_channel, json.dumps(pose_data))
                print(f"步骤 {i+1}/{num_steps+1} - 发送位姿: {pose_data}")
            except redis.RedisError as e:
                print(f"Redis发送错误: {e}")
                break
            except Exception as e:
                print(f"发送数据时发生错误: {e}")
                break
            
            # 每3秒发送一次
            time.sleep(3)
        
        print("垂直路径发送完成！")
        
    except KeyboardInterrupt:
        print("\n停止发送位姿命令")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        try:
            redis_client.close()
        except:
            pass


if __name__ == "__main__":   
    simulate_vertical_path()