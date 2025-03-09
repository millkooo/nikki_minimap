import time
import math
import numpy as np


class SpeedCalculator:
    """
    根据位置变化和时间计算速度的工具类
    """
    def __init__(self):
        # 上一个位置坐标
        self.last_position = None
        # 上一次记录的时间戳
        self.last_time = None
        # 速度历史记录(用于平滑计算)
        self.speed_history = []
        # 历史记录最大长度
        self.history_max_len = 5
        # 当前速度
        self.current_speed = 0
    
    def calculate_speed(self, current_position):
        """
        计算当前速度
        
        参数:
        current_position (tuple): 当前位置坐标 (x, y)
        
        返回:
        float: 当前速度 (像素/秒)
        """
        current_time = time.time()
        
        # 如果是首次记录位置，只保存不计算
        if self.last_position is None or self.last_time is None:
            self.last_position = current_position
            self.last_time = current_time
            return 0
        
        # 计算位置变化的欧几里得距离
        distance = math.sqrt(
            (current_position[0] - self.last_position[0]) ** 2 + 
            (current_position[1] - self.last_position[1]) ** 2
        )
        
        # 计算时间差
        time_diff = current_time - self.last_time
        
        # 防止除零错误
        if time_diff < 0.001:
            speed = 0
        else:
            speed = distance / time_diff  # 像素/秒
        
        # 更新历史记录
        self.speed_history.append(speed)
        if len(self.speed_history) > self.history_max_len:
            self.speed_history.pop(0)
        
        # 计算平均速度(滑动窗口)
        self.current_speed = sum(self.speed_history) / len(self.speed_history)
        
        # 更新上一次的位置和时间
        self.last_position = current_position
        self.last_time = current_time
        
        return self.current_speed
    
    def get_current_speed(self):
        """获取当前速度值"""
        return self.current_speed
    
    def get_formatted_speed(self, decimal_places=2):
        """获取格式化的速度字符串"""
        return f"{self.current_speed:.{decimal_places}f} px/s"
    
    def reset(self):
        """重置计算器状态"""
        self.last_position = None
        self.last_time = None
        self.speed_history = []
        self.current_speed = 0 