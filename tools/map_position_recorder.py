import logging
import time
import os
import datetime
import json
from pynput.keyboard import Listener, Key
import sys
import cv2
import numpy as np

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Ui_Manage.WindowManager import WinControl
from capture.img_processor import ImageProcessor
from minimap_match.FeatureMatch import FeatureMatcher
from minimap_match.AngleDetector import AdaptiveAngleDetector

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MapPositionRecorder:
    """小地图位置记录器，按下"."键时记录当前位置和角度"""
    
    def __init__(self, config):
        self.config = config
        self.target_hwnd = None
        self.image_processor = None
        self.features_matcher = {}
        self.current_region = None
        self.last_position = (0, 0)
        self.last_angle = 0
        self.keyboard_listener = None
        self.running = True
        
        # 初始化角度检测器
        self.angle_detector = AdaptiveAngleDetector()
        
        # 确保waypoints目录存在
        self.waypoints_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources", "waypoints")
        if not os.path.exists(self.waypoints_dir):
            os.makedirs(self.waypoints_dir)
        
        # 初始化JSON文件
        self.json_file = os.path.join(self.waypoints_dir, "waypoints.json")
        self.waypoints_data = {"points": []}
        self._save_json_data()
        
        logger.info(f"位置记录器已初始化，JSON文件: {self.json_file}")
    
    def initialize(self):
        """初始化窗口和匹配器"""
        # 查找目标窗口
        self.target_hwnd = WinControl.find_target_window(self.config["window"])
        if not self.target_hwnd:
            logger.error("未找到目标窗口")
            return False
        
        # 激活窗口并初始化图像处理器
        WinControl.activate_window(self.target_hwnd)
        self.image_processor = ImageProcessor(self.target_hwnd)
        
        # 初始化特征匹配器
        for region, path in self.config["matching"]["region_features"].items():
            self.features_matcher[region] = FeatureMatcher(path, self.config["matching"])
        
        # 设置键盘监听
        self._setup_keyboard_listener()
        
        logger.info("初始化完成，开始监听键...")
        return True
    
    def _setup_keyboard_listener(self):
        """设置键盘监听器"""
        def on_press(key):
            try:
                if hasattr(key, 'char') and key.char == '.':
                    self._record_current_position(include_metadata=True)
                elif hasattr(key, 'char') and key.char == ',':
                    self._record_current_position(include_metadata=False)
                elif key == Key.esc:
                    logger.info("检测到ESC键，退出程序")
                    self.running = False
                    return False
            except AttributeError:
                pass
        
        # 启动键盘监听
        self.keyboard_listener = Listener(on_press=on_press)
        self.keyboard_listener.daemon = True
        self.keyboard_listener.start()
    
    def _record_current_position(self, include_metadata=True):
        """记录当前位置和角度
        
        Args:
            include_metadata: 是否包含区域和角度信息，按","键时为False，仅保存坐标
        """
        if not self.last_position:
            logger.warning("当前没有有效的位置信息")
            return
        
        x, y = self.last_position
        
        # 创建点位数据
        point_data = {"x": x, "y": y}
        
        # 如果需要包含元数据，则添加区域和角度信息
        if include_metadata and self.current_region:
            point_data["region"] = self.current_region
            point_data["angle"] = self.last_angle
            log_msg = f"已记录位置 - 区域: {self.current_region}, 坐标: ({x}, {y}), 角度: {self.last_angle}°"
        else:
            log_msg = f"已记录坐标: ({x}, {y})"
        
        # 添加到数据列表并保存
        self.waypoints_data["points"].append(point_data)
        self._save_json_data()
        
        logger.info(log_msg)
    
    def process_frame(self):
        """处理当前帧并更新位置信息"""
        # 捕获小地图区域
        frame = self.image_processor.capture_region({ "x_offset": 77,"y_offset": 38,"width": 204,"height": 204})
        circular_frame = ImageProcessor.circle_crop(frame, self.config["capture"]["diameter"])
        
        # 裁剪中心区域用于角度检测
        crop_center = ImageProcessor.circle_crop(circular_frame, 32)
        
        # 特征匹配
        best_match = None
        best_match_count = 0
        
        for region, matcher in self.features_matcher.items():
            result = matcher.process_frame(circular_frame)
            if result and result["matches"] >= self.config["matching"]["min_matches"]:
                if not best_match or result["matches"] > best_match_count:
                    best_match = result
                    best_match_count = result["matches"]
                    self.current_region = region
                    self.last_position = result["center"]
                    # 从特征匹配结果中获取初步角度
                    feature_angle = result["angle"]
                    logger.debug(f"特征匹配角度: {feature_angle}°")
        
        # 使用角度检测器获取更精确的角度
        detected_angle = self.angle_detector.calculate_angle(crop_center)
        if detected_angle:
            logger.debug(f"角度检测器角度: {detected_angle}°")
            self.last_angle = detected_angle
        elif best_match:

            self.last_angle = best_match["angle"]
            logger.debug(f"使用特征匹配角度: {self.last_angle}°")
        
        return best_match is not None
    
    def _save_json_data(self):
        """保存JSON数据到文件"""
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(self.waypoints_data, f, ensure_ascii=False, indent=2)
    
    def run(self):
        """主循环"""
        if not self.initialize():
            return
        
        try:
            while self.running:
                self.process_frame()
                time.sleep(0.1)  # 降低CPU使用率
        except KeyboardInterrupt:
            logger.info("程序被用户中断")
        finally:
            if self.keyboard_listener:
                self.keyboard_listener.stop()
            logger.info(f"程序结束，位置记录已保存到 {self.json_file}")

# 默认配置
DEFAULT_CONFIG = {
    "window": {
        "title_part": "无限暖暖",
        "window_class": "UnrealWindow",
        "process_exe": "X6Game-Win64-Shipping.exe"
    },
    "capture": {
        "list": [{ "x_offset": 77, "y_offset": 38, "width": 204, "height": 204 }],
        "diameter": 204
    },
    "matching": {
        "region_features": {
            "huayuanzhen": "resources/features/huayuanzhen_features.npz",
            "qiyuansenlin": "resources/features/qiyuansenlin_features.npz",
            "weifenglvye": "resources/features/weifenglvye_features.npz",
            "xiaoshishu": "resources/features/xiaoshishu_features.npz",
            "panduan": "resources/features/panduan_features.npz",
            "huayanqundao": "resources/features/huayanqundao_features.npz"
        },
        "min_matches": 10,
        "flann_checks": 50,
        "match_ratio": 0.8,
        "max_angle": 5
    }
}

def main():
    recorder = MapPositionRecorder(DEFAULT_CONFIG)
    recorder.run()

if __name__ == "__main__":
    main()