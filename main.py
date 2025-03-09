import logging
import cv2
import json
import os
from Ui_Manage.WindowManager import WinControl
from Ui_Manage.TransparentOverlay import TransparentOverlay
from PyQt5.QtWidgets import QApplication
import sys
from capture.img_processor import ImageProcessor
from match.template_matcher import TemplateMatcher
from minimap_match.FeatureMatch import FeatureMatcher
from minimap_match.AngleDetector import AdaptiveAngleDetector

# 新导入模块
from Ui_Manage.visualizer import ResultVisualizer
from navigation import NavigationState
from minimap_match.boundary_manager import BoundaryManager
from detection_pipeline import DetectionPipeline

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 读取配置文件
def load_config(config_path='config.json'):
    """从JSON文件加载配置"""
    try:
        if not os.path.exists(config_path):
            logger.error(f"配置文件不存在: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
        logger.info(f"成功加载配置文件: {config_path}")
        return config
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        sys.exit(1)

def main():
    """主程序"""
    # 加载配置
    CONFIG = load_config()
    
    # 初始化QApplication
    app = QApplication(sys.argv)
    
    # 初始化透明覆盖层
    overlay_config = CONFIG["window"]
    overlay = TransparentOverlay(overlay_config)
    overlay.show()
    
    # 初始化窗口控制
    target_hwnd = WinControl.find_target_window(CONFIG["window"])
    if not target_hwnd:
        logger.error("未找到目标窗口")
        return
    WinControl.activate_window(target_hwnd)
    
    # 初始化各组件
    image_processor = ImageProcessor(target_hwnd)
    template_matcher = TemplateMatcher()
    template_matcher.load_templates(CONFIG["templates"]["configs"])
    
    # 加载特征匹配器
    features_matcher = {}
    for region, path in CONFIG["matching"]["region_features"].items():
        features_matcher[region] = FeatureMatcher(path, CONFIG["matching"])
    
    # 加载模板图像
    huayan_img = cv2.imread('resources/img/huayanqundao.png', cv2.IMREAD_GRAYSCALE)
    xinyuanyuanye_img = cv2.imread('resources/img/nuanuan_map.png', cv2.IMREAD_GRAYSCALE)
    
    # 初始化各管理器
    state = NavigationState()
    angle_detector = AdaptiveAngleDetector(
        threshold_angle=45,
        angle_tolerance=5,
        initial_threshold=200,
        min_threshold=180,
        threshold_step=5,
        hough_params=(20, 16, 10),
        save_debug_images=False
    )
    boundary_manager = BoundaryManager(CONFIG)
    
    # 初始化检测流水线
    pipeline = DetectionPipeline(
        CONFIG, image_processor, template_matcher, features_matcher, angle_detector
    )
    
    try:
        while True:
            # 通过流水线处理当前帧
            best_match, angle = pipeline.process_frame(state, boundary_manager)
            
            # 更新显示
            if best_match:
                # 获取当前速度信息
                speed_text = state.speed_calculator.get_formatted_speed()
                
                # 更新透明覆盖层显示的位置、角度和速度
                overlay.update_position_info(best_match["center"], state.last_angle, speed_text)
                
                # 显示匹配结果
                if state.current_region == "huayanqundao":
                    ResultVisualizer.show_match_result(best_match, huayan_img, state.last_angle)
                else:
                    ResultVisualizer.show_match_result(best_match, xinyuanyuanye_img, state.last_angle)
            else:
                # 如果没有匹配结果，显示空值
                overlay.update_position_info(None, state.last_angle, "0.00 px/s")

            # 处理Qt事件
            QApplication.processEvents()
            
            if cv2.waitKey(1) == 27:
                break
                
    finally:
        cv2.destroyAllWindows()
        # 关闭应用
        sys.exit(app.exec_())

if __name__ == "__main__":
    main()