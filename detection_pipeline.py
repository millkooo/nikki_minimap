import logging
import time
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class DetectionPipeline:
    def __init__(self, config, image_processor, template_matcher, features_matcher, angle_detector):
        self.config = config
        self.image_processor = image_processor
        self.template_matcher = template_matcher
        self.features_matcher = features_matcher
        self.angle_detector = angle_detector
        
    def process_frame(self, state, boundary_manager):
        # 捕获当前帧
        frame = self.image_processor.capture_region({
            "x_offset": 0, "y_offset": 0, "width": 281, "height": 242
        })
        
        # 检查是否处于模板匹配状态
        if state.template_state["active"]:
            if state.should_check_template():
                template_result = self.template_matcher.match_template(
                    frame, state.template_state["current_template"]
                )
                
                if not template_result:
                    state.exit_template_state()
                    logger.info(f"模板 {state.template_state['current_template']} 不再匹配，恢复正常检测")
                else:
                    logger.debug(f"模板 {template_result['name']} 仍然匹配，得分: {template_result['score']:.2f}")
            return None, None  # 模板状态下不进行特征匹配
            
        # 处理从loading状态退出的情况
        if not state.template_state["active"] and state.template_state["previous_template"] == "loading":
            time.sleep(2)
            logger.info("检测到从loading状态退出，加载全部npz文件进行匹配")
            state.prev_result = None
            regions = list(self.config["matching"]["region_features"].keys())
            boundary_manager.update_matchers(regions, state, self.features_matcher)
            state.template_state["previous_template"] = None
        
        # 尝试模板匹配
        template_result = self.template_matcher.match_template(frame)
        if template_result:
            state.enter_template_state(template_result["name"])
            logger.info(f"检测到模板 {template_result['name']}，得分: {template_result['score']:.2f}，暂停SIFT检测")
            return None, None
            
        # 预处理图像用于特征匹配和角度检测
        frame1 = self.image_processor._crop_image(frame, {"x": 77, "y": 38, "w": 204, "h": 204})
        circular_frame = self.image_processor.circle_crop(frame1, self.config["capture"]["diameter"])
        crop_imgg = self.image_processor.circle_crop(circular_frame, 32)
        
        # 全局哈希检查
        current_hash = self.image_processor.compute_image_hash(circular_frame)
        if current_hash == state.prev_hash and state.prev_result:
            best_match = state.prev_result
        else:
            state.prev_hash = current_hash
            best_match = self._perform_feature_matching(state, circular_frame, boundary_manager)
            
        # 角度检测
        angle = self.angle_detector.calculate_angle(crop_imgg)
        if angle:
            state.last_angle = angle
            logger.debug(f"检测到角度: {angle}°")
            
        return best_match, angle
        
    def _perform_feature_matching(self, state, circular_frame, boundary_manager):
        """执行特征匹配"""
        best_match = None
        
        # 动态加载匹配器
        if state.current_region and state.prev_result:
            # 获取边界区域
            boundary_regions = boundary_manager.check_boundaries(*state.last_position, state)
            # 更新匹配器
            boundary_manager.update_matchers(boundary_regions, state, self.features_matcher, state.prev_result)
        else:
            # 初始状态加载所有匹配器
            regions = list(self.config["matching"]["region_features"].keys())
            boundary_manager.update_matchers(regions, state, self.features_matcher)
            
        # 执行特征匹配
        for matcher in state.active_matchers:
            result = matcher.process_frame(circular_frame)
            if result and result["matches"] >= self.config["matching"]["min_matches"]:
                if not best_match or result["matches"] > best_match["matches"]:
                    best_match = result
                    state.current_region = matcher.region_name
                    state.last_position = result["center"]
                    state.prev_result = best_match
                    
                    # 计算速度
                    state.speed_calculator.calculate_speed(result["center"])
                    
        return best_match 