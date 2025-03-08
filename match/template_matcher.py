import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class TemplateMatcher:
    """模板匹配类，支持带透明度的图像匹配"""

    def __init__(self):
        self.templates = {}
        self.last_match = None

    def load_templates(self, template_configs):
        """加载模板配置
        :param template_configs: 模板配置列表，每个元素包含name, path, threshold
        """
        for config in template_configs:
            try:
                # 读取图像，保留透明通道
                template_img = cv2.imread(config["path"], cv2.IMREAD_UNCHANGED)
                if template_img is None:
                    logger.error(f"无法加载模板图像: {config['path']}")
                    continue

                # 存储模板信息
                self.templates[config["name"]] = {
                    "image": template_img,
                    "threshold": config["threshold"],
                    "path": config["path"]
                }
                logger.info(f"已加载模板: {config['name']}")
            except Exception as e:
                logger.error(f"加载模板 {config['name']} 失败: {str(e)}")

    def match_template(self, frame, template_name=None):
        """匹配模板
        :param frame: 输入图像帧
        :param template_name: 指定模板名称，如果为None则匹配所有模板
        :return: 匹配结果字典或None
        """
        best_match = None
        best_score = 0

        # 确定要匹配的模板
        templates_to_match = {}
        if template_name:
            if template_name in self.templates:
                templates_to_match[template_name] = self.templates[template_name]
            else:
                logger.warning(f"未找到指定模板: {template_name}")
                return None
        else:
            templates_to_match = self.templates

        # 对每个模板进行匹配
        for name, template_info in templates_to_match.items():
            template_img = template_info["image"]
            threshold = template_info["threshold"]
            
            # 处理带透明度的模板
            result = self._match_with_alpha(frame, template_img, threshold)
            
            if result and result["score"] > best_score:
                best_score = result["score"]
                best_match = {
                    "name": name,
                    "score": result["score"],
                    "location": result["location"],
                    "size": (template_img.shape[1], template_img.shape[0])
                }

        self.last_match = best_match
        return best_match

    def _match_with_alpha(self, frame, template, threshold):
        """带透明度的模板匹配
        :param frame: 输入图像
        :param template: 模板图像（可能带透明通道）
        :param threshold: 匹配阈值
        :return: 匹配结果或None
        """
        # 检查模板是否有透明通道
        has_alpha = template.shape[2] == 4 if len(template.shape) > 2 else False
        
        if has_alpha:
            # 分离RGB和Alpha通道
            bgr = template[:, :, 0:3]
            alpha = template[:, :, 3]
            
            # 创建掩码（只考虑非透明区域）
            # 将布尔掩码转换为uint8类型，OpenCV要求掩码为CV_8U类型
            mask = np.uint8(alpha > 0) * 255
            
            # 使用掩码进行模板匹配
            result = cv2.matchTemplate(frame, bgr, cv2.TM_CCORR_NORMED, mask=mask)
        else:
            # 无透明通道，直接匹配
            result = cv2.matchTemplate(frame, template, cv2.TM_CCORR_NORMED)
        
        # 查找最佳匹配位置
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # 使用相关系数匹配方法，值越大越好
        if max_val >= threshold:
            return {
                "score": max_val,
                "location": max_loc
            }
        return None