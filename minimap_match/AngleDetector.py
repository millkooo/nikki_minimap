import cv2
import numpy as np
import os
import logging

# 配置日志
logger = logging.getLogger(__name__)

class AdaptiveAngleDetector:

    def __init__(self,
                 threshold_angle=45,  # 目标夹角为45度
                 angle_tolerance=5,   # 角度容差增加到±5度
                 initial_threshold=220,  # 降低初始阈值
                 min_threshold=180,    # 降低最小阈值
                 threshold_step=20,    # 增加阈值递减步长
                 dilation_kernel_size=3,  # 膨胀核大小
                 hough_params=(30, 5, 10),  # 调整霍夫变换参数：降低阈值，减小最小线长，增加最大间隙
                 save_debug_images=True):  # 默认保存调试图像
        # 算法参数配置
        self.threshold_angle = threshold_angle
        self.angle_tolerance = angle_tolerance
        self.initial_threshold = initial_threshold
        self.min_threshold = min_threshold
        self.threshold_step = threshold_step
        self.dilation_kernel_size = dilation_kernel_size
        self.hough_threshold, self.min_length, self.max_gap = hough_params
        self.save_debug_images = save_debug_images
        self.result_dir = "angle_detection_results"
        
        if self.save_debug_images:
            os.makedirs(self.result_dir, exist_ok=True)

    def calculate_angle(self, image):
        """
        计算图像中箭头的角度，兼容main.py中的接口
        :param image: OpenCV格式的图像对象
        :return: 检测到的角度（整数），如果未检测到则返回None
        """
        result = self.detect_angle(image, simple_mode=True)
        if result is not None:
            # 记录检测到的角度
            logger.info(f"成功检测到角度: {int(result)}°")
            return int(result)
        logger.debug("未能检测到角度，返回None")
        return None

    def detect_angle(self, image, simple_mode=False):
        """
        检测图像中箭头的角度
        :param image: 图像路径或OpenCV格式的图像对象
        :param simple_mode: 简单模式，只返回角度值
        :return: 如果simple_mode为True，则只返回角度值；否则返回(角度, 最佳线段对, 交点, 中心点)
        """
        # 处理输入图像（可以是路径字符串或图像对象）
        if isinstance(image, str):
            # 如果是路径字符串，读取图像
            img = cv2.imread(image)
            if img is None:
                logger.error(f"无法读取图像: {image}")
                return None
        else:
            # 如果已经是图像对象，直接使用
            img = image.copy()

        # 获取图像中心
        height, width = img.shape[:2]
        center = (width // 2, height // 2)  # 中心点

        # 保存原始图像（如果需要）
        if self.save_debug_images:
            cv2.imwrite(os.path.join(self.result_dir, "original.jpg"), img)

        # 自适应阈值处理
        current_threshold = self.initial_threshold
        angle = None
        best_lines = None
        best_vertex = None

        while current_threshold >= self.min_threshold and angle is None:
            logger.debug(f"尝试阈值: {current_threshold}")
            
            # 转换为灰度图
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # 二值化
            _, binary = cv2.threshold(gray, current_threshold, 255, cv2.THRESH_BINARY)
            
            # 保存二值化结果（如果需要）
            if self.save_debug_images:
                cv2.imwrite(os.path.join(self.result_dir, f"binary_threshold_{current_threshold}.jpg"), binary)
            
            # 膨胀处理，连接断线
            kernel = np.ones((self.dilation_kernel_size, self.dilation_kernel_size), np.uint8)
            dilated = cv2.dilate(binary, kernel, iterations=1)
            
            # 保存膨胀处理结果（如果需要）
            if self.save_debug_images:
                cv2.imwrite(os.path.join(self.result_dir, f"dilated_threshold_{current_threshold}.jpg"), dilated)
            
            # 霍夫变换检测直线
            lines = cv2.HoughLinesP(dilated, 1, np.pi / 180,
                                    threshold=self.hough_threshold,
                                    minLineLength=self.min_length,
                                    maxLineGap=self.max_gap)
            
            # 异常处理
            if lines is None or len(lines) < 2:
                logger.debug(f"阈值 {current_threshold} 未检测到足够的直线")
                current_threshold -= self.threshold_step
                continue
            
            # 转换线段格式并按长度排序
            lines = [line[0] for line in lines]
            lines_with_length = [(line, self._calculate_line_length(line)) for line in lines]
            lines_with_length.sort(key=lambda x: x[1], reverse=True)  # 按长度降序排序
            
            # 获取最长的线段
            sorted_lines = [line for line, _ in lines_with_length]
            
            # 绘制检测到的所有直线（如果需要）
            if self.save_debug_images:
                lines_img = img.copy()
                for line in sorted_lines:
                    x1, y1, x2, y2 = line
                    cv2.line(lines_img, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.imwrite(os.path.join(self.result_dir, f"all_lines_threshold_{current_threshold}.jpg"), lines_img)
            
            # 寻找符合条件的直线对
            result = self._find_best_angle_pair(sorted_lines, center)
            
            if result:
                best_pair, angle = result
                line1, line2, vertex_x, vertex_y = best_pair
                best_lines = (line1, line2)
                best_vertex = (vertex_x, vertex_y)
                
                # 绘制最佳直线对和方向（如果需要）
                if self.save_debug_images:
                    result_img = img.copy()
                    # 绘制两条最佳直线
                    x1, y1, x2, y2 = line1
                    cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    x1, y1, x2, y2 = line2
                    cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    
                    # 绘制交点
                    cv2.circle(result_img, (int(vertex_x), int(vertex_y)), 2, (255, 0, 0), -1)
                    
                    # 绘制中心点和方向线
                    cv2.line(result_img, center, (int(vertex_x), int(vertex_y)), (255, 255, 0), 1)
                    cv2.circle(result_img, center, 1, (255, 0, 255), -1)
                    
                    # 添加文本说明
                    cv2.putText(result_img, f"Angle: {angle:.1f} degrees", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    cv2.imwrite(os.path.join(self.result_dir, f"result_threshold_{current_threshold}.jpg"), result_img)
                break
            else:
                logger.debug(f"阈值 {current_threshold} 未找到符合条件的直线对")
                current_threshold -= self.threshold_step
        
        if angle is not None:
            logger.debug(f"检测到的角度: {angle}°")
            if simple_mode:
                return angle
            return angle, best_lines, best_vertex, center
        else:
            logger.debug("未能检测到角度")
            return None

    def _calculate_line_length(self, line):
        """
        计算线段长度
        :param line: 线段坐标 [x1, y1, x2, y2]
        :return: 线段长度
        """
        x1, y1, x2, y2 = line
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _find_best_angle_pair(self, lines, center):
        """
        寻找最佳角度对
        :param lines: 线段列表
        :param center: 中心点坐标
        :return: (最佳线段对, 角度) 或 None
        """
        best_pair = None
        best_diff = float('inf')
        best_angle = None

        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                line1, line2 = lines[i], lines[j]
                vertex = self._calculate_intersection(line1, line2)

                if vertex is None:
                    continue  # 忽略平行线

                # 计算两条线的夹角
                angle_between = self._calculate_angle_between(line1, line2)
                angle_diff = abs(angle_between - self.threshold_angle)

                # 检查夹角是否在容差范围内
                if angle_diff <= self.angle_tolerance and angle_diff < best_diff:
                    # 计算交点到中心的方向
                    final_angle = self._calculate_final_angle((line1, line2, *vertex), center)
                    best_diff = angle_diff
                    best_pair = (line1, line2, *vertex)
                    best_angle = final_angle

        if best_pair:
            return best_pair, best_angle
        return None

    def _calculate_intersection(self, line1, line2):
        """
        计算两条线段的交点
        :param line1: 第一条线段坐标 [x1, y1, x2, y2]
        :param line2: 第二条线段坐标 [x1, y1, x2, y2]
        :return: 交点坐标 (x, y) 或 None（如果平行）
        """
        # 解包线段坐标
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2

        dx1, dy1 = x2 - x1, y2 - y1
        dx2, dy2 = x4 - x3, y4 - y3
        
        # 计算交点
        denominator = dx2 * dy1 - dx1 * dy2
        if denominator == 0:
            return None  # 平行线跳过
            
        # 计算参数t和s
        t_numerator = dx2 * (y3 - y1) + dy2 * (x1 - x3)
        s_numerator = dx1 * (y3 - y1) + dy1 * (x1 - x3)
        t = t_numerator / denominator
        s = s_numerator / denominator

        # 计算交点坐标
        ix = x1 + t * dx1
        iy = y1 + t * dy1
        return ix, iy

    def _calculate_angle_between(self, line1, line2):
        """
        计算两条线段之间的夹角
        :param line1: 第一条线段坐标 [x1, y1, x2, y2]
        :param line2: 第二条线段坐标 [x1, y1, x2, y2]
        :return: 夹角（度）
        """
        # 计算两向量夹角
        vec1 = np.array([line1[2] - line1[0], line1[3] - line1[1]])
        vec2 = np.array([line2[2] - line2[0], line2[3] - line2[1]])

        cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        angle = np.degrees(np.arccos(np.clip(cos_theta, -1, 1)))
        return min(angle, 180 - angle)  # 始终返回锐角

    def _calculate_final_angle(self, best_pair, center):
        """
        计算最终角度
        :param best_pair: 最佳线段对和交点 (line1, line2, ix, iy)
        :param center: 中心点坐标 (x, y)
        :return: 角度（度）
        """
        _, _, ix, iy = best_pair
        dx = ix - center[0]
        dy = center[1] - iy  # 转换为数学坐标系
        
        # 计算角度并确保结果在0-360度范围内
        angle = np.degrees(np.arctan2(dy, dx)) % 360
        
        # 添加调试信息
        logger.debug(f"计算角度: dx={dx}, dy={dy}, 原始角度={angle}°")
        
        return angle
