import cv2
import numpy as np
import win32con
import win32gui
import logging

logger = logging.getLogger(__name__)

class ResultVisualizer:
    """结果可视化类，负责结果显示和窗口管理"""
    
    @staticmethod
    def show_match_result(result: dict, template_image: np.ndarray, angle):
        """显示匹配结果"""
        display_img = template_image.copy()
        M = result["transform"]
        center = result["center"]

        # 绘制变换后的边界框
        h, w = template_image.shape[:2]
        corners = np.array([
            [-w / 2, -h / 2], [w / 2, -h / 2],
            [w / 2, h / 2], [-w / 2, h / 2]
        ])
        rotated_corners = np.dot(corners, M[:2, :2].T) + [M[0, 2], M[1, 2]]
        pts = np.int32(rotated_corners.reshape(-1, 1, 2))
        cv2.polylines(display_img, [pts], True, (0, 255, 0), 2)

        # 绘制中心点
        cv2.circle(display_img, center, 5, (0, 0, 255), -1)
        print(angle)
        # 将角度转换为弧度
        angle_rad = np.deg2rad(angle)
        # 计算箭头终点坐标
        start_x = center[0]- 8 * np.cos(angle_rad)
        start_y = center[1]+ 8 * np.sin(angle_rad)
        end_x = center[0] + 14 * np.cos(angle_rad)
        end_y = center[1] - 14 * np.sin(angle_rad)
        start_point = (int(start_x),int(start_y))

        end_point = (int(end_x), int(end_y))

        # 绘制箭杆
        cv2.line(display_img, start_point, end_point, (255, 255, 0), 2)

        # 计算箭头头的两个辅助点
        # 箭头头的方向
        theta_rad = np.deg2rad(25)
        left_angle = angle_rad + np.pi - theta_rad  # 左分支角度
        right_angle = angle_rad + np.pi + theta_rad  # 右分支角度

        # 计算左右分支端点
        left_end = (
            int(end_x + 8 * np.cos(left_angle)),
            int(end_y - 8 * np.sin(left_angle))  # 注意保持y轴方向一致
        )

        right_end = (
            int(end_x + 8 * np.cos(right_angle)),
            int(end_y - 8 * np.sin(right_angle))
        )
        # 绘制箭头分支
        cv2.line(display_img, end_point, left_end, (255, 255, 0), 2)
        cv2.line(display_img, end_point, right_end, (255, 255, 0), 2)
        # 创建显示区域
        cropped = ResultVisualizer._create_viewport(display_img, center, 100)
        cv2.putText(cropped, f'Matches: {result["matches"]}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示结果
        cv2.namedWindow('Matching Result', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Matching Result', 200, 200)
        cv2.imshow('Matching Result', cropped)
        ResultVisualizer.set_topmost('Matching Result')

    @staticmethod
    def _create_viewport(image: np.ndarray, center: tuple, size: int) -> np.ndarray:
        """创建视口区域"""
        h, w = image.shape[:2]
        x1 = max(0, center[0] - size)
        y1 = max(0, center[1] - size)
        x2 = min(w, center[0] + size)
        y2 = min(h, center[1] + size)
        return image[y1:y2, x1:x2]

    @staticmethod
    def set_topmost(window_name: str):
        """设置窗口置顶"""
        hwnd = win32gui.FindWindow(None, window_name)
        if hwnd:
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
