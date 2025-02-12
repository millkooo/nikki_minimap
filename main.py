import logging
import time
from ctypes import windll

import cv2
import mss
import numpy as np
import psutil
import pythoncom
import win32api
import win32con
import win32gui
import win32process
from PIL import Image

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量配置
DEFAULT_CONFIG = {
    "window": {
        "title_part": "无限暖暖",
        "window_class": "UnrealWindow",
        "process_exe": "X6Game-Win64-Shipping.exe"
    },
    "capture": {
        "x_offset": 74,
        "y_offset": 35,
        "width": 206,
        "height": 206,
        "diameter": 220
    },
    "matching": {
        "min_matches": 10,
        "flann_checks": 50,
        "match_ratio": 0.8,
        "max_angle": 5,
        "fixed_scale": 1.17
    }
}


class WindowManager:
    """窗口管理类，负责窗口查找和操作"""

    @staticmethod
    def find_target_window(config: dict) -> int:
        """根据配置查找目标窗口句柄"""

        def callback(hwnd, hwnd_list):
            current_title = win32gui.GetWindowText(hwnd)
            current_class = win32gui.GetClassName(hwnd)
            _, pid = win32process.GetWindowThreadProcessId(hwnd)

            if (config["title_part"] in current_title and
                    current_class == config["window_class"] and
                    WindowManager._check_process_exe(pid, config["process_exe"])):
                hwnd_list.append(hwnd)
            return True

        hwnd_list = []
        win32gui.EnumWindows(callback, hwnd_list)
        return hwnd_list[0] if hwnd_list else None

    @staticmethod
    def _check_process_exe(pid: int, target_exe: str) -> bool:
        """验证进程可执行文件"""
        try:
            process = psutil.Process(pid)
            exe_name = process.exe().split('\\')[-1].lower()
            return exe_name == target_exe.lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False

    @staticmethod
    def activate_window(hwnd: int):
        """激活并置顶窗口"""
        try:
            pythoncom.CoInitialize()

            if win32gui.IsIconic(hwnd):
                win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)

            win32api.keybd_event(win32con.VK_MENU, 0, 0, 0)
            win32gui.SetForegroundWindow(hwnd)
            win32api.keybd_event(win32con.VK_MENU, 0, win32con.KEYEVENTF_KEYUP, 0)
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"窗口激活失败: {str(e)}")
        finally:
            pythoncom.CoUninitialize()


class ImageProcessor:
    """图像处理类，负责图像采集和处理"""

    def __init__(self, hwnd: int, config: dict):
        self.hwnd = hwnd
        self.capture_config = config
        self.window_rect = win32gui.GetWindowRect(hwnd)
        windll.user32.SetProcessDPIAware()

    def capture_window(self) -> np.ndarray:
        """捕获窗口指定区域"""
        region = {
            "left": self.window_rect[0] + self.capture_config["x_offset"],
            "top": self.window_rect[1] + self.capture_config["y_offset"],
            "width": self.capture_config["width"],
            "height": self.capture_config["height"]
        }

        with mss.mss() as sct:
            screenshot = sct.grab(region)
            pil_image = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")

        return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    @staticmethod
    def crop_to_circle(image: np.ndarray, diameter: int) -> np.ndarray:
        """将图像裁剪为圆形"""
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)
        radius = (diameter - 18) // 2

        cv2.circle(mask, center, radius, 255, -1)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        result = np.zeros((height, width, 4), dtype=np.uint8)
        result[:, :, :3] = masked_image
        result[:, :, 3] = mask
        return result


class FeatureMatcher:
    """特征匹配类，负责特征匹配和结果可视化"""

    def __init__(self, template_path: str, config: dict):
        self.config = config
        self.kp_template, self.des_template = self._load_template_features(template_path)
        self.last_match_center = None
        self.sift = cv2.SIFT_create(nOctaveLayers=5, contrastThreshold=0.01, edgeThreshold=15)
        self.flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, table_number=6, key_size=12),
            dict(checks=config["flann_checks"])
        )

    def _load_template_features(self, filename: str) -> tuple:
        """加载模板特征"""
        try:
            data = np.load(filename)
            kp_data = data['kp']
            des = data['des']

            kp = []
            for p in kp_data:
                keypoint = cv2.KeyPoint(
                    x=float(p[0]),  # pt_x
                    y=float(p[1]),  # pt_y
                    size=float(p[2]),  # size
                    angle=float(p[3]),  # angle
                    response=float(p[4]),  # response
                    octave=int(p[5]),  # octave需要显式转int
                    class_id=int(p[6])  # 兼容class_id参数
                )
                kp.append(keypoint)

            return kp, des
        except FileNotFoundError:
            logger.error("特征文件未找到，请先运行特征提取脚本")
            raise
    def process_frame(self, query_image: np.ndarray):
        """处理单个帧并进行特征匹配"""

        query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGRA2GRAY)
        query_color = cv2.cvtColor(query_image, cv2.COLOR_BGRA2BGR)

        # 获取当前匹配区域特征点
        kp_template, des_template = self._get_current_features()

        # 特征检测和匹配
        kp_query, des_query = self.sift.detectAndCompute(query_gray, None)
        matches = self.flann.knnMatch(des_query, des_template, k=2)
        good_matches = [m for m, n in matches if m.distance < self.config["match_ratio"] * n.distance]

        if len(good_matches) >= 3:
            return self._handle_successful_match(kp_query, kp_template, good_matches, query_color)
        return None

    def _get_current_features(self) -> tuple:
        """获取当前使用的特征点"""
        if self.last_match_center:
            return self._filter_features_by_region()
        return self.kp_template, self.des_template

    def _filter_features_by_region(self) -> tuple:
        """根据上次匹配位置筛选特征点"""
        cx, cy = self.last_match_center
        filtered = [
            (kp, des) for kp, des in zip(self.kp_template, self.des_template)
            if (kp.pt[0] - cx) ** 2 + (kp.pt[1] - cy) ** 2 <= 1000 ** 2
        ]

        if filtered:
            # 解压筛选后的特征点
            kp_filtered, des_filtered = zip(*filtered)
            # 将描述子转换为二维numpy数组
            des_array = np.vstack(des_filtered)
            return kp_filtered, des_array
        return self.kp_template, self.des_template

    def _filter_features_by_region1(self) -> tuple:
        """根据上次匹配位置筛选特征点"""
        cx, cy = self.last_match_center
        filtered = [
            (kp, des) for kp, des in zip(self.kp_template, self.des_template)
            if (kp.pt[0] - cx) ** 2 + (kp.pt[1] - cy) ** 2 <= 1000 ** 2
        ]
        return zip(*filtered) if filtered else (self.kp_template, self.des_template)

    def _handle_successful_match(self, kp_query, kp_template, matches, query_color):
        """处理成功匹配的情况"""
        src_pts = np.float32([kp_query[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp_template[m.trainIdx].pt for m in matches])

        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None:
            return None

        theta = np.arctan2(M[1, 0], M[0, 0])
        if abs(np.rad2deg(theta)) > self.config["max_angle"]:
            logger.debug("角度超出允许范围")
            return None

        tx, ty = M[0, 2], M[1, 2]
        center = self._calculate_center(M, query_color.shape)
        self._update_match_center(len(matches), center)

        return {
            "transform": M,
            "center": center,
            "matches": len(matches),
            "angle": np.rad2deg(theta),
            "translation": (tx, ty)
        }

    def _calculate_center(self, M, shape):
        """计算匹配中心点"""
        h, w = shape[:2]
        scaled_w = int(w * self.config["fixed_scale"])
        scaled_h = int(h * self.config["fixed_scale"])
        return (
            int(M[0, 2] + scaled_w / 2),
            int(M[1, 2] + scaled_h / 2)
        )

    def _update_match_center(self, matches_count, center):
        """更新匹配中心位置"""
        if matches_count >= self.config["min_matches"]:
            self.last_match_center = center
            logger.info(f"更新匹配中心到: {center}")
        else:
            self.last_match_center = None


class ResultVisualizer:
    """结果可视化类，负责结果显示和窗口管理"""

    @staticmethod
    def show_match_result(result: dict, template_image: np.ndarray):
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

        # 创建显示区域
        cropped = ResultVisualizer._create_viewport(display_img, center, 250)
        cv2.putText(cropped, f'Matches: {result["matches"]}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示结果
        cv2.namedWindow('Matching Result', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Matching Result', 500, 500)
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


def main():
    """主程序"""
    # 初始化模块
    target_hwnd = WindowManager.find_target_window(DEFAULT_CONFIG["window"])
    if not target_hwnd:
        logger.error("未找到目标窗口")
        return

    WindowManager.activate_window(target_hwnd)
    image_processor = ImageProcessor(target_hwnd, DEFAULT_CONFIG["capture"])
    feature_matcher = FeatureMatcher('preprocessed_features.npz', DEFAULT_CONFIG["matching"])

    # 加载模板图像
    template_img = cv2.imread('result100.png', cv2.IMREAD_GRAYSCALE)
    template_color = cv2.cvtColor(template_img, cv2.COLOR_GRAY2BGR)

    try:
        while True:
            # 处理帧
            frame = image_processor.capture_window()
            circular_frame = ImageProcessor.crop_to_circle(frame, DEFAULT_CONFIG["capture"]["diameter"])

            # 特征匹配
            match_result = feature_matcher.process_frame(circular_frame)

            # 显示结果
            if match_result:
                ResultVisualizer.show_match_result(match_result, template_color)

            if cv2.waitKey(1) == 27:  # ESC退出
                break

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()