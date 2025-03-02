import ctypes
import logging
import time
import cv2
import numpy as np
import psutil
import pythoncom
import win32con
import win32gui
import win32print
import win32process

from capture.img_processor import ImageProcessor
from minimap_match.FeatureMatch import FeatureMatcher

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
        #截图的参数（小地图）
        "x_offset": 77,
        "y_offset": 38,
        "width": 204,
        "height": 204,
        "diameter": 204
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
            current_title = win32gui.GetWindowText(hwnd).strip()
            current_class = win32gui.GetClassName(hwnd).strip()
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
            win32gui.SetForegroundWindow(hwnd)
            time.sleep(0.5)
        except Exception as e:
            logger.error(f"窗口激活失败: {str(e)}")
        finally:
            pythoncom.CoUninitialize()

    @staticmethod
    def close_window(hwnd: int) -> None:
        """关闭指定窗口句柄对应的窗口"""
        try:
            import win32gui
            import win32con
            win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
            print("游戏窗口已关闭")
        except Exception as e:
            print(f"关闭窗口失败: {e}")

    @staticmethod
    def is_window_minimized(hwnd: int) -> bool:
        """检测窗口是否处于最小化状态"""
        try:
            import win32gui
            import win32con
            placement = win32gui.GetWindowPlacement(hwnd)
            return placement[1] == win32con.SW_SHOWMINIMIZED
        except:
            return False

    @staticmethod
    def get_scaling_factor(hwnd: int) -> float:
        """获取精确到小数点后两位的缩放因子"""
        try:
            dpi = ctypes.windll.user32.GetDpiForWindow(hwnd)
            return round(dpi / 96.0, 2)
        except AttributeError:
            hdc = win32gui.GetDC(hwnd)
            dpi = win32print.GetDeviceCaps(hdc, win32con.LOGPIXELSX)
            win32gui.ReleaseDC(hwnd, hdc)
            return round(dpi / 96.0, 2)


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
    # 初始化
    target_hwnd = WindowManager.find_target_window(DEFAULT_CONFIG["window"])
    if not target_hwnd:
        logger.error("未找到目标窗口")
        return

    WindowManager.activate_window(target_hwnd)
    image_processor = ImageProcessor(target_hwnd, DEFAULT_CONFIG["capture"])
    feature_matcher = FeatureMatcher('resources/preprocessed_features.npz', DEFAULT_CONFIG["matching"])

    # 加载模板图
    template_img = cv2.imread('resources/nuanuan_map.png', cv2.IMREAD_GRAYSCALE)
    template_color = cv2.cvtColor(template_img, cv2.COLOR_GRAY2BGR)

    prev_hash = None
    prev_result = None
    try:
        while True:
            # 处理帧
            frame = image_processor.capture_window()
            circular_frame = ImageProcessor.circle_crop(frame, DEFAULT_CONFIG["capture"]["diameter"])

            # 特征匹配
            current_hash = ImageProcessor.compute_image_hash(circular_frame)
            if current_hash == prev_hash:
                match_result = prev_result
                #print("检测到相同帧，复用先前结果")
            else:
                match_result = feature_matcher.process_frame(circular_frame)
                prev_hash = current_hash
                prev_result = match_result
                #print("检测到新帧，执行特征匹配")

            # 显示结果
            if match_result:
                ResultVisualizer.show_match_result(match_result, template_color)

            if cv2.waitKey(1) == 27:  # ESC退出
                break

    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()