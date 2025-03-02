from capture import *

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
    def circle_crop(image: np.ndarray, diameter: int) -> np.ndarray:
        """将图像裁剪为圆形"""
        height, width = image.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (width // 2, height // 2)
        radius = diameter // 2

        cv2.circle(mask, center, radius, 255, -1)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        result = np.zeros((height, width, 4), dtype=np.uint8)
        #result[:, :, :3] = masked_image
        #result[:, :, 3] = mask
        return masked_image

    @staticmethod
    def compute_image_hash(image: np.ndarray) -> str:
        """计算图像的快速哈希值（基于aHash算法）"""
        # 转换为灰度图并缩小尺寸
        resized = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (8, 8), interpolation=cv2.INTER_AREA)
        # 计算像素平均值
        avg = resized.mean()
        # 生成二进制哈希字符串
        return ''.join(['1' if pixel > avg else '0' for row in resized for pixel in row])
