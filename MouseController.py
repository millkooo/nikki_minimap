import ctypes
import time
from ctypes import wintypes
import win32api
import win32gui

from main import WindowManager, DEFAULT_CONFIG

if ctypes.sizeof(ctypes.c_void_p) == 4:
    ULONG_PTR = ctypes.c_ulong  # 32-bit
else:
    ULONG_PTR = ctypes.c_ulonglong  # 64-bit
# 定义SendInput需要的数据结构
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ('dx', wintypes.LONG),
        ('dy', wintypes.LONG),
        ('mouseData', wintypes.DWORD),
        ('dwFlags', wintypes.DWORD),
        ('time', wintypes.DWORD),
        ('dwExtraInfo', ULONG_PTR),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ('type', wintypes.DWORD),
        ('mi', MOUSEINPUT),
    ]


def _set_dpi_awareness():
    """设置DPI感知以获取正确的物理分辨率"""
    try:
        # 对于Windows 8.1及以上版本
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except:
        # 回退到旧版Windows API
        ctypes.windll.user32.SetProcessDPIAware()


class MouseController:
    """鼠标控制类"""
    INPUT_MOUSE = 0
    MOUSEEVENTF_MOVE = 0x0001
    MOUSEEVENTF_ABSOLUTE = 0x8000
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    MOUSEEVENTF_RIGHTDOWN = 0x0008
    MOUSEEVENTF_RIGHTUP = 0x0010
    MOUSEEVENTF_WHEEL = 0x0800
    WHEEL_DELTA = 120
    def __init__(self, window_manager: WindowManager, config: dict = DEFAULT_CONFIG):
        """
        初始化鼠标控制器
        :param window_manager: 窗口管理器实例
        :param config: 窗口配置信息
        """
        self.window_manager = window_manager
        self.config = config
        self.hwnd = None
        self._init_sendinput()
        self._set_dpi_awareness()

    def _set_dpi_awareness(self):
        """设置DPI感知以获取正确的物理分辨率"""
        try:
            # 对于Windows 8.1及以上版本
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except:
            # 回退到旧版Windows API
            ctypes.windll.user32.SetProcessDPIAware()


    def _init_sendinput(self):
        """初始化SendInput函数（修正参数类型）"""
        self.SendInput = ctypes.windll.user32.SendInput
        self.SendInput.argtypes = [
            ctypes.c_uint,
            ctypes.POINTER(INPUT),
            ctypes.c_int
        ]
        self.SendInput.restype = ctypes.c_uint

    def _get_hwnd(self) -> int:
        """获取并缓存窗口句柄"""
        if not self.hwnd:
            self.hwnd = self.window_manager.find_target_window(self.config["window"])
            if not self.hwnd:
                raise ValueError("目标窗口未找到")
        return self.hwnd

    def _activate_window(self):
        """激活目标窗口"""
        hwnd = self._get_hwnd()
        self.window_manager.activate_window(hwnd)

    def _to_absolute_coordinates(self, x: int, y: int) -> tuple:
        """
        将窗口坐标转换为绝对坐标(0-65535)
        :return: (abs_x, abs_y)
        """
        hwnd = self._get_hwnd()
        client_rect = win32gui.GetClientRect(hwnd)
        left, top = win32gui.ClientToScreen(hwnd, (client_rect[0], client_rect[1]))

        # 获取物理分辨率
        screen_width = win32api.GetSystemMetrics(0)
        screen_height = win32api.GetSystemMetrics(1)

        abs_x = int((left + x) / screen_width * 65535)
        abs_y = int((top + y) / screen_height * 65535)
        return abs_x, abs_y

    def move_absolute(self, x: int, y: int):
        """
        绝对坐标移动（适用于UI操作）
        :param x: 窗口客户区X坐标
        :param y: 窗口客户区Y坐标
        """
        self._activate_window()
        abs_x, abs_y = self._to_absolute_coordinates(x, y)

        # 构造输入事件
        mi = MOUSEINPUT(abs_x, abs_y, 0, self.MOUSEEVENTF_ABSOLUTE | self.MOUSEEVENTF_MOVE, 0, 0)
        input_struct = INPUT(self.INPUT_MOUSE, mi)

        # 发送非阻塞输入
        self.SendInput(1, ctypes.byref(input_struct), ctypes.sizeof(INPUT))

    def move_relative(self, dx: int, dy: int):
        """
        相对坐标移动（适用于视角控制）
        :param dx: X轴偏移量
        :param dy: Y轴偏移量
        """
        self._activate_window()

        # 构造输入事件
        mi = MOUSEINPUT(dx, dy, 0, self.MOUSEEVENTF_MOVE, 0, 0)
        input_struct = INPUT(self.INPUT_MOUSE, mi)

        # 发送输入
        self.SendInput(1, ctypes.byref(input_struct), ctypes.sizeof(INPUT))

    @property
    def current_position(self) -> tuple:
        """获取当前鼠标在窗口客户区中的坐标"""
        hwnd = self._get_hwnd()
        screen_x, screen_y = win32api.GetCursorPos()
        window_left, window_top = win32gui.ClientToScreen(hwnd, (0, 0))
        return (screen_x - window_left, screen_y - window_top)

    def press_left(self):
        """按下左键"""
        self._activate_window()
        mi = MOUSEINPUT(0, 0, 0, self.MOUSEEVENTF_LEFTDOWN, 0, 0)
        input_struct = INPUT(self.INPUT_MOUSE, mi)
        self.SendInput(1, ctypes.byref(input_struct), ctypes.sizeof(INPUT))

    def release_left(self):
        """释放左键"""
        self._activate_window()
        mi = MOUSEINPUT(0, 0, 0, self.MOUSEEVENTF_LEFTUP, 0, 0)
        input_struct = INPUT(self.INPUT_MOUSE, mi)
        self.SendInput(1, ctypes.byref(input_struct), ctypes.sizeof(INPUT))

    def click_left(self, duration: float = 0.05):
        """
        单击左键（含按住时间）
        :param duration: 按键保持时间（秒）
        """
        self.press_left()
        time.sleep(duration)
        self.release_left()

    def press_right(self):
        """按下右键"""
        self._activate_window()
        mi = MOUSEINPUT(0, 0, 0, self.MOUSEEVENTF_RIGHTDOWN, 0, 0)
        input_struct = INPUT(self.INPUT_MOUSE, mi)
        self.SendInput(1, ctypes.byref(input_struct), ctypes.sizeof(INPUT))

    def release_right(self):
        """释放右键"""
        self._activate_window()
        mi = MOUSEINPUT(0, 0, 0, self.MOUSEEVENTF_RIGHTUP, 0, 0)
        input_struct = INPUT(self.INPUT_MOUSE, mi)
        self.SendInput(1, ctypes.byref(input_struct), ctypes.sizeof(INPUT))

    def click_right(self, duration: float = 0.1):
        """
        单击右键（含按住时间）
        :param duration: 按键保持时间（秒）
        """
        self.press_right()
        time.sleep(duration)
        self.release_right()

    def wheel(self, delta: int):
        """
        滚动鼠标滚轮
        :param delta: 滚动量（正数向上，负数向下）
        """
        self._activate_window()
        mi = MOUSEINPUT(0, 0, delta * self.WHEEL_DELTA, self.MOUSEEVENTF_WHEEL, 0, 0)
        input_struct = INPUT(self.INPUT_MOUSE, mi)
        self.SendInput(1, ctypes.byref(input_struct), ctypes.sizeof(INPUT))

# 初始化窗口管理器
window_manager = WindowManager()

# 创建鼠标控制器
mouse = MouseController(window_manager)

#mouse.wheel(1)
#mouse.click_right()
#mouse.click_left()

# 绝对移动（UI操作）
#mouse.move_absolute(100, 100)  # 移动到窗口客户区(100,200)位置

# 相对移动（视角控制）
#mouse.move_relative(100, 30)   # 向右移动100像素，向上移动30像素