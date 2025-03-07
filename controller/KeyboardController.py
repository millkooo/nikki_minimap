import ctypes
import time
from ctypes import wintypes
import win32con
import win32api
import win32gui
from Ui_Manage.WindowManager import WinControl
from pynput.keyboard import Controller, Listener, Key, KeyCode

class InputHandler:

    def __init__(self, config, foreground=True):
            self.config = config
            self.foreground = foreground
            self.game_hwnd = None
            self.stop_flag = False
            self.keyboard = Controller()
            # 初始化窗口查找
            self.refresh_window_handle()
            # 设置键盘监听
            self.listener = Listener(on_press=self._on_key_press)
            self.listener.start()


    def refresh_window_handle(self):
        """刷新游戏窗口句柄"""
        self.game_hwnd = WinControl.find_target_window(self.config)
        if not self.game_hwnd:
            raise RuntimeError("未找到游戏窗口")


    def _on_key_press(self, key):
        """按键监听回调"""
        try:
            if key == Key.f9:
                print("F9 已被按下，尝试停止运行")
                self.stop_flag = True
        except AttributeError:
            pass


    def _get_vk_code(self, key):
        """获取虚拟键码"""
        if isinstance(key, str) and len(key) == 1:
            # 处理字母按键
            return 0x41 + ord(key.lower()) - ord('a')
        elif isinstance(key, KeyCode):
            # 处理字符KeyCode对象
            return key.vk if key.vk is not None else 0
        elif isinstance(key, Key):
            # 处理特殊按键
            return key.value.vk
        else:
            raise ValueError(f"不支持的按键类型：{key}")


    def _ensure_foreground_window(self):
        """确保游戏窗口在前台"""
        while not self.stop_flag:
            if win32gui.GetForegroundWindow() == self.game_hwnd:
                return
            print("尝试将游戏窗口置于前台...")
            WinControl.activate_window(self.game_hwnd)
            time.sleep(1)


    def _send_foreground_key_down(self, key):
        """前台模式按下按键"""
        self._ensure_foreground_window()
        self.keyboard.press(key)


    def _send_foreground_key_up(self, key):
        """前台模式释放按键"""
        self._ensure_foreground_window()
        self.keyboard.release(key)


    def _send_background_key_down(self, key):
        """后台模式按下按键"""
        if self._is_window_minimized():
            win32gui.ShowWindow(self.game_hwnd, win32con.SW_RESTORE)
        vk_code = self._get_vk_code(key)
        win32api.PostMessage(self.game_hwnd, win32con.WM_KEYDOWN, vk_code, 0)


    def _send_background_key_up(self, key):
        """后台模式释放按键"""
        vk_code = self._get_vk_code(key)
        win32api.PostMessage(self.game_hwnd, win32con.WM_KEYUP, vk_code, 0)


    def _is_window_minimized(self):
        """检查窗口是否最小化"""
        placement = win32gui.GetWindowPlacement(self.game_hwnd)
        return placement[1] == win32con.SW_SHOWMINIMIZED


    def press_down(self, key):
        """按下按键（不释放）"""
        if self.stop_flag:
            return
        if self.foreground:
            self._send_foreground_key_down(key)
        else:
            self._send_background_key_down(key)

    def press_up(self, key):
        """释放按键"""
        if self.stop_flag:
            return
        if self.foreground:
            self._send_foreground_key_up(key)
        else:
            self._send_background_key_up(key)

    def press(self, key, tm=0.2, keyup=True):
        """使用 win32api 实现后台按键"""
        vk_code = self._get_vk_code(key)
        # 发送按下事件
        win32api.PostMessage(self.game_hwnd, win32con.WM_KEYDOWN, vk_code, 0)
        time.sleep(tm)
        if keyup:
            # 发送释放事件
            win32api.PostMessage(self.game_hwnd, win32con.WM_KEYUP, vk_code, 0)

    def __del__(self):
        """确保释放资源"""
        if self.listener.is_alive():
            self.listener.stop()



_instance = None

def get_input_handler(config=None, foreground=True):
    global _instance
    if _instance is None:
        if config is None:
            from main import DEFAULT_CONFIG  # 确保从配置模块导入
            config = DEFAULT_CONFIG["window"]
        _instance = InputHandler(config, foreground=foreground)
    return _instance


# input_handler = get_input_handler()
# time.sleep(3)
# input_handler.press_down('w')
# time.sleep(25)
# input_handler.press_up('w')
