import win32gui
import time
import win32api
import win32con
import win32gui
from pynput.keyboard import Controller, Listener, Key

from main import WindowManager, DEFAULT_CONFIG


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
        self.game_hwnd = WindowManager.find_target_window(self.config)
        if not self.game_hwnd:
            raise RuntimeError("未找到游戏窗口")

    def _on_key_press(self, key):
        """按键监听回调"""
        try:
            if key == Key.f8:
                print("F8 已被按下，尝试停止运行")
                self.stop_flag = True
        except AttributeError:
            pass

    def _send_background_key(self, key, tm, keyup):
        """后台模式发送按键"""
        if self._is_window_minimized():
            win32gui.ShowWindow(self.game_hwnd, win32con.SW_RESTORE)

        vk_code = 0x41 + ord(key.lower()) - ord('a')

        win32api.PostMessage(self.game_hwnd, win32con.WM_KEYDOWN, vk_code, 0)
        time.sleep(tm)
        if keyup:
            win32api.PostMessage(self.game_hwnd, win32con.WM_KEYUP, vk_code, 0)

    def _send_foreground_key(self, key, tm, keyup):
        """前台模式发送按键"""
        # 确保窗口在前台
        while not self.stop_flag:
            if win32gui.GetForegroundWindow() == self.game_hwnd:
                break
            print("尝试将游戏窗口置于前台...")
            self.refresh_window_handle()
            time.sleep(1)

        # 使用pynput发送按键
        self.keyboard.press(key)
        time.sleep(tm)
        if keyup:
            self.keyboard.release(key)

    def _is_window_minimized(self):
        """检查窗口是否最小化"""
        placement = win32gui.GetWindowPlacement(self.game_hwnd)
        return placement[1] == win32con.SW_SHOWMINIMIZED

    def press(self, key, tm=0.2, keyup=True):
        """统一按键接口"""
        if self.stop_flag:
            return

        if self.foreground:
            self._send_foreground_key(key, tm, keyup)
        else:
            self._send_background_key(key, tm, keyup)

    def __del__(self):
        """析构函数确保释放资源"""
        if self.listener.is_alive():
            self.listener.stop()


# 初始化输入处理器（默认前台模式）
input_handler = InputHandler(DEFAULT_CONFIG["window"], foreground=True)

# 发送按键（自动根据模式处理）
input_handler.press('w',tm=2)  # 按下w键2秒
input_handler.press(Key.space, tm=0.5)  # 按下空格0.5秒
