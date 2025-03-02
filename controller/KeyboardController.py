from controller import *

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
            WindowManager.activate_window(self.game_hwnd)
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
        """按下并保持tm秒后释放"""
        self.press_down(key)
        time.sleep(tm)
        if keyup:
            self.press_up(key)

    def __del__(self):
        """确保释放资源"""
        if self.listener.is_alive():
            self.listener.stop()


#初始化输入处理器（默认前台模式）
input_handler = InputHandler(DEFAULT_CONFIG["window"], foreground=True)

#input_handler.press_down('w')
#input_handler.press_down(Key.space)
#time.sleep(5)
#input_handler.press_up('w')
#input_handler.press_up(Key.space)