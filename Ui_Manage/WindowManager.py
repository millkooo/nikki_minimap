import ctypes
import logging
import psutil
import pythoncom
import win32con
import win32gui
import win32print
import win32process

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WinControl:
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
                    WinControl._check_process_exe(pid, config["process_exe"])):
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
        """获取dpi"""
        try:
            dpi = ctypes.windll.user32.GetDpiForWindow(hwnd)
            return round(dpi / 96.0, 2)
        except AttributeError:
            hdc = win32gui.GetDC(hwnd)
            dpi = win32print.GetDeviceCaps(hdc, win32con.LOGPIXELSX)
            win32gui.ReleaseDC(hwnd, hdc)
            return round(dpi / 96.0, 2)
