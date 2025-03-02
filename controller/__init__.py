import ctypes
import time
from ctypes import wintypes
import win32con
import win32api
import win32gui
from main import WindowManager, DEFAULT_CONFIG
from pynput.keyboard import Controller, Listener, Key, KeyCode
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
