import time
from controller.MouseController import mouse

from pynput.keyboard import Key
from controller.KeyboardController import input_handler
time.sleep(2)
input_handler.press(Key.esc)
mouse.move_absolute(900,520)#美鸭梨
mouse.click_left()
mouse.move_absolute(1820,76,0)#一键收获
time.sleep(1)
mouse.click_left(0.5)
time.sleep(0.5)
mouse.move_absolute(1111,800)#再次挖掘
mouse.click_left()
input_handler.press(Key.esc)
time.sleep(0.5)
input_handler.press(Key.esc)
time.sleep(0.5)
input_handler.press("m")
time.sleep(0.5)
mouse.move_absolute(1700,170)#地图区域选择
time.sleep(0.5)
mouse.click_left()
mouse.move_absolute(1500,360)#花园镇
time.sleep(0.2)
mouse.click_left()
mouse.move_absolute(681,981)#心愿花街
time.sleep(0.5)
mouse.click_left()
mouse.move_absolute(1630,1000)#传送
mouse.click_left()
time.sleep(14)
input_handler.press("s",tm=1.2)
input_handler.press("a",tm=1)
input_handler.press("f")
time.sleep(2)
mouse.move_absolute(1400,800)
mouse.click_left()
time.sleep(0.5)
mouse.move_absolute(1400,1000)
mouse.click_left()
time.sleep(8)
input_handler.press("w",tm=1.5)
input_handler.press("f")
time.sleep(0.5)
mouse.move_absolute(1500,777)
mouse.click_left()
time.sleep(0.5)
mouse.move_absolute(500,200)
mouse.click_left()
time.sleep(0.5)
mouse.move_absolute(1130,475)
mouse.click_left()
time.sleep(0.5)
mouse.move_absolute(1100,700)
mouse.click_left()
time.sleep(0.5)
mouse.move_absolute(1520,944)
mouse.click_left()
time.sleep(0.5)
input_handler.press("f")
time.sleep(1)
input_handler.press("f")
time.sleep(1)
input_handler.press("f")
time.sleep(1)
input_handler.press(Key.esc)
time.sleep(1)
input_handler.press(Key.esc)
time.sleep(0.5)
mouse.move_absolute(1230,1024)
mouse.click_left()
time.sleep(0.5)
mouse.move_absolute(1110,686)
mouse.click_left()
#美鸭梨挖掘 900，520
""""
美鸭梨挖掘-900，520
       一键收获 1800,760
       再次挖掘 1111 800
邮件-960，1028
邮件领取全部-550，1010
地图切换-1700，170
            1500,360
            674,976
            1630,1000
每日（
    550，600
    770，300
    1100，400
    1500，700
    1700，400
    ）



"""