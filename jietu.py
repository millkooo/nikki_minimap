import time

import cv2
from main import ImageProcessor, DEFAULT_CONFIG, logger
from Ui_Manage.WindowManager import WinControl

# ---------------------- 主程序 ----------------------
if __name__ == "__main__":

    target_hwnd = WinControl.find_target_window(DEFAULT_CONFIG["window"])
    if not target_hwnd:
        logger.error("未找到目标窗口")

    WinControl.activate_window(target_hwnd)
    image_processor = ImageProcessor(target_hwnd)
    time.sleep(1)
    frame = image_processor.capture_region({"x_offset": 0, "y_offset": 0, "width": 281, "height": 242})
    cv2.imwrite('jietu-1.jpg', frame[:, :, :3])
    frame = image_processor._crop_image(frame, {"x": 77, "y": 38, "w": 204, "h": 204})

    #frame = image_processor.capture_region({ "x_offset": 77,"y_offset": 38,"width": 204,"height": 204})
    circular_frame = ImageProcessor.circle_crop(frame, DEFAULT_CONFIG["capture"]["diameter"])
    cv2.imwrite('jietu-2.jpg', circular_frame[:, :, :3])

