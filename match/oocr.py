import threading

import cv2
import time
from paddleocr import PaddleOCR


class RealTimePaddleOCR:
    def __init__(self):
        self.ocr = PaddleOCR(use_angle_cls=False)
        self.region = None
        self._stop_event = threading.Event()
        self._recognition_thread = None
        self._callback = None

    def set_region(self, region):
        """
        设置识别区域
        :param region: (x1, y1, x2, y2) 格式的元组/列表，设为None时识别全图
        """
        self.region = region

    def _preprocess_image(self, img):
        """
        图像预处理：颜色空间转换 + 区域裁剪
        :return: 处理后的图像，区域坐标补偿量
        """
        # BGR转RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 区域裁剪处理
        offset_x, offset_y = 0, 0
        if self.region is not None:
            x1, y1, x2, y2 = self.region
            offset_x, offset_y = x1, y1
            img_rgb = img_rgb[y1:y2, x1:x2]

        return img_rgb, (offset_x, offset_y)

    def _convert_coordinates(self, result, offset):
        """
        转换坐标到原始图像坐标系
        :param result: OCR原始结果
        :param offset: 坐标补偿量 (x, y)
        """
        converted = []
        for line in result:
            box, text = line[0], line[1][0]
            # 坐标补偿
            adjusted_box = [[point[0] + offset[0], point[1] + offset[1]] for point in box]
            converted.append((adjusted_box, text))
        return converted

    def recognize(self, img):
        """
        :return: list of (coordinates, text)
                 coordinates格式：[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        processed_img, offset = self._preprocess_image(img)
        result = self.ocr.ocr(processed_img, cls=False)
        return self._convert_coordinates(result[0], offset) if result else []

    def start_continuous_recognition(self, image_processor, target_text,
                                     callback, timeout=30, interval=0.5,
                                     case_sensitive=False, contains=True):
        """
        非阻塞启动连续识别
        :param callback: 回调函数格式：def callback(status, result)
        """
        # 如果已有线程在运行则先停止
        if self._recognition_thread and self._recognition_thread.is_alive():
            self.stop_continuous_recognition()

        self._callback = callback
        self._stop_event.clear()

        args = (image_processor, target_text, timeout, interval, case_sensitive, contains)
        self._recognition_thread = threading.Thread(
            target=self._continuous_recognition_worker,
            args=args,
            daemon=True
        )
        self._recognition_thread.start()

    def stop_continuous_recognition(self):
        """主动停止识别"""
        self._stop_event.set()
        if self._recognition_thread and self._recognition_thread.is_alive():
            self._recognition_thread.join(timeout=2)

    def _continuous_recognition_worker(self, image_processor, target_text,
                                       timeout, interval, case_sensitive, contains):
        """工作线程执行体"""
        start_time = time.time()
        target = target_text if case_sensitive else target_text.lower()

        try:
            while not self._stop_event.is_set():
                # 捕获并处理图像帧
                frame = image_processor.capture_region()
                ocr_results = self.recognize(frame)

                # 遍历识别结果
                for (box, text) in ocr_results:
                    current_text = text if case_sensitive else text.lower()

                    if (contains and target in current_text) or \
                            (not contains and target == current_text):
                        self._trigger_callback('found', {
                            'text': text,
                            'coordinates': box,
                            'elapsed': time.time() - start_time
                        })
                        return

                # 超时判断
                if time.time() - start_time > timeout:
                    self._trigger_callback('timeout', {
                        'elapsed': time.time() - start_time,
                        'message': f'未在{timeout}秒内检测到目标文本'
                    })
                    return

                # 间隔等待（可中断）
                self._stop_event.wait(interval)
        except Exception as e:
            self._trigger_callback('error', {
                'exception': e,
                'message': '识别过程中发生异常'
            })

    def _trigger_callback(self, status, result):
        """安全触发回调"""
        if self._callback:
            try:
                self._callback(status, result)
            except Exception as e:
                print(f"回调函数执行出错: {str(e)}")



if __name__ == "__main__":
    # 初始化识别器
    ocr_processor = RealTimePaddleOCR()

    # 示例1：单次识别测试
    test_img = cv2.imread('test.jpg')
    ocr_processor.set_region((50, 100, 400, 300))  # 设置识别区域
    single_result = ocr_processor.recognize(test_img)
    print("Single Recognition Results:")
    for box, text in single_result:
        print(f"Text: {text}\nCoordinates: {box}\n")

    # 示例2：连续识别
    ocr_processor.set_region(None)  # 识别全屏
    print("Starting continuous recognition...")

    for results in ocr_processor.continuous_recognize(
            frame_generator=camera_generator(),
            target_text="歌声",  # 寻找包含"EXIT"的文字
            timeout=60
    ):
        # 实时显示结果
        for box, text in results:
            print(f"Found: {text}")

        # 在此处添加图像显示逻辑（可选）
        # cv2.imshow('Live OCR', frame)
        # if cv2.waitKey(1) == 27:  # ESC退出
        #     break