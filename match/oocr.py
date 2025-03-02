import cv2
import time
from paddleocr import PaddleOCR


class RealTimePaddleOCR:
    def __init__(self):
        """
        初始化OCR识别器
        """
        self.ocr = PaddleOCR(use_angle_cls=False)
        self.region = None  # 存储识别区域 (x1, y1, x2, y2)

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
        单次识别模式
        :return: list of (coordinates, text)
                 coordinates格式：[[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        """
        processed_img, offset = self._preprocess_image(img)
        result = self.ocr.ocr(processed_img, cls=False)
        return self._convert_coordinates(result[0], offset) if result else []

    def continuous_recognize(self, frame_generator, target_text=None, timeout=30):
        """
        连续识别模式 (约30fps)
        :param frame_generator: 帧生成器（需支持迭代）
        :param target_text: 目标识别文本（None时为持续识别）
        :param timeout: 超时时间（秒）
        :return: 生成器持续输出识别结果，找到目标时终止
        """
        start_time = time.time()
        frame_count = 0

        for frame in frame_generator:
            # 计算帧率控制
            frame_count += 1
            elapsed = time.time() - start_time
            expected_time = frame_count / 30

            # 处理帧
            result = self.recognize(frame)

            # 检查超时
            if elapsed > timeout:
                print(f"Timeout reached: {timeout}s")
                break

            # 检查目标文本
            if target_text is not None:
                if any(target_text in res[1] for res in result):
                    print(f"Target text '{target_text}' found")
                    yield result
                    break

            # 维持30fps
            if elapsed < expected_time:
                time.sleep(expected_time - elapsed)

            yield result


# #######################
# 使用示例
# #######################




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

    # 示例2：摄像头连续识别
    ocr_processor.set_region(None)  # 识别全屏
    print("Starting continuous recognition...")

    for results in ocr_processor.continuous_recognize(
            frame_generator=camera_generator(),
            target_text="EXIT",  # 寻找包含"EXIT"的文字
            timeout=60
    ):
        # 实时显示结果
        for box, text in results:
            print(f"Found: {text}")

        # 在此处添加图像显示逻辑（可选）
        # cv2.imshow('Live OCR', frame)
        # if cv2.waitKey(1) == 27:  # ESC退出
        #     break