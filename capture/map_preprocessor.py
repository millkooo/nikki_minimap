# map_preprocessor.py
import os
import json
import cv2
import numpy as np
from tqdm import tqdm


class MapPreprocessor:
    def __init__(self, config=None):
        self.config = config or {
            'input_dir': 'resources/map_raw',
            'output_file': 'resources/map_edges.json',
            'canny_threshold': (100, 200),
            'approx_epsilon': 2.0,
            'neighbor_threshold': 15.0,
            'simplify_points': True,
            'max_points': 200
        }

        # 存储处理结果的字典
        self.map_data = {}

        # 创建输出目录
        os.makedirs(os.path.dirname(self.config['output_file']), exist_ok=True)

    def process_all_maps(self):
        """处理所有地图文件"""
        file_list = [f for f in os.listdir(self.config['input_dir'])
                     if f.endswith('.png') and 'panduan' not in f]

        # 第一阶段：处理单个地图
        for filename in tqdm(file_list, desc="Processing maps"):
            zone_name = filename.split('.')[0]
            img_path = os.path.join(self.config['input_dir'], filename)
            self._process_single_map(zone_name, img_path)

        # 第二阶段：建立相邻关系
        self._detect_neighbors()

        # 保存结果
        with open(self.config['output_file'], 'w') as f:
            json.dump(self.map_data, f, indent=2)

        print(f"预处理完成，结果已保存到 {self.config['output_file']}")

    def _process_single_map(self, zone_name, img_path):
        """处理单个地图文件"""
        # 读取并预处理图像
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        edges = cv2.Canny(img, *self.config['canny_threshold'])

        # 查找轮廓
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # 合并所有轮廓点
        all_points = np.vstack([c.squeeze() for c in contours if len(c) >= 3])

        # 简化点集
        if self.config['simplify_points']:
            epsilon = self.config['approx_epsilon']
            while True:
                approx = cv2.approxPolyDP(
                    np.array(all_points),
                    epsilon,
                    closed=True
                ).squeeze()
                if len(approx) <= self.config['max_points'] or epsilon >= 10:
                    break
                epsilon *= 1.2

        # 格式化为坐标列表
        points = [(int(x), int(y)) for x, y in approx.tolist()]

        # 保存到数据结构
        self.map_data[zone_name] = {
            'original_size': img.shape[:2],
            'contour_points': points,
            'neighbors': {}
        }

    def _detect_neighbors(self):
        """检测相邻区域"""
        zone_names = list(self.map_data.keys())

        # 建立空间索引加速搜索
        spatial_index = {}
        for name in zone_names:
            points = np.array(self.map_data[name]['contour_points'])
            spatial_index[name] = {
                'min_x': np.min(points[:, 0]),
                'max_x': np.max(points[:, 0]),
                'min_y': np.min(points[:, 1]),
                'max_y': np.max(points[:, 1]),
                'points': points
            }

        # 比较所有区域对
        for i, name1 in enumerate(zone_names):
            data1 = spatial_index[name1]
            for j in range(i + 1, len(zone_names)):
                name2 = zone_names[j]
                data2 = spatial_index[name2]

                # 快速排除不相交区域
                if (data1['max_x'] < data2['min_x'] - self.config['neighbor_threshold'] or
                        data2['max_x'] < data1['min_x'] - self.config['neighbor_threshold'] or
                        data1['max_y'] < data2['min_y'] - self.config['neighbor_threshold'] or
                        data2['max_y'] < data1['min_y'] - self.config['neighbor_threshold']):
                    continue

                # 精确检测相邻点
                border_points = self._find_border_points(
                    data1['points'],
                    data2['points']
                )

                if border_points:
                    # 记录双向关系
                    self._record_neighbor(name1, name2, border_points)
                    self._record_neighbor(name2, name1, border_points)

    def _find_border_points(self, points1, points2):
        """查找相邻边界点"""
        border = []
        threshold = self.config['neighbor_threshold']

        # 建立KDTree加速搜索
        tree1 = cv2.ml.KNearest_create()
        tree1.train(points1.astype(np.float32), cv2.ml.ROW_SAMPLE)

        for pt in points2:
            _, result, _, _ = tree1.findNearest(np.array([pt], dtype=np.float32), 1)
            distance = np.linalg.norm(pt - result[0][0])
            if distance < threshold:
                border.append({
                    'self': pt.tolist(),
                    'neighbor': result[0][0].tolist(),
                    'distance': float(distance)
                })

        return border if len(border) > 5 else None  # 最小有效点数

    def _record_neighbor(self, src, dst, points):
        """记录相邻关系"""
        # 计算连接边界的大致方向
        dx = np.mean([p['self'][0] - p['neighbor'][0] for p in points])
        dy = np.mean([p['self'][1] - p['neighbor'][1] for p in points])

        # 简化为主要方向
        direction = []
        if abs(dx) > abs(dy):
            direction.append('east' if dx > 0 else 'west')
        else:
            direction.append('south' if dy > 0 else 'north')

        # 保存相邻信息
        self.map_data[src]['neighbors'][dst] = {
            'direction': direction,
            'border_points': points
        }


if __name__ == "__main__":
    # 使用示例
    processor = MapPreprocessor({
        'input_dir': '../resources/img/xinyuanyuanye',  # 原始地图存放目录
        'output_file': '../resources/map_edges.json',  # 输出文件路径
        'canny_threshold': (80, 160),  # Canny边缘检测阈值
        'approx_epsilon': 1.5,  # 轮廓简化精度
        'neighbor_threshold': 12.0,  # 相邻判定阈值(像素)
        'max_points': 150  # 最大保留点数
    })

    processor.process_all_maps()