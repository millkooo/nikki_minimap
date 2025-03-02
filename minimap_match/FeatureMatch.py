import cv2
import numpy as np
import logging
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureMatcher:
    """特征匹配类，负责特征匹配和结果可视化"""

    def __init__(self, template_path: str, config: dict):
        self.config = config
        self.kp_template, self.des_template = self._load_template_features(template_path)
        self.last_match_center = None
        self.sift = cv2.SIFT_create(nOctaveLayers=5, contrastThreshold=0.01, edgeThreshold=15)
        self.flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, table_number=6, key_size=12),
            dict(checks=config["flann_checks"])
        )

    def _load_template_features(self, filename: str) -> tuple:
        """加载模板特征"""
        try:
            data = np.load(filename)
            kp_data = data['kp']
            des = data['des']

            kp = []
            for p in kp_data:
                keypoint = cv2.KeyPoint(
                    x=float(p[0]),  # pt_x
                    y=float(p[1]),  # pt_y
                    size=float(p[2]),  # size
                    angle=float(p[3]),  # angle
                    response=float(p[4]),  # response
                    octave=int(p[5]),
                    class_id=int(p[6])
                )
                kp.append(keypoint)

            return kp, des
        except FileNotFoundError:
            logger.error("特征文件未找到，请先运行特征提取脚本")
            raise
    def process_frame(self, query_image: np.ndarray):
        """处理单个帧并进行特征匹配"""
        query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGRA2GRAY)
        query_color = cv2.cvtColor(query_image, cv2.COLOR_BGRA2BGR)

        # 获取当前匹配区域特征点
        kp_template, des_template = self._get_current_features()

        # 特征检测和匹配
        kp_query, des_query = self.sift.detectAndCompute(query_gray, None)
        matches = self.flann.knnMatch(des_query, des_template, k=2)
        good_matches = [m for m, n in matches if m.distance < self.config["match_ratio"] * n.distance]

        if len(good_matches) >= 3:
            return self._handle_successful_match(kp_query, kp_template, good_matches, query_color)
        return None

    def _get_current_features(self) -> tuple:
        """获取当前使用的特征点"""
        if self.last_match_center:
            return self._filter_features_by_region()
        return self.kp_template, self.des_template

    def _filter_features_by_region(self) -> tuple:
        """根据上次匹配位置筛选特征点"""
        cx, cy = self.last_match_center
        filtered = [
            (kp, des) for kp, des in zip(self.kp_template, self.des_template)
            if (kp.pt[0] - cx) ** 2 + (kp.pt[1] - cy) ** 2 <= 1000 ** 2
        ]

        if filtered:
            # 解压筛选后的特征点
            kp_filtered, des_filtered = zip(*filtered)
            # 将描述子转换为二维numpy数组
            des_array = np.vstack(des_filtered)
            return kp_filtered, des_array
        return self.kp_template, self.des_template

    def _handle_successful_match(self, kp_query, kp_template, matches, query_color):
        """处理成功匹配的情况"""
        src_pts = np.float32([kp_query[m.queryIdx].pt for m in matches])
        dst_pts = np.float32([kp_template[m.trainIdx].pt for m in matches])

        M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None:
            return None

        theta = np.arctan2(M[1, 0], M[0, 0])
        if abs(np.rad2deg(theta)) > self.config["max_angle"]:
            logger.debug("角度超出允许范围")
            return None

        tx, ty = M[0, 2], M[1, 2]
        center = self._calculate_center(M, query_color.shape)
        self._update_match_center(len(matches), center)
        return {
            "transform": M,
            "center": center,
            "matches": len(matches),
            "angle": np.rad2deg(theta),
            "translation": (tx, ty)
        }

    def _calculate_center(self, M, shape):
        """计算匹配中心点"""
        h, w = shape[:2]
        scaled_w = int(w * self.config["fixed_scale"])
        scaled_h = int(h * self.config["fixed_scale"])
        return (
            int(M[0, 2] + scaled_w / 2),
            int(M[1, 2] + scaled_h / 2)
        )

    def _update_match_center(self, matches_count, center):
        """更新匹配中心位置"""
        if matches_count >= self.config["min_matches"]:
            self.last_match_center = center
            logger.info(f"更新匹配中心到: {center}")
        else:
            logger.info("NONE")
            self.last_match_center = None
