
from minimap_match import *

class FeatureMatcher:
    """特征匹配类，负责特征匹配和结果可视化"""

    def __init__(self, template_path: str, config: dict):
        self.config = config
        self.region_name = os.path.basename(template_path).replace("_features.npz", "")
        self.kp_template, self.des_template = self._load_template_features(template_path)
        self.last_match_center = None
        self.sift = cv2.SIFT_create(nOctaveLayers=5, contrastThreshold=0.01, edgeThreshold=15)
        self.flann = cv2.FlannBasedMatcher(
            dict(algorithm=1, table_number=6, key_size=12),
            dict(checks=config["flann_checks"])
        )
    @staticmethod
    def _load_template_features(filename: str) -> tuple:
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

        # 修改点：加入缩放比例过滤条件
        good_matches = []
        for m, n in matches:
            # 原始的距离比例条件
            if m.distance < self.config["match_ratio"] * n.distance:
                # 新增缩放比例条件
                query_scale = kp_query[m.queryIdx].size  # 查询关键点的尺度
                template_scale = kp_template[m.trainIdx].size  # 模板关键点的尺度
                scale_ratio = query_scale / template_scale

                # 通过阈值过滤缩放比例异常的匹配
                if (0.6) <= scale_ratio<= (1.3):
                    good_matches.append(m)

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
            if (kp.pt[0] - cx) ** 2 + (kp.pt[1] - cy) ** 2 <= 150 ** 2
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
        if matches:  # 确保有匹配
            first_match = matches[0]
            query_scale = kp_query[first_match.queryIdx].size
            template_scale = kp_template[first_match.trainIdx].size
            scale_ratio = query_scale / template_scale
        else:
            scale_ratio = 1.0  # 无匹配时默认比例

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
        cx = w / 2
        cy = h / 2
        # 应用仿射变换矩阵到中心点
        center_x = M[0, 0] * cx + M[0, 1] * cy + M[0, 2]
        center_y = M[1, 0] * cx + M[1, 1] * cy + M[1, 2]
        return (int(center_x), int(center_y))
    def _update_match_center(self, matches_count, center):
        """更新匹配中心位置"""
        if matches_count >= self.config["min_matches"]:
            self.last_match_center = center
            logger.info(f"更新匹配中心到: {center}")
        else:
            logger.info("NONE")
            self.last_match_center = None
