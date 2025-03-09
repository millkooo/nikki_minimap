import logging
import time
import cv2
import numpy as np
import win32con
import win32gui
from Ui_Manage.WindowManager import WinControl
from capture.img_processor import ImageProcessor
from match.template_matcher import TemplateMatcher
from minimap_match.FeatureMatch import FeatureMatcher
from minimap_match.AngleDetector import AdaptiveAngleDetector

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 常量配置
DEFAULT_CONFIG = {
    "window": {
        "title_part": "无限暖暖",
        "window_class": "UnrealWindow",
        "process_exe": "X6Game-Win64-Shipping.exe"
    },
    "capture": {
        #截图的参数（小地图）
        "list": [ { "x_offset": 77,"y_offset": 38,"width": 204,"height": 204},
                  {"x_offset": 0, "y_offset": 0, "width": 1920, "height": 1080},],
        "diameter": 204
    },
    "matching": {
        "boundaries": [
            {
                "range": (5600, 4900, 6100, 6500),  # x1,y1,x2,y2
                "regions": ["huayuanzhen", "xiaoshishu"]  # 相邻区域
            },
            {
                "range": (9427, 4175, 9767, 4390),  # x1,y1,x2,y2
                "regions": ["huayuanzhen", "qiyuansenlin"]  # 相邻区域
            },
            {
                "range": (6540, 4290, 7450, 4520),  # x1,y1,x2,y2
                "regions": ["huayuanzhen", "weifenglvye"]  # 相邻区域
            },

        ],
        "region_features": {
            "huayuanzhen": "resources/features/huayuanzhen_features.npz",
            "qiyuansenlin": "resources/features/qiyuansenlin_features.npz",
            "weifenglvye": "resources/features/weifenglvye_features.npz",
            "xiaoshishu": "resources/features/xiaoshishu_features.npz",
            "huayanqundao": "resources/features/huayanqundao_features.npz"
        },
        "min_matches": 10,#更新匹配中心所需匹配的特征点数量
        "flann_checks": 50,#搜索的次数和准确性。较大的 checks 值会提高匹配的准确性
        "match_ratio": 0.8,#控制特征匹配严格程度，默认0.7-0.8
        "max_angle": 5,#识别允许的最大偏转角
        "fixed_scale": 1.17,#默认的缩放比例
        "min_scale_ratio":1,
        "max_scale_ratio":1.2
    },
    "templates": {
        "configs": [
            {
                "name": "arrow",
                "path": "resources/img/arrow.png",
                "threshold": 0.97
            },
            {
                "name": "no_minimap",
                "path": "resources/img/no_minimap.png",
                "threshold": 0.97
            },
            {
                "name": "loading",
                "path": "resources/img/loading.png",
                "threshold": 0.98
            }
        ]
    }
}


class ResultVisualizer:
    """结果可视化类，负责结果显示和窗口管理"""
    @staticmethod
    def show_match_result(result: dict, template_image: np.ndarray,angle):
        """显示匹配结果"""
        display_img = template_image.copy()
        M = result["transform"]
        center = result["center"]

        # 绘制变换后的边界框
        h, w = template_image.shape[:2]
        corners = np.array([
            [-w / 2, -h / 2], [w / 2, -h / 2],
            [w / 2, h / 2], [-w / 2, h / 2]
        ])
        rotated_corners = np.dot(corners, M[:2, :2].T) + [M[0, 2], M[1, 2]]
        pts = np.int32(rotated_corners.reshape(-1, 1, 2))
        cv2.polylines(display_img, [pts], True, (0, 255, 0), 2)

        # 绘制中心点
        cv2.circle(display_img, center, 5, (0, 0, 255), -1)
        print(angle)
        # 将角度转换为弧度
        angle_rad = np.deg2rad(angle)
        # 计算箭头终点坐标
        start_x = center[0]- 8 * np.cos(angle_rad)
        start_y = center[1]+ 8 * np.sin(angle_rad)
        end_x = center[0] + 14 * np.cos(angle_rad)
        end_y = center[1] - 14 * np.sin(angle_rad)
        start_point = (int(start_x),int(start_y))

        end_point = (int(end_x), int(end_y))

        # 绘制箭杆
        cv2.line(display_img, start_point, end_point, (255, 255, 0), 2)

        # 计算箭头头的两个辅助点
        # 箭头头的方向
        theta_rad = np.deg2rad(25)
        left_angle = angle_rad + np.pi - theta_rad  # 左分支角度
        right_angle = angle_rad + np.pi + theta_rad  # 右分支角度

        # 计算左右分支端点
        left_end = (
            int(end_x + 8 * np.cos(left_angle)),
            int(end_y - 8 * np.sin(left_angle))  # 注意保持y轴方向一致
        )

        right_end = (
            int(end_x + 8 * np.cos(right_angle)),
            int(end_y - 8 * np.sin(right_angle))
        )
        # 绘制箭头分支
        cv2.line(display_img, end_point, left_end, (255, 255, 0), 2)
        cv2.line(display_img, end_point, right_end, (255, 255, 0), 2)
        # 创建显示区域
        cropped = ResultVisualizer._create_viewport(display_img, center, 100)
        cv2.putText(cropped, f'Matches: {result["matches"]}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示结果
        cv2.namedWindow('Matching Result', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Matching Result', 200, 200)
        cv2.imshow('Matching Result', cropped)
        ResultVisualizer.set_topmost('Matching Result')

    @staticmethod
    def _create_viewport(image: np.ndarray, center: tuple, size: int) -> np.ndarray:
        """创建视口区域"""
        h, w = image.shape[:2]
        x1 = max(0, center[0] - size)
        y1 = max(0, center[1] - size)
        x2 = min(w, center[0] + size)
        y2 = min(h, center[1] + size)
        return image[y1:y2, x1:x2]

    @staticmethod
    def set_topmost(window_name: str):
        """设置窗口置顶"""
        hwnd = win32gui.FindWindow(None, window_name)
        if hwnd:
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0,
                                  win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)


class NavigationState:
    def __init__(self):
        self.current_region = None
        self.active_matchers = []
        self.last_position = (0, 0)
        self.last_angle = 90
        self.prev_hash = None
        self.prev_result = None
        self.boundary_status = {}  # 修正为字典类型
        self.in_panduan = False  # 保留兼容性
        self.last_panduan_check = 0  # 保留兼容性

        # 模板匹配状态管理
        self.template_state = {
            "active": False,       # 是否处于模板匹配状态
            "current_template": None,  # 当前匹配到的模板名称
            "previous_template": None, # 上一个模板名称
            "last_check_time": 0,     # 上次检查时间
            "check_interval": 0.5,    # 检查间隔（秒）
            "history": {}         # 记录各模板状态历史
        }
    
    def enter_template_state(self, template_name):
        """进入模板匹配状态"""
        self.template_state["active"] = True
        self.template_state["current_template"] = template_name
        self.template_state["last_check_time"] = time.time()
        
        # 记录状态历史
        if template_name not in self.template_state["history"]:
            self.template_state["history"][template_name] = {
                "count": 0,
                "last_seen": time.time()
            }
        
        self.template_state["history"][template_name]["count"] += 1
        self.template_state["history"][template_name]["last_seen"] = time.time()
        
        logger.info(f"进入模板状态: {template_name}")
    
    def exit_template_state(self):
        """退出模板匹配状态"""
        if self.template_state["active"]:
            template_name = self.template_state["current_template"]
            logger.info(f"退出模板状态: {template_name}")
            
            # 更新状态历史
            if template_name in self.template_state["history"]:
                self.template_state["history"][template_name]["duration"] = \
                    time.time() - self.template_state["history"][template_name]["last_seen"]
            
            # 保存上一个模板状态，用于后续处理
            self.template_state["previous_template"] = template_name
            
            self.template_state["active"] = False
            self.template_state["current_template"] = None
    
    def should_check_template(self):
        """是否应该检查模板状态"""
        current_time = time.time()
        if current_time - self.template_state["last_check_time"] >= self.template_state["check_interval"]:
            self.template_state["last_check_time"] = current_time
            return True
        return False
        
    def get_template_status(self, template_name=None):
        """获取模板状态信息"""
        if template_name:
            if template_name in self.template_state["history"]:
                return self.template_state["history"][template_name]
            return None
        return self.template_state["history"]


def main():
    """主程序"""
    # 初始化
    target_hwnd = WinControl.find_target_window(DEFAULT_CONFIG["window"])
    if not target_hwnd:
        logger.error("未找到目标窗口")
        return

    WinControl.activate_window(target_hwnd)
    image_processor = ImageProcessor(target_hwnd)
    # 加载模板图
    huayan_img = cv2.imread('resources/img/huayanqundao.png', cv2.IMREAD_GRAYSCALE)
    xinyuanyuanye_img = cv2.imread('resources/img/nuanuan_map.png', cv2.IMREAD_GRAYSCALE)

    # 初始化模板匹配器
    template_matcher = TemplateMatcher()
    template_matcher.load_templates(DEFAULT_CONFIG["templates"]["configs"])

    features_matcher = {}
    for region, path in DEFAULT_CONFIG["matching"]["region_features"].items():
        features_matcher[region] = FeatureMatcher(path, DEFAULT_CONFIG["matching"])

    state = NavigationState()
    AngleD = AdaptiveAngleDetector(
        threshold_angle=45,  # 目标夹角为45度
        angle_tolerance=5,   # 角度容差为±5度
        initial_threshold=200,  # 初始阈值
        min_threshold=180,    # 最小阈值
        threshold_step=5,    # 阈值递减步长
        hough_params=(20, 16, 10),  # 霍夫变换参数
        save_debug_images=False  # 保存调试图像
    )

    def check_boundaries(x, y):
        """边界检测并返回需要预加载的区域（带状态跟踪和自动卸载）"""
        required_regions = set()
        active_boundaries = set()

        # 第一步：检测当前处于哪些边界
        for boundary in DEFAULT_CONFIG["matching"]["boundaries"]:
            bid = id(boundary)
            x1, y1, x2, y2 = boundary["range"]
            in_boundary = x1 <= x <= x2 and y1 <= y <= y2

            # 记录当前活跃边界
            if in_boundary:
                active_boundaries.add(bid)
                required_regions.update(boundary["regions"])

            # 更新边界状态变化
            prev_status = bid in state.boundary_status
            if in_boundary and not prev_status:
                print(f"进入边界区域：{boundary['regions']}")
            elif not in_boundary and prev_status:
                print(f"离开边界区域：{boundary['regions']}")

        # 第二步：清理过期边界状态
        to_remove = []
        for bid in state.boundary_status:
            if bid not in active_boundaries:
                if state.boundary_status[bid] >= 20:  # 连续20帧不在边界内卸载
                    to_remove.append(bid)
                else:
                    state.boundary_status[bid] += 1
        for bid in to_remove:
            regions = next(b["regions"] for b in DEFAULT_CONFIG["matching"]["boundaries"] if id(b) == bid)
            print(f"卸载边界区域：{regions}")
            del state.boundary_status[bid]

        # 第三步：合并所有活跃边界的区域
        return list(required_regions)

    def update_matchers(required_regions):
        """智能更新匹配器列表（带自动清理）"""
        # 必须保留的匹配器：当前区域 + 活跃边界区域
        must_keep = {state.current_region}
        must_keep.update(required_regions)

        # 过滤无效区域
        valid_regions = [r for r in must_keep if r in features_matcher]

        # 计算需要卸载的匹配器
        current_matchers = {m.region_name for m in state.active_matchers}
        to_remove = current_matchers - set(valid_regions)

        # 执行卸载
        if to_remove:
            print(f"卸载未使用区域：{list(to_remove)}")
            state.active_matchers = [m for m in state.active_matchers if m.region_name not in to_remove]

        # 添加需要的新匹配器
        for region in valid_regions:
            if region not in current_matchers:
                matcher = features_matcher[region]
                state.active_matchers.append(matcher)
                print(f"加载区域匹配器：{region}")
    try:
        while True:

            # 检查是否处于模板匹配状态
            if state.template_state["active"]:
                # 定期检查模板状态是否仍然存在
                if state.should_check_template():
                    frame = image_processor.capture_region({ "x_offset": 0,"y_offset": 0,"width": 281,"height": 242})

                    template_result = template_matcher.match_template(frame, state.template_state["current_template"])
                    
                    if not template_result:
                        # 模板不再匹配，退出模板状态
                        state.exit_template_state()
                        logger.info(f"模板 {state.template_state['current_template']} 不再匹配，恢复正常检测")
                    else:
                        # 模板仍然匹配，继续保持当前状态
                        logger.debug(f"模板 {template_result['name']} 仍然匹配，得分: {template_result['score']:.2f}")
                continue

            frame = image_processor.capture_region({ "x_offset": 0,"y_offset": 0,"width": 281,"height": 242})
            # 先进行模板匹配检测
            if not state.template_state["active"]:
                # 检查是否刚从loading状态退出
                if state.template_state["previous_template"] == "loading":
                    logger.info("检测到从loading状态退出，加载全部npz文件进行匹配")
                    # 加载所有区域的匹配器
                    regions = list(DEFAULT_CONFIG["matching"]["region_features"].keys())
                    update_matchers(regions)
                    # 重置previous_template，避免重复处理
                    state.template_state["previous_template"] = None
                
                template_result = template_matcher.match_template(frame)
                print(template_result)
                if template_result:
                    # 匹配到模板，进入模板状态
                    state.enter_template_state(template_result["name"])
                    logger.info(f"检测到模板 {template_result['name']}，得分: {template_result['score']:.2f}，暂停SIFT检测")
                    continue
            frame1 = image_processor._crop_image(frame,{"x": 77, "y": 38, "w": 204, "h": 204})
            circular_frame = ImageProcessor.circle_crop(frame1, DEFAULT_CONFIG["capture"]["diameter"])
            crop_imgg = ImageProcessor.circle_crop(circular_frame, 32)

            # 全局哈希检查
            current_hash = ImageProcessor.compute_image_hash(circular_frame)
            if current_hash == state.prev_hash and state.prev_result:
                #print("检测到相同帧，复用结果")
                best_match = state.prev_result
                time.sleep(0.3)
            else:
                state.prev_hash = current_hash
                best_match = None

                # 动态加载匹配器
                if state.current_region:
                    # 获取所有需要预加载的边界区域
                    boundary_regions = check_boundaries(*state.last_position)
                    # 智能更新匹配器（自动清理+加载）
                    update_matchers(boundary_regions)
                else:
                    # 初始状态加载所有匹配器
                    regions = list(DEFAULT_CONFIG["matching"]["region_features"].keys())
                    update_matchers(regions)

                # 特征匹配
                for matcher in state.active_matchers:
                    result = matcher.process_frame(circular_frame)
                    if result and result["matches"] >= DEFAULT_CONFIG["matching"]["min_matches"]:
                        if not best_match or result["matches"] > best_match["matches"]:
                            best_match = result
                            state.current_region = matcher.region_name
                            state.last_position = result["center"]
                            state.prev_result = best_match  # 保存最新结果
                
                # 角度检测（移到循环外部，确保每帧都能正确检测角度）
                jiaodu = AngleD.calculate_angle(crop_imgg)
                if jiaodu:
                    state.last_angle = jiaodu
                    print(f"检测到角度: {jiaodu}°")
            # 4. 显示逻辑
            if best_match:
                if state.current_region == "huayanqundao":
                    ResultVisualizer.show_match_result(best_match, huayan_img, state.last_angle)
                else:
                    ResultVisualizer.show_match_result(best_match, xinyuanyuanye_img, state.last_angle)

            if cv2.waitKey(1) == 27:
                break


    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()