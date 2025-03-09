import logging

logger = logging.getLogger(__name__)

class BoundaryManager:
    def __init__(self, config):
        self.config = config
        self.boundary_status = {}
        
    def check_boundaries(self, x, y, state):
        """边界检测并返回需要预加载的区域（带状态跟踪和自动卸载）"""
        required_regions = set()
        active_boundaries = set()

        # 第一步：检测当前处于哪些边界
        for boundary in self.config["matching"]["boundaries"]:
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
                logger.info(f"进入边界区域：{boundary['regions']}")
            elif not in_boundary and prev_status:
                logger.info(f"离开边界区域：{boundary['regions']}")

        # 第二步：清理过期边界状态
        to_remove = []
        for bid in state.boundary_status:
            if bid not in active_boundaries:
                if state.boundary_status[bid] >= 20:  # 连续20帧不在边界内卸载
                    to_remove.append(bid)
                else:
                    state.boundary_status[bid] += 1
        for bid in to_remove:
            regions = next(b["regions"] for b in self.config["matching"]["boundaries"] if id(b) == bid)
            logger.info(f"卸载边界区域：{regions}")
            del state.boundary_status[bid]

        # 第三步：合并所有活跃边界的区域
        return list(required_regions)
    
    def update_matchers(self, required_regions, state, features_matcher, current_match=None):
        """智能更新匹配器列表（带自动清理）"""
        # 必须保留的匹配器：当前区域 + 活跃边界区域
        must_keep = set()
        if state.current_region:
            must_keep.add(state.current_region)
        must_keep.update(required_regions)

        # 过滤无效区域
        valid_regions = [r for r in must_keep if r in features_matcher]

        # 计算需要卸载的匹配器
        current_matchers = {m.region_name for m in state.active_matchers}
        
        # 只有在已经有匹配结果且当前区域确定，并且特征点数量达到最小要求时才卸载
        if (state.current_region and current_match and 
            current_match.get("matches", 0) >= self.config["matching"]["min_matches"]):
            to_remove = current_matchers - set(valid_regions)
            
            # 执行卸载
            if to_remove:
                logger.info(f"卸载未使用区域：{list(to_remove)}，当前匹配点数：{current_match['matches']}")
                state.active_matchers = [m for m in state.active_matchers if m.region_name not in to_remove]
        
        # 添加需要的新匹配器
        for region in valid_regions:
            if region not in current_matchers:
                matcher = features_matcher[region]
                state.active_matchers.append(matcher)
                logger.info(f"加载区域匹配器：{region}") 