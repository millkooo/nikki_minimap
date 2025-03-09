import time
import logging
from minimap_match.speed_calculator import SpeedCalculator

logger = logging.getLogger(__name__)

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
        # 添加速度计算器
        self.speed_calculator = SpeedCalculator()

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
