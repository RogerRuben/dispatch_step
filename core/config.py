"""
core/config.py — Stage 4 全局参数配置
"""
import os

# ============================================================
# 路径配置
# ============================================================
# Stage 4 根目录
STAGE4_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Stage 1-3 代码目录（同级）
STAGE123_DIR = os.path.join(os.path.dirname(STAGE4_DIR), "predict_access_step")

# 原始数据目录
DATA_DIR = os.path.join(os.path.dirname(STAGE4_DIR), "output", "data_split")
TOPO_PATH = os.path.join(os.path.dirname(STAGE4_DIR), "output", "nextlinks", "nextlinks.txt")

# Stage 4 数据目录
DISPATCH_DATA_DIR = os.path.join(STAGE4_DIR, "dispatch_data")
RESULTS_DIR = os.path.join(STAGE4_DIR, "results")
FIGURES_DIR = os.path.join(STAGE4_DIR, "figures")

for d in [DISPATCH_DATA_DIR, RESULTS_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# 调度参数
# ============================================================
DISPATCH_INTERVAL = 120          # 秒（2 分钟一个调度窗口）
THETA_NEAR_FREE = 120            # 在途预指派阈值（秒）
SECONDS_PER_DAY = 86400

# ============================================================
# 车队参数
# ============================================================
AV_RATIO = 0.1                  # AV 占总车队比例
AV_ALWAYS_ONLINE = True          # AV 全天在线
HV_POOL_SCALE = 0.1            # HV 供给缩放系数
# ============================================================
# 区域划分
# ============================================================
N_ZONES = 50                     # 拓扑区域数
TOPO_MAX_HOPS = 5                # link 邻居距离表的最大跳数
MAX_CANDIDATE_ZONES = 3          # 每个订单最多从几个相邻 zone 借车

# ============================================================
# 成本函数参数
# ============================================================
DEFAULT_PICKUP_TIME = 300.0      # 代理距离查不到时的默认接驾时间（秒）
BIG_M = 1e7                      # 不可行配对的惩罚成本

# AV 执行高难度订单的额外成本系数
ALPHA_DIFFICULTY = 0.5           # 难度折损
BETA_RISK = 1.0                  # 风险惩罚
GAMMA_UNCERTAINTY = 0.3          # 不确定性惩罚

# ============================================================
# 安全约束
# ============================================================
TAU_SAFE_Q3 = 0.30               # q3 > 此值的订单不允许 AV 执行

# ============================================================
# 实验参数
# ============================================================
DAY = "30"
RANDOM_SEED = 42

# Stage 3 输出文件（难度评分）
DIFFICULTY_CSV = os.path.join(STAGE123_DIR, "difficulty_test.csv")

# ============================================================
# 敏感性分析
# ============================================================
SENSITIVITY_AV_RATIOS = [0.10, 0.20, 0.30]
SENSITIVITY_THETAS = [60, 120, 180, 300]
SENSITIVITY_ZONES = [20, 50, 100]

# ============================================================
# 列名映射（兼容原始数据的列名处理）
# ============================================================
COLUMN_RENAME = {
    "order id": "order_id", "cross id": "cross_id", "cross time": "cross_time",
    "link id": "link_id", "link time": "link_time", "link ratio": "link_ratio",
    "link current status": "link_current_status",
    "link arrival status": "link_arrival_status",
    "simple eta": "simple_eta", "driver id": "driver_id", "slice id": "slice_id",
}

SPLIT_SUBDIRS = {
    "head": "head_split",
    "link": "link_split",
    "cross": "cross_split",
}
MAX_CANDIDATES_PER_ZONE = 200
# ============================================================
# 供给侧校准
# ============================================================
# 目标供需比：每个窗口的目标在线车辆数 = ratio × 当前窗口订单数
# 分时段设置（更贴近真实运营）
TARGET_SUPPLY_RATIOS = {
    "peak":    1.3,    # 高峰（7-9, 17-19）：供给略紧
    "normal":  1.8,    # 平峰：供给适中
    "night":   2.5,    # 夜间（22-6）：供给宽松（车少单少）
}

# AV 目标利用率（AV 池大小 = AV总数 × 此系数）
AV_ONLINE_RATIO = 1.0   # AV 全天全量在线

# HV 最小在线数（防止极端低谷时 pool 太小）
HV_MIN_ONLINE = 50

# ========== 新增 AV 成本优势配置 ==========
# AV 空驶边际成本折扣（相对于 HV）。设为 0.6 表示 AV 跑 1 分钟空驶的成本仅相当于 HV 的 0.6 分钟。
# 学术参考值：AV 运营成本约为 HV 的 40%~70%。
AV_COST_DISCOUNT = 0.6

# 是否在成本矩阵中启用 AV 成本优势（便于开关对比实验）
AV_COST_ADVANTAGE_ENABLED = True
# ========== 新增空闲车辆巡游配置 ==========
RELOCATION_ENABLED = True          # 是否开启空闲重定位
RELOCATION_STEP_LINKS = 2          # 每个窗口（2分钟）移动的 link 步数（约 200-400 米）
RELOCATION_MIN_IDLE_WINDOWS = 2    # 车辆空闲至少 2 个窗口（4分钟）后才开始巡游，避免频繁微动