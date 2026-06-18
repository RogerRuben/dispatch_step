"""
core/cost.py — 成本矩阵构建（修复版）

修复:
  1. NEAR_FREE 车辆的接驾时间加上剩余完成时间
  2. compute_cost_matrix 接受 current_time 参数
  3. AV 在低难度订单上有成本优势
"""
import numpy as np
from core.config import (
    DEFAULT_PICKUP_TIME, BIG_M,
    ALPHA_DIFFICULTY, BETA_RISK, GAMMA_UNCERTAINTY,
    TAU_SAFE_Q3,AV_COST_DISCOUNT, AV_COST_ADVANTAGE_ENABLED
)


def estimate_pickup_time(vehicle_link: int, order_origin_link: int,
                         link_distances: dict,
                         remaining_time: float = 0.0) -> float:
    """
    估算接驾时间。

    Parameters
    ----------
    vehicle_link : 车辆当前/释放位置 link
    order_origin_link : 订单起点 link
    link_distances : 代理距离表
    remaining_time : NEAR_FREE 车辆的剩余完成时间（秒）

    Returns
    -------
    总接驾时间 = 剩余等待 + 空驶时间
    """
    if vehicle_link is None or order_origin_link is None:
        return DEFAULT_PICKUP_TIME + remaining_time

    v_link = int(vehicle_link)
    o_link = int(order_origin_link)

    # 同一个 link
    if v_link == o_link:
        return 10.0 + remaining_time

    # 正向查
    travel = None
    if v_link in link_distances:
        if o_link in link_distances[v_link]:
            travel = float(link_distances[v_link][o_link])

    # 反向查
    if travel is None and o_link in link_distances:
        if v_link in link_distances[o_link]:
            travel = float(link_distances[o_link][v_link])

    # 都查不到用默认值
    if travel is None:
        travel = DEFAULT_PICKUP_TIME

    return travel + remaining_time


def compute_cost_matrix(orders: list, vehicles: list,
                        link_distances: dict,
                        current_time: float = 0.0) -> tuple:
    """
    构建 (n_orders, n_vehicles) 成本矩阵。

    Parameters
    ----------
    orders : list of dict
    vehicles : list of Vehicle
    link_distances : 代理距离表
    current_time : 当前仿真时间（用于计算 NEAR_FREE 剩余时间）

    Returns
    -------
    cost : np.ndarray (n_orders, n_vehicles)
    pickup_times : np.ndarray (n_orders, n_vehicles)
    """
    n_o, n_v = len(orders), len(vehicles)
    cost_mat = np.full((n_o, n_v), BIG_M, dtype=np.float64)
    pickup_mat = np.full((n_o, n_v), BIG_M, dtype=np.float64)  # 实际接驾时间（记录用）

    for i, order in enumerate(orders):
        o_link = int(order.get("origin_link", 0))
        for j, vehicle in enumerate(vehicles):
            # 1. 计算车辆状态带来的硬约束（剩余服务时间）
            remaining_time = 0.0
            if vehicle.status == "NEAR_FREE":
                remaining_time = max(0.0, vehicle.busy_until - current_time)
            elif vehicle.status == "BUSY":
                # 理论上 BUSY 不会进入可用池，但做防御
                remaining_time = max(0.0, vehicle.busy_until - current_time)

            # 2. 计算空驶接驾时间（纯距离成本）
            v_link = int(vehicle.release_link if vehicle.status in ("NEAR_FREE", "BUSY") else vehicle.location_link)
            travel_time = DEFAULT_PICKUP_TIME  # 默认值

            if v_link == o_link:
                travel_time = 10.0  # 同一 link 最小接驾时间
            else:
                # 双向查表
                if v_link in link_distances and o_link in link_distances[v_link]:
                    travel_time = float(link_distances[v_link][o_link])
                elif o_link in link_distances and v_link in link_distances[o_link]:
                    travel_time = float(link_distances[o_link][v_link])
                else:
                    travel_time = DEFAULT_PICKUP_TIME  # 保底

            # 3. ★★★ 核心修复：AV 空驶时间享受边际成本折扣 ★★★
            if AV_COST_ADVANTAGE_ENABLED and vehicle.is_av():
                # 仅对空驶部分打折，剩余时间（乘客在车上）必须原价，否则会破坏时空连续性
                discounted_travel = travel_time * AV_COST_DISCOUNT
                total_cost = discounted_travel + remaining_time
            else:
                total_cost = travel_time + remaining_time

            # 实际物理接驾时间（用于记录和用户体验统计）
            actual_pickup = travel_time + remaining_time

            # 写入矩阵
            cost_mat[i, j] = total_cost
            pickup_mat[i, j] = actual_pickup

    return cost_mat, pickup_mat


def filter_infeasible(cost_matrix: np.ndarray) -> np.ndarray:
    """可行性掩码"""
    return cost_matrix < BIG_M * 0.5