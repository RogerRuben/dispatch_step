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
    TAU_SAFE_Q3,
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
    n_orders = len(orders)
    n_vehicles = len(vehicles)

    cost = np.full((n_orders, n_vehicles), BIG_M, dtype=np.float64)
    pickup_times = np.full((n_orders, n_vehicles), DEFAULT_PICKUP_TIME, dtype=np.float64)

    for i, order in enumerate(orders):
        o_link = order.get("origin_link")
        grade = order.get("grade_num", 2)
        difficulty = float(order.get("difficulty", 0.0))
        cong_prob = float(order.get("pred_cong_prob", 0.05))
        risk_prob = float(order.get("pred_risk_prob", 0.01))
        entropy = float(order.get("pred_entropy", 0.5))
        eta = float(order.get("simple_eta", 300))

        for j, vehicle in enumerate(vehicles):
            # ---- 不可行检查 ----
            if vehicle.is_av():
                if grade == 4:
                    continue
                if risk_prob > TAU_SAFE_Q3:
                    continue

            # ---- 接驾时间 ----
            v_link = vehicle.location_link
            remaining = 0.0

            if vehicle.status == "NEAR_FREE":
                # ★ 修复：用释放位置 + 加上剩余完成时间
                v_link = vehicle.release_link
                remaining = max(vehicle.busy_until - current_time, 0.0)

            pt = estimate_pickup_time(v_link, o_link, link_distances, remaining)
            pickup_times[i, j] = pt

            # ---- 成本计算 ----
            if vehicle.is_av():
                if grade <= 2:
                    # ★ G1/G2: AV 有 10% 效率优势
                    av_bonus = -0.10 * eta
                else:
                    av_bonus = 0.0

                av_penalty = (
                    ALPHA_DIFFICULTY * difficulty * cong_prob
                    + BETA_RISK * risk_prob * (1.0 + entropy)
                    + GAMMA_UNCERTAINTY * entropy
                ) * eta

                cost[i, j] = pt + av_penalty + av_bonus
            else:
                # HV: 纯接驾时间
                cost[i, j] = pt

    return cost, pickup_times


def filter_infeasible(cost_matrix: np.ndarray) -> np.ndarray:
    """可行性掩码"""
    return cost_matrix < BIG_M * 0.5