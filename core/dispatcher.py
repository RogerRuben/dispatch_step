"""
core/dispatcher.py — 分层匹配调度器

分层匹配策略:
  Layer 1: G4 订单 → 只从 HV 中匹配
  Layer 2: G1 订单 → 优先从 AV 中匹配
  Layer 3: G2/G3 订单 → 在剩余车辆中匹配（带难度成本修正）

每层内部用匈牙利算法（scipy.optimize.linear_sum_assignment）求最优匹配。
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from core.config import BIG_M


class TieredDispatcher:
    """
    分层匹配调度器。

    设计理念:
      - 安全优先: G4 先处理，保证不给 AV
      - 效率优先: G1 优先给 AV（AV 的优势场景）
      - 剩余订单: G2/G3 在所有剩余车辆中做最优匹配
    """

    def __init__(self, name="TieredDispatcher"):
        self.name = name

    def dispatch(self, orders: list, vehicles: list,
                 cost_matrix: np.ndarray,
                 pickup_times: np.ndarray) -> tuple:
        """
        执行分层匹配。

        Parameters
        ----------
        orders : list of dict
            当前窗口的订单
        vehicles : list of Vehicle
            当前可用车辆
        cost_matrix : (n_orders, n_vehicles)
        pickup_times : (n_orders, n_vehicles)

        Returns
        -------
        matched : list of (order_idx, vehicle_idx, pickup_time)
        unmatched_order_indices : list of int
        """
        n_orders = len(orders)
        n_vehicles = len(vehicles)

        if n_orders == 0 or n_vehicles == 0:
            return [], list(range(n_orders))

        # 已使用的订单和车辆索引
        used_orders = set()
        used_vehicles = set()
        all_matches = []

        # ---- Layer 1: G4 订单 → HV only ----
        g4_orders = [i for i, o in enumerate(orders) if o.get("grade_num", 2) == 4]
        hv_vehicles = [j for j, v in enumerate(vehicles) if not v.is_av()]

        if g4_orders and hv_vehicles:
            matches = self._match_subset(
                g4_orders, hv_vehicles, cost_matrix, pickup_times
            )
            for oi, vi, pt in matches:
                used_orders.add(oi)
                used_vehicles.add(vi)
                all_matches.append((oi, vi, pt))

        # ---- Layer 2: G1 订单 → AV preferred ----
        g1_orders = [
            i for i, o in enumerate(orders)
            if o.get("grade_num", 2) == 1 and i not in used_orders
        ]
        av_vehicles = [
            j for j, v in enumerate(vehicles)
            if v.is_av() and j not in used_vehicles
        ]

        if g1_orders and av_vehicles:
            matches = self._match_subset(
                g1_orders, av_vehicles, cost_matrix, pickup_times
            )
            for oi, vi, pt in matches:
                used_orders.add(oi)
                used_vehicles.add(vi)
                all_matches.append((oi, vi, pt))

        # ---- Layer 3: 剩余订单 → 剩余车辆 ----
        remaining_orders = [
            i for i in range(n_orders) if i not in used_orders
        ]
        remaining_vehicles = [
            j for j in range(n_vehicles) if j not in used_vehicles
        ]

        if remaining_orders and remaining_vehicles:
            matches = self._match_subset(
                remaining_orders, remaining_vehicles,
                cost_matrix, pickup_times
            )
            for oi, vi, pt in matches:
                used_orders.add(oi)
                used_vehicles.add(vi)
                all_matches.append((oi, vi, pt))

        # 未匹配订单
        unmatched = [i for i in range(n_orders) if i not in used_orders]

        return all_matches, unmatched

    def _match_subset(self, order_indices: list, vehicle_indices: list,
                      full_cost: np.ndarray,
                      full_pickup: np.ndarray) -> list:
        """
        在子集上做匈牙利匹配。

        Returns
        -------
        matches : list of (original_order_idx, original_vehicle_idx, pickup_time)
        """
        n_o = len(order_indices)
        n_v = len(vehicle_indices)

        # 提取子矩阵
        sub_cost = np.full((n_o, n_v), BIG_M, dtype=np.float64)
        sub_pickup = np.full((n_o, n_v), 0.0, dtype=np.float64)

        for ii, oi in enumerate(order_indices):
            for jj, vj in enumerate(vehicle_indices):
                sub_cost[ii, jj] = full_cost[oi, vj]
                sub_pickup[ii, jj] = full_pickup[oi, vj]

        # 匈牙利匹配
        # 如果订单多于车辆，需要 pad；反之亦然
        max_dim = max(n_o, n_v)
        padded_cost = np.full((max_dim, max_dim), BIG_M, dtype=np.float64)
        padded_cost[:n_o, :n_v] = sub_cost

        row_ind, col_ind = linear_sum_assignment(padded_cost)

        matches = []
        for r, c in zip(row_ind, col_ind):
            if r < n_o and c < n_v:
                if sub_cost[r, c] < BIG_M * 0.5:
                    orig_oi = order_indices[r]
                    orig_vi = vehicle_indices[c]
                    pt = sub_pickup[r, c]
                    matches.append((orig_oi, orig_vi, pt))

        return matches