"""
core/baselines.py — 基线调度策略

基线 1: RandomDispatcher    — 随机指派
基线 2: NearestDispatcher   — 就近指派（不区分 AV/HV）
基线 3: NearestSafeDispatcher — 就近但 G4/高风险不给 AV
基线 4: GradeOnlyDispatcher — 用 Grade 分层但不用连续难度修正成本
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from core.config import BIG_M


class RandomDispatcher:
    """随机指派：随机配对可行的订单和车辆"""

    def __init__(self):
        self.name = "Random"

    def dispatch(self, orders, vehicles, cost_matrix, pickup_times):
        n_o = len(orders)
        n_v = len(vehicles)

        if n_o == 0 or n_v == 0:
            return [], list(range(n_o))

        feasible = cost_matrix < BIG_M * 0.5

        matched = []
        used_vehicles = set()
        order_indices = list(range(n_o))
        np.random.shuffle(order_indices)

        for oi in order_indices:
            candidates = [
                j for j in range(n_v)
                if j not in used_vehicles and feasible[oi, j]
            ]
            if candidates:
                vj = candidates[np.random.randint(len(candidates))]
                matched.append((oi, vj, pickup_times[oi, vj]))
                used_vehicles.add(vj)

        matched_ois = {m[0] for m in matched}
        unmatched = [i for i in range(n_o) if i not in matched_ois]

        return matched, unmatched


class NearestDispatcher:
    """就近指派：每个订单选接驾时间最短的可行车辆"""

    def __init__(self):
        self.name = "Nearest"

    def dispatch(self, orders, vehicles, cost_matrix, pickup_times):
        n_o = len(orders)
        n_v = len(vehicles)

        if n_o == 0 or n_v == 0:
            return [], list(range(n_o))

        feasible = cost_matrix < BIG_M * 0.5

        matched = []
        used_vehicles = set()

        # 按接驾时间排序每个订单的候选
        for oi in range(n_o):
            best_vj = None
            best_pt = float("inf")

            for vj in range(n_v):
                if vj in used_vehicles:
                    continue
                if not feasible[oi, vj]:
                    continue
                pt = pickup_times[oi, vj]
                if pt < best_pt:
                    best_pt = pt
                    best_vj = vj

            if best_vj is not None:
                matched.append((oi, best_vj, best_pt))
                used_vehicles.add(best_vj)

        matched_ois = {m[0] for m in matched}
        unmatched = [i for i in range(n_o) if i not in matched_ois]

        return matched, unmatched


class NearestSafeDispatcher:
    """
    就近安全指派：
    和 Nearest 一样选最近，但 G4 和高风险订单不给 AV。
    """

    def __init__(self):
        self.name = "NearestSafe"

    def dispatch(self, orders, vehicles, cost_matrix, pickup_times):
        # cost_matrix 里已经把不可行配对设为 BIG_M
        # 所以直接用 Nearest 的逻辑就行
        return NearestDispatcher().dispatch(
            orders, vehicles, cost_matrix, pickup_times
        )


class GradeOnlyDispatcher:
    """
    Grade 分层但不用连续难度修正：
    Layer 1: G4 → HV only
    Layer 2: G1 → AV preferred
    Layer 3: G2/G3 → 剩余

    和 TieredDispatcher 的区别：成本矩阵只用接驾时间，不加难度/风险惩罚。
    """

    def __init__(self):
        self.name = "GradeOnly"

    def dispatch(self, orders, vehicles, cost_matrix, pickup_times):
        n_o = len(orders)
        n_v = len(vehicles)

        if n_o == 0 or n_v == 0:
            return [], list(range(n_o))

        # 用 pickup_times 替代 cost_matrix（忽略难度修正）
        # 但保留不可行约束
        grade_cost = np.where(
            cost_matrix < BIG_M * 0.5,
            pickup_times,
            BIG_M
        )

        # 复用 TieredDispatcher 的分层逻辑
        from core.dispatcher import TieredDispatcher
        td = TieredDispatcher(name="GradeOnly")
        return td.dispatch(orders, vehicles, grade_cost, pickup_times)


class GlobalMatchDispatcher:
    """
    全局最优匹配（不分层）：
    直接对整个成本矩阵做匈牙利匹配。
    作为"理论上界"参考。
    """

    def __init__(self):
        self.name = "GlobalMatch"

    def dispatch(self, orders, vehicles, cost_matrix, pickup_times):
        n_o = len(orders)
        n_v = len(vehicles)

        if n_o == 0 or n_v == 0:
            return [], list(range(n_o))

        max_dim = max(n_o, n_v)
        padded = np.full((max_dim, max_dim), BIG_M, dtype=np.float64)
        padded[:n_o, :n_v] = cost_matrix

        row_ind, col_ind = linear_sum_assignment(padded)

        matched = []
        for r, c in zip(row_ind, col_ind):
            if r < n_o and c < n_v:
                if cost_matrix[r, c] < BIG_M * 0.5:
                    matched.append((r, c, pickup_times[r, c]))

        matched_ois = {m[0] for m in matched}
        unmatched = [i for i in range(n_o) if i not in matched_ois]

        return matched, unmatched