"""
simulation/recorder.py — 仿真结果记录器

记录两个层级的数据：
  1. 窗口级（window_log）: 每个调度窗口的汇总统计
  2. 订单级（order_log）: 每个订单的匹配结果
"""
import os
import numpy as np
import pandas as pd
from core.config import RESULTS_DIR


class DispatchRecorder:
    """仿真结果记录器"""

    def __init__(self):
        self.window_records = []
        self.order_records = []

    # ============================================================
    # 记录接口
    # ============================================================

    def record_window(self, window_id: int, time_sec: float,
                      n_orders: int, n_matched: int,
                      n_available: int, n_av_available: int, n_hv_available: int,
                      n_av_assigned: int, n_hv_assigned: int,
                      n_cross_zone: int,
                      avg_pickup_time: float,
                      n_av_high_risk: int, n_av_extreme_risk: int,
                      fleet_stats: dict = None):
        """记录一个调度窗口的汇总"""
        rec = {
            "window_id": window_id,
            "time_sec": time_sec,
            "hour": time_sec / 3600,
            "n_orders": n_orders,
            "n_matched": n_matched,
            "n_unmatched": n_orders - n_matched,
            "match_rate": n_matched / max(n_orders, 1),
            "n_available": n_available,
            "n_av_available": n_av_available,
            "n_hv_available": n_hv_available,
            "n_av_assigned": n_av_assigned,
            "n_hv_assigned": n_hv_assigned,
            "n_cross_zone": n_cross_zone,
            "avg_pickup_time": avg_pickup_time,
            "n_av_high_risk": n_av_high_risk,
            "n_av_extreme_risk": n_av_extreme_risk,
        }

        if fleet_stats:
            for k, v in fleet_stats.items():
                if k not in rec:
                    rec[k] = v

        self.window_records.append(rec)

    def record_order(self, order_id: int, window_id: int,
                     grade: str, grade_num: int,
                     difficulty: float,
                     pred_risk_prob: float,
                     matched: bool,
                     vehicle_id: int = None,
                     vehicle_type: str = None,
                     pickup_time: float = 0.0,
                     total_cost: float = 0.0,
                     cross_zone: bool = False):
        """记录单个订单的匹配结果"""
        self.order_records.append({
            "order_id": order_id,
            "window_id": window_id,
            "grade": grade,
            "grade_num": grade_num,
            "difficulty": difficulty,
            "pred_risk_prob": pred_risk_prob,
            "matched": matched,
            "vehicle_id": vehicle_id,
            "vehicle_type": vehicle_type,
            "pickup_time": pickup_time,
            "total_cost": total_cost,
            "cross_zone": cross_zone,
        })

    # ============================================================
    # 数据访问
    # ============================================================

    def get_window_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.window_records)

    def get_order_log(self) -> pd.DataFrame:
        return pd.DataFrame(self.order_records)

    def get_summary(self) -> dict:
        """计算全局汇总指标"""
        ol = self.get_order_log()

        if len(ol) == 0:
            return {"strategy": "unknown", "total_orders": 0}

        total = len(ol)
        matched_mask = ol["matched"]
        matched = ol[matched_mask]

        n_matched = int(matched_mask.sum())
        n_av = int((matched["vehicle_type"] == "AV").sum()) if n_matched > 0 else 0
        n_hv = int((matched["vehicle_type"] == "HV").sum()) if n_matched > 0 else 0

        # AV 风险指标
        av_orders = matched[matched["vehicle_type"] == "AV"] if n_matched > 0 else pd.DataFrame()
        n_av_high_risk = int((av_orders["grade_num"] >= 3).sum()) if len(av_orders) > 0 else 0
        n_av_extreme = int((av_orders["pred_risk_prob"] > 0.3).sum()) if len(av_orders) > 0 else 0

        summary = {
            "total_orders": total,
            "total_matched": n_matched,
            "total_unmatched": total - n_matched,
            "match_rate": n_matched / max(total, 1),
            "avg_pickup_time": float(matched["pickup_time"].mean()) if n_matched > 0 else 0.0,
            "total_cost": float(matched["total_cost"].sum()) if n_matched > 0 else 0.0,
            "avg_cost": float(matched["total_cost"].mean()) if n_matched > 0 else 0.0,
            "n_av_assigned": n_av,
            "n_hv_assigned": n_hv,
            "av_utilization": n_av / max(n_av + n_hv, 1),
            "av_high_risk_rate": n_av_high_risk / max(n_av, 1),
            "av_extreme_risk_rate": n_av_extreme / max(n_av, 1),
            "cross_zone_rate": float(matched["cross_zone"].mean()) if n_matched > 0 else 0.0,
        }

        return summary

    # ============================================================
    # 保存
    # ============================================================

    def save(self, strategy_name: str, output_dir: str = RESULTS_DIR):
        """保存结果"""
        os.makedirs(output_dir, exist_ok=True)

        summary = self.get_summary()
        summary["strategy"] = strategy_name

        wl = self.get_window_log()
        ol = self.get_order_log()

        pd.to_pickle(summary, os.path.join(output_dir, f"{strategy_name}_summary.pkl"))
        wl.to_pickle(os.path.join(output_dir, f"{strategy_name}_windows.pkl"))
        ol.to_pickle(os.path.join(output_dir, f"{strategy_name}_orders.pkl"))

        print(f"[recorder] Saved {strategy_name}: "
              f"match_rate={summary['match_rate']:.4f} "
              f"avg_pickup={summary['avg_pickup_time']:.1f}s "
              f"av_util={summary['av_utilization']:.4f}")