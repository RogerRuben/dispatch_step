"""
evaluation/analysis.py — 指标计算 + 策略对比

功能:
  1. 从 recorder 计算单策略指标
  2. 多策略对比表
  3. 敏感性分析汇总
"""
import os
import pandas as pd
import numpy as np
from core.config import RESULTS_DIR


def load_all_summaries(results_dir=RESULTS_DIR) -> dict:
    """加载 results/ 下所有策略的 summary"""
    summaries = {}
    for f in sorted(os.listdir(results_dir)):
        if f.endswith("_summary.pkl"):
            name = f.replace("_summary.pkl", "")
            summaries[name] = pd.read_pickle(os.path.join(results_dir, f))
    return summaries


def load_all_window_logs(results_dir=RESULTS_DIR) -> dict:
    """加载 results/ 下所有策略的 window_log"""
    logs = {}
    for f in sorted(os.listdir(results_dir)):
        if f.endswith("_windows.pkl"):
            name = f.replace("_windows.pkl", "")
            logs[name] = pd.read_pickle(os.path.join(results_dir, f))
    return logs


def load_all_order_logs(results_dir=RESULTS_DIR) -> dict:
    """加载 results/ 下所有策略的 order_log"""
    logs = {}
    for f in sorted(os.listdir(results_dir)):
        if f.endswith("_orders.pkl"):
            name = f.replace("_orders.pkl", "")
            logs[name] = pd.read_pickle(os.path.join(results_dir, f))
    return logs


def compare_strategies(results_dir=RESULTS_DIR) -> pd.DataFrame:
    """
    多策略核心指标对比表。
    返回 DataFrame，每行一个策略。
    """
    summaries = load_all_summaries(results_dir)

    if not summaries:
        print("[analysis] No results found.")
        return pd.DataFrame()

    rows = []
    for name, s in summaries.items():
        rows.append({
            "Strategy": name,
            "Match Rate": s.get("match_rate", 0),
            "Avg Pickup (s)": s.get("avg_pickup_time", 0),
            "Total Cost": s.get("total_cost", 0),
            "AV Utilization": s.get("av_utilization", 0),
            "AV High-Risk Rate": s.get("av_high_risk_rate", 0),
            "AV Extreme-Risk Rate": s.get("av_extreme_risk_rate", 0),
            "Cross-Zone Rate": s.get("cross_zone_rate", 0),
            "Total Matched": s.get("total_matched", 0),
            "Total Unmatched": s.get("total_unmatched", 0),
        })

    df = pd.DataFrame(rows).set_index("Strategy")

    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)
    print(df.to_string())
    print()

    return df


def grade_distribution_analysis(results_dir=RESULTS_DIR) -> pd.DataFrame:
    """
    分析每个策略下，各 Grade 的 AV/HV 分配比例。
    """
    order_logs = load_all_order_logs(results_dir)

    if not order_logs:
        return pd.DataFrame()

    rows = []
    for name, ol in order_logs.items():
        matched = ol[ol["matched"]]
        if len(matched) == 0:
            continue

        for grade_num in [1, 2, 3, 4]:
            grade_orders = matched[matched["grade_num"] == grade_num]
            n_total = len(grade_orders)
            n_av = int((grade_orders["vehicle_type"] == "AV").sum())
            n_hv = int((grade_orders["vehicle_type"] == "HV").sum())

            rows.append({
                "Strategy": name,
                "Grade": f"G{grade_num}",
                "Total": n_total,
                "AV": n_av,
                "HV": n_hv,
                "AV_Ratio": n_av / max(n_total, 1),
            })

    df = pd.DataFrame(rows)

    if len(df) > 0:
        print("\n" + "=" * 80)
        print("GRADE-LEVEL AV/HV DISTRIBUTION")
        print("=" * 80)
        pivot = df.pivot_table(
            index="Strategy", columns="Grade",
            values="AV_Ratio", aggfunc="first"
        )
        print(pivot.round(4).to_string())
        print()

    return df


def temporal_summary(results_dir=RESULTS_DIR) -> dict:
    """
    时序汇总：高峰/低谷时段的指标差异。
    """
    window_logs = load_all_window_logs(results_dir)
    result = {}

    for name, wl in window_logs.items():
        if len(wl) == 0:
            continue

        # 高峰时段：7-9, 17-19
        peak_mask = (
            ((wl["hour"] >= 7) & (wl["hour"] < 9)) |
            ((wl["hour"] >= 17) & (wl["hour"] < 19))
        )
        off_mask = ~peak_mask & (wl["n_orders"] > 0)

        peak = wl[peak_mask]
        off = wl[off_mask]

        result[name] = {
            "peak_match_rate": float(peak["match_rate"].mean()) if len(peak) > 0 else 0,
            "peak_avg_pickup": float(peak["avg_pickup_time"].mean()) if len(peak) > 0 else 0,
            "off_match_rate": float(off["match_rate"].mean()) if len(off) > 0 else 0,
            "off_avg_pickup": float(off["avg_pickup_time"].mean()) if len(off) > 0 else 0,
        }

    if result:
        print("\n" + "=" * 80)
        print("PEAK vs OFF-PEAK COMPARISON")
        print("=" * 80)
        df = pd.DataFrame(result).T
        print(df.round(4).to_string())
        print()

    return result


def full_analysis(results_dir=RESULTS_DIR):
    """运行所有分析"""
    compare_strategies(results_dir)
    grade_distribution_analysis(results_dir)
    temporal_summary(results_dir)