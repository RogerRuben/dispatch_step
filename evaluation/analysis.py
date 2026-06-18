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
    zone_analysis(results_dir)
    # ★ 可选：AV 比例对比（若存在多 AV 比例结果）
    compare_av_ratios(results_dir)


def zone_analysis(results_dir=RESULTS_DIR, grid_cols=10):
    """
    区域级别分析：计算每个 Zone 的多维指标，并分配网格坐标。
    输出 CSV 供可视化使用。
    """
    order_logs = load_all_order_logs(results_dir)
    if not order_logs:
        print("[zone_analysis] No order logs found.")
        return None

    all_zone_stats = []
    for name, df in order_logs.items():
        if 'zone' not in df.columns:
            continue
        matched = df[df["matched"]]
        if len(matched) == 0:
            continue

        grouped = matched.groupby("zone").agg(
            total_orders=("order_id", "count"),
            av_assigned=("vehicle_type", lambda x: (x == "AV").sum()),
            hv_assigned=("vehicle_type", lambda x: (x == "HV").sum()),
            avg_pickup=("pickup_time", "mean"),
            avg_total_cost=("total_cost", "mean"),
            avg_grade=("grade_num", "mean"),
            avg_difficulty=("difficulty", "mean"),
            avg_risk=("pred_risk_prob", "mean"),
            match_rate=("matched", "mean"),  # 该区域匹配率
        ).reset_index()

        grouped["av_ratio"] = grouped["av_assigned"] / grouped["total_orders"]
        grouped["strategy"] = name
        all_zone_stats.append(grouped)

    if not all_zone_stats:
        return None

    combined = pd.concat(all_zone_stats, ignore_index=True)

    # 为每个 Zone 分配网格坐标（基于 zone_id 排序后均匀分布）
    unique_zones = sorted(combined["zone"].unique())
    n_zones = len(unique_zones)
    n_cols = grid_cols
    n_rows = (n_zones + n_cols - 1) // n_cols
    zone_to_coord = {}
    for i, z in enumerate(unique_zones):
        row = i // n_cols
        col = i % n_cols
        zone_to_coord[z] = (col, row)  # x=col, y=row

    # 添加坐标列
    combined["x"] = combined["zone"].map(lambda z: zone_to_coord[z][0])
    combined["y"] = combined["zone"].map(lambda z: zone_to_coord[z][1])

    out_path = os.path.join(results_dir, "zone_stats.csv")
    combined.to_csv(out_path, index=False)
    print(f"[zone_analysis] Saved to {out_path}")
    return combined



def compare_av_ratios(results_dir=RESULTS_DIR):
    """
    对比不同 AV 比例（0%, 10%, 20%）的实验结果。
    要求结果文件命名包含 _AV0, _AV10, _AV20 后缀。
    """
    summaries = load_all_summaries(results_dir)
    rows = []
    for name, s in summaries.items():
        if "AV" not in name:
            continue
        # 解析 AV 比例
        try:
            av_pct = int(name.split("AV")[-1])
        except:
            av_pct = -1
        rows.append({
            "strategy": name,
            "av_pct": av_pct,
            "match_rate": s.get("match_rate", 0),
            "avg_pickup": s.get("avg_pickup_time", 0),
            "av_utilization": s.get("av_utilization", 0),
            "av_order_share": s.get("av_order_share", 0),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(results_dir, "av_ratio_comparison.csv"), index=False)
    print("[compare_av_ratios] Saved comparison.")
    return df