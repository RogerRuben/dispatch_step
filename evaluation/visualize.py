"""
evaluation/visualize.py — 图表生成

图表:
  1. 策略核心指标对比（分组柱状图）
  2. 时序动态（匹配率 + 接驾时间）
  3. AV 风险暴露对比
  4. 供需时序
  5. Grade 级 AV/HV 分配比例
  6. 敏感性分析（折线图）
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from core.config import RESULTS_DIR, FIGURES_DIR
from evaluation.analysis import (
    load_all_summaries, load_all_window_logs,
    load_all_order_logs, compare_strategies,
    grade_distribution_analysis,
)

# 全局样式
plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

# 策略颜色
STRATEGY_COLORS = {
    "Random": "#999999",
    "Nearest": "#4DBEEE",
    "NearestSafe": "#77AC30",
    "GradeOnly": "#EDB120",
    "GlobalMatch": "#7E2F8E",
    "TieredDispatcher": "#D95319",
    "Full": "#D95319",
}


def _get_color(name):
    for key, color in STRATEGY_COLORS.items():
        if key.lower() in name.lower():
            return color
    return "#0072BD"


def _save_fig(fig, filename):
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================
# 图 1: 策略核心指标对比
# ============================================================

def plot_strategy_comparison(results_dir=RESULTS_DIR):
    """各策略的核心指标对比（分组柱状图）"""
    summaries = load_all_summaries(results_dir)
    if not summaries:
        return

    names = list(summaries.keys())
    colors = [_get_color(n) for n in names]

    metrics = [
        ("match_rate", "Match Rate ↑"),
        ("avg_pickup_time", "Avg Pickup Time (s) ↓"),
        ("av_utilization", "AV Utilization ↑"),
        ("av_high_risk_rate", "AV High-Risk Rate ↓"),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))

    for idx, (metric, title) in enumerate(metrics):
        values = [summaries[n].get(metric, 0) for n in names]
        axes[idx].bar(range(len(names)), values, color=colors)
        axes[idx].set_title(title, fontsize=11)
        axes[idx].set_xticks(range(len(names)))
        axes[idx].set_xticklabels(names, rotation=45, ha="right", fontsize=9)

    fig.suptitle("Strategy Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    _save_fig(fig, "strategy_comparison.pdf")


# ============================================================
# 图 2: 时序动态
# ============================================================

def plot_temporal_dynamics(results_dir=RESULTS_DIR):
    """逐窗口的匹配率和接驾时间"""
    window_logs = load_all_window_logs(results_dir)
    if not window_logs:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    for name, wl in window_logs.items():
        if len(wl) == 0:
            continue
        color = _get_color(name)
        hours = wl["hour"]

        ax1.plot(hours, wl["match_rate"], label=name, alpha=0.7, color=color)
        ax2.plot(hours, wl["avg_pickup_time"], label=name, alpha=0.7, color=color)

    ax1.set_ylabel("Match Rate")
    ax1.set_title("Match Rate over Time")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.set_ylim(0, 1.05)

    ax2.set_ylabel("Avg Pickup Time (s)")
    ax2.set_xlabel("Hour of Day")
    ax2.set_title("Average Pickup Time over Time")
    ax2.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    _save_fig(fig, "temporal_dynamics.pdf")


# ============================================================
# 图 3: AV 风险暴露对比
# ============================================================

def plot_av_risk_comparison(results_dir=RESULTS_DIR):
    """各策略下 AV 执行高风险订单的比例"""
    summaries = load_all_summaries(results_dir)
    if not summaries:
        return

    names = list(summaries.keys())
    high_risk = [summaries[n].get("av_high_risk_rate", 0) for n in names]
    extreme = [summaries[n].get("av_extreme_risk_rate", 0) for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, high_risk, width, label="High Risk (G3+G4)", color="orange")
    ax.bar(x + width / 2, extreme, width, label="Extreme Risk (q3>τ)", color="red")

    ax.set_ylabel("Rate")
    ax.set_title("AV Risk Exposure by Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.legend()

    plt.tight_layout()
    _save_fig(fig, "av_risk_comparison.pdf")


# ============================================================
# 图 4: 供需时序
# ============================================================

def plot_supply_demand(results_dir=RESULTS_DIR):
    """订单到达量 vs 可用车辆量"""
    window_logs = load_all_window_logs(results_dir)
    if not window_logs:
        return

    # 用第一个策略的数据（供需对所有策略一样）
    name = list(window_logs.keys())[0]
    wl = window_logs[name]

    if len(wl) == 0:
        return

    hours = wl["hour"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    ax1.fill_between(hours, wl["n_orders"], alpha=0.6, color="steelblue", label="Orders")
    ax1.set_ylabel("Order Count")
    ax1.set_title("Order Arrivals over Time")
    ax1.legend()

    ax2.fill_between(hours, wl["n_hv_available"], alpha=0.5, color="green", label="HV Available")
    hv_top = wl["n_hv_available"]
    av_top = hv_top + wl["n_av_available"]
    ax2.fill_between(hours, hv_top, av_top, alpha=0.5, color="purple", label="AV Available")
    ax2.set_ylabel("Vehicle Count")
    ax2.set_xlabel("Hour of Day")
    ax2.set_title("Fleet Availability over Time")
    ax2.legend()

    plt.tight_layout()
    _save_fig(fig, "supply_demand_temporal.pdf")


# ============================================================
# 图 5: Grade 级 AV/HV 分配
# ============================================================

def plot_grade_av_distribution(results_dir=RESULTS_DIR):
    """各策略下各 Grade 的 AV 分配比例"""
    df = grade_distribution_analysis(results_dir)
    if len(df) == 0:
        return

    strategies = df["Strategy"].unique()
    grades = ["G1", "G2", "G3", "G4"]
    x = np.arange(len(grades))
    width = 0.8 / len(strategies)

    fig, ax = plt.subplots(figsize=(10, 5))

    for idx, strat in enumerate(strategies):
        sub = df[df["Strategy"] == strat]
        values = []
        for g in grades:
            row = sub[sub["Grade"] == g]
            values.append(float(row["AV_Ratio"].values[0]) if len(row) > 0 else 0)

        offset = (idx - len(strategies) / 2 + 0.5) * width
        ax.bar(x + offset, values, width, label=strat, color=_get_color(strat))

    ax.set_xlabel("Grade")
    ax.set_ylabel("AV Assignment Ratio")
    ax.set_title("AV Assignment Ratio by Grade and Strategy")
    ax.set_xticks(x)
    ax.set_xticklabels(grades)
    ax.legend(fontsize=9)

    plt.tight_layout()
    _save_fig(fig, "grade_av_distribution.pdf")


# ============================================================
# 图 6: 敏感性分析
# ============================================================

def plot_sensitivity(param_name: str, param_values: list,
                     results: list, metrics=None):
    """
    敏感性分析折线图。

    Parameters
    ----------
    param_name : 参数名（如 "AV Ratio"）
    param_values : 参数值列表
    results : list of dict（每个值对应一个 summary dict）
    metrics : 要画的指标列表
    """
    if metrics is None:
        metrics = ["match_rate", "avg_pickup_time", "av_utilization", "av_high_risk_rate"]

    titles = {
        "match_rate": "Match Rate ↑",
        "avg_pickup_time": "Avg Pickup Time (s) ↓",
        "av_utilization": "AV Utilization ↑",
        "av_high_risk_rate": "AV High-Risk Rate ↓",
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))

    for idx, metric in enumerate(metrics):
        values = [r.get(metric, 0) for r in results]
        axes[idx].plot(param_values, values, "o-", color="#0072BD", linewidth=2)
        axes[idx].set_xlabel(param_name)
        axes[idx].set_title(titles.get(metric, metric))
        axes[idx].grid(True, alpha=0.3)

    fig.suptitle(f"Sensitivity Analysis: {param_name}", fontsize=14, y=1.02)
    plt.tight_layout()

    safe_name = param_name.lower().replace(" ", "_")
    _save_fig(fig, f"sensitivity_{safe_name}.pdf")


# ============================================================
# 一键生成所有图
# ============================================================

def plot_all(results_dir=RESULTS_DIR):
    """生成所有图表"""
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    plot_strategy_comparison(results_dir)
    plot_temporal_dynamics(results_dir)
    plot_av_risk_comparison(results_dir)
    plot_supply_demand(results_dir)
    plot_grade_av_distribution(results_dir)

    print("\nAll figures saved to:", FIGURES_DIR)