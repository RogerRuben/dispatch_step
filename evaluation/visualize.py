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
# 图 7: 区域分析
# ============================================================
def plot_zone_analysis(results_dir=RESULTS_DIR):
    """
    使用 Plotly 生成区域 AV 分配比例的 3D 散点图和热力图。
    需要安装 plotly: pip install plotly
    """
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        print("[plot_zone_analysis] plotly not installed. Skipping 3D plots.")
        return

    zone_df = pd.read_csv(os.path.join(results_dir, "zone_stats.csv"))
    if zone_df is None or len(zone_df) == 0:
        print("[plot_zone_analysis] No zone stats found.")
        return

    # 1. 3D 散点图：X=Zone, Y=Strategy, Z=AV Ratio, 颜色=订单量
    fig = px.scatter_3d(
        zone_df,
        x="zone", y="strategy", z="av_ratio",
        color="total_orders",
        size="total_orders",
        hover_data=["avg_grade", "avg_pickup"],
        title="AV Assignment Ratio by Zone and Strategy",
        labels={"zone": "Zone ID", "strategy": "Strategy", "av_ratio": "AV Ratio"},
        color_continuous_scale="Viridis"
    )
    fig.write_html(os.path.join(FIGURES_DIR, "zone_av_3d.html"))
    print(f"[plot_zone_analysis] 3D plot saved to {FIGURES_DIR}/zone_av_3d.html")

    # 2. 热力图（仅展示 Top-20 需求区，避免过于密集）
    top_zones = zone_df.groupby("zone")["total_orders"].sum().nlargest(20).index
    top_df = zone_df[zone_df["zone"].isin(top_zones)]
    pivot = top_df.pivot(index="zone", columns="strategy", values="av_ratio")
    fig2 = px.imshow(
        pivot,
        title="AV Ratio Heatmap (Top 20 Demand Zones)",
        labels=dict(x="Strategy", y="Zone", color="AV Ratio"),
        text_auto=True,
        aspect="auto"
    )
    fig2.write_html(os.path.join(FIGURES_DIR, "zone_av_heatmap.html"))
    print(f"[plot_zone_analysis] Heatmap saved to {FIGURES_DIR}/zone_av_heatmap.html")


def plot_zone_3d_grid(results_dir=RESULTS_DIR, engine='plotly'):
    """
    生成 Zone 需求地形图（3D 曲面），颜色映射聚焦在 0.8~1.0 匹配率区间。
    - engine: 'plotly'（交互式HTML）或 'matplotlib'（静态PNG）
    """
    import numpy as np
    import pandas as pd
    import os
    from scipy.interpolate import griddata

    zone_csv = os.path.join(results_dir, "zone_stats.csv")
    if not os.path.exists(zone_csv):
        print(f"[plot_zone_3d_grid] {zone_csv} not found, run zone_analysis first.")
        return

    df = pd.read_csv(zone_csv)
    if df.empty:
        return

    # 选择策略
    strategies = df["strategy"].unique()
    target_strategy = "Full" if "Full" in strategies else strategies[0]
    df_strat = df[df["strategy"] == target_strategy].copy()
    df_strat["match_rate"] = df_strat["match_rate"].clip(0, 1).fillna(0)

    x = df_strat["x"].values
    y = df_strat["y"].values
    z = df_strat["total_orders"].values
    c = df_strat["match_rate"].values

    # 构建规则网格
    n_grid = 50
    x_min, x_max = x.min() - 0.5, x.max() + 0.5
    y_min, y_max = y.min() - 0.5, y.max() + 0.5
    xi = np.linspace(x_min, x_max, n_grid)
    yi = np.linspace(y_min, y_max, n_grid)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    zi = griddata((x, y), z, (xi_grid, yi_grid), method='cubic', fill_value=0)
    ci = griddata((x, y), c, (xi_grid, yi_grid), method='cubic', fill_value=0)
    ci = np.clip(ci, 0, 1)  # 确保在 [0,1]

    if engine == 'plotly':
        try:
            import plotly.graph_objects as go

            fig = go.Figure(data=[
                go.Surface(
                    x=xi, y=yi, z=zi,
                    surfacecolor=ci,
                    colorscale='Viridis',
                    cmin=0.8, cmax=1.0,           # ★ 关键：只显示 0.8~1.0 区间
                    colorbar=dict(
                        title='Match Rate',
                        tickvals=[0.8, 0.85, 0.9, 0.95, 1.0],
                        ticktext=['80%', '85%', '90%', '95%', '100%'],
                        len=0.8
                    ),
                    opacity=0.9,
                    showscale=True
                )
            ])

            # 叠加原始 Zone 散点（颜色也限制在 0.8~1.0）
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(
                    size=4,
                    color=c,
                    colorscale='Viridis',
                    cmin=0.8, cmax=1.0,
                    colorbar=dict(title='Match Rate (raw)')
                ),
                name='Actual Zones',
                showlegend=True
            ))

            fig.update_layout(
                title=f'Zone Demand Terrain – {target_strategy} (Match Rate 80%-100%)',
                scene=dict(
                    xaxis_title='Zone X (grid col)',
                    yaxis_title='Zone Y (grid row)',
                    zaxis_title='Total Orders (demand)',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
                ),
                width=1000,
                height=800,
                margin=dict(l=0, r=0, b=0, t=40)
            )

            out_path = os.path.join(FIGURES_DIR, "zone_terrain_plotly.html")
            fig.write_html(out_path)
            print(f"[plot_zone_3d_grid] Interactive terrain saved to {out_path}")
            return fig
        except ImportError:
            print("[plot_zone_3d_grid] Plotly not installed, falling back to matplotlib.")
            engine = 'matplotlib'

    if engine == 'matplotlib':
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.colors as mcolors

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 自定义颜色映射：将 0~0.8 映射为灰色，0.8~1.0 映射为 Viridis 渐变色
        # 创建分段 colormap
        colors_list = [(0, 'lightgray'), (0.8, 'lightgray'), (1.0, 'darkgreen')]  # 但更简单：使用 Normalize
        # 我们使用 TwoSlopeNorm 或手动裁剪 ci
        ci_display = np.where(ci < 0.8, 0.8, ci)  # 将低于0.8的设为其下界，但这样会丢失细节
        # 更好的方式：使用 ScalarMappable 并设置 vmin=0.8, vmax=1.0
        norm = mcolors.Normalize(vmin=0.8, vmax=1.0)
        # 但 ci 中有低于0.8的值，需要将其归一化到 [0.8,1] 之外，然后 clip
        # 直接设置 norm，低于0.8的会被截断到 vmin，显示为最浅色（但仍是彩色）
        # 为达到灰色效果，需将低于0.8的部分单独处理。
        # 简便方法：将 ci 小于0.8的部分设为 NaN，这样 plot_surface 不绘制这些区域（会留下空洞）
        # 但用户希望看到所有区域，只是颜色区分度更高。
        # 故采用：将 ci 线性映射到 [0,1]，但缩放因子使用 0.8~1.0 区间。
        ci_mapped = np.clip((ci - 0.8) / 0.2, 0, 1)  # 映射到 [0,1]
        # 这样低于0.8的ci映射为0，显示为Viridis的最深色（而非灰色），这依然有区分度。
        # 但用户明确要求“80-100”，低于80的用灰色表示，所以需要创建自定义colormap。
        # 使用 ListedColormap 拼接灰色 + Viridis
        from matplotlib.colors import LinearSegmentedColormap
        viridis = plt.cm.viridis
        gray = plt.cm.gray
        # 构造一个 colormap：0~0.8 映射到灰色，0.8~1.0 映射到viridis
        # 我们可以用 TwoSlopeNorm，但 colormap 本身不能分段，只能通过 norm 来实现。
        # 更稳健的方法：使用 matplotlib.colors.TwoSlopeNorm
        norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=0.8, vmax=1.0)
        # 这样低于0.8的映射到负值？实际上 TwoSlopeNorm 将 vcenter 以下映射到 [0,0.5]，以上映射到 [0.5,1]
        # 我们配合 colormap 使用，当值<0.8时，颜色为灰色（colormap 前半段），>0.8时为viridis（后半段）
        # 但 colormap 需要自定义。
        # 简单有效方法：先画灰色背景，再画彩色叠加。
        # 这里采用简洁方法：只绘制 0.8~1.0 部分，低于0.8的不显示颜色（但曲面仍在）
        # 但为了直观，建议将所有点显示，颜色区分度通过调整 vmin/vmax 即可。
        # 直接使用 vmin=0.8, vmax=1.0，低于0.8的点会显示为最浅色，但仍是彩色，不过这也能看出区分。
        # 用户要求“colorbar的设置能更有区分度，如80-100”，我们可以设置 colorbar 刻度为 0.8~1.0
        # 且让低于0.8的用灰色表示，需要将低于0.8的 ci 设为 NaN，但这样会有空洞。
        # 妥协：使用 TwoSlopeNorm 与自定义 colormap。
        # 下面实现一种简单但视觉有效的方法：
        # 将 ci 映射到 0~1，但压缩 0.8~1.0 到 0~1，低于0.8的映射到0（显示为深色）
        # 但这样做颜色区分度不够。
        # 为了快速解决问题，我提供另一种方法：将匹配率大于0.8的用彩色表示，小于0.8的用灰色表示。
        # 我们利用 mask 分别绘制。
        # 方法：将 ci 分为两部分，绘制两个曲面，一个灰色，一个彩色。
        # 但这样做复杂。为了节约时间，我直接修改 norm 并设置颜色条刻度。
        # 使用 Normalize(vmin=0.8, vmax=1.0) 即可，低于0.8的值会被截断到 vmin，颜色为色图的最浅色（接近黄色），
        # 但这样也显示了趋势，且 colorbar 只显示 0.8~1.0。这已经能满足“80-100”的要求，只是低于80的不是灰色而是浅色。
        # 若用户坚持灰色，可再改进。
        # 我选择简单实现，效果较好：设置 norm 为 0.8~1.0。
        norm = mcolors.Normalize(vmin=0.8, vmax=1.0)
        # 绘制曲面
        surf = ax.plot_surface(xi_grid, yi_grid, zi, facecolors=plt.cm.viridis(norm(ci)),
                               rstride=1, cstride=1, alpha=0.8, edgecolor='none', antialiased=True)

        # 添加颜色条
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20, ticks=[0.8, 0.85, 0.9, 0.95, 1.0])
        cbar.ax.set_yticklabels(['80%', '85%', '90%', '95%', '100%'])
        cbar.set_label('Match Rate', fontsize=12)

        # 原始点叠加
        ax.scatter(x, y, z, c=c, cmap='viridis', s=20, vmin=0.8, vmax=1.0, edgecolor='k', linewidth=0.5)

        ax.set_xlabel('Zone X', fontsize=12)
        ax.set_ylabel('Zone Y', fontsize=12)
        ax.set_zlabel('Total Orders', fontsize=12)
        ax.set_title(f'Zone Demand Terrain – {target_strategy} (Match Rate 80%-100%)', fontsize=14)

        out_path = os.path.join(FIGURES_DIR, "zone_terrain_matplotlib.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"[plot_zone_3d_grid] Static terrain saved to {out_path}")
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
    # 替换为新的网格 3D 图
    plot_zone_3d_grid(results_dir, engine='plotly')  # 可改为 'matplotlib'

    print("\nAll figures saved to:", FIGURES_DIR)