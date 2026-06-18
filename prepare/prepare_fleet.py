"""
prepare/prepare_fleet.py — 基于订单链重建忙闲状态

核心改进:
  不再用"附近有订单就算活跃"
  而是精确重建每个司机的 busy/idle 段
  每个窗口只统计处于 idle 段的司机
"""
import os
import sys
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import *


def prepare_fleet(day=DAY):
    print(f"[prepare_fleet] Loading day {day} ...")

    orders_path = os.path.join(DISPATCH_DATA_DIR, f"orders_day{day}.pkl")
    if not os.path.exists(orders_path):
        raise FileNotFoundError(f"{orders_path} not found.")

    orders = pd.read_pickle(orders_path)
    orders = orders.sort_values(["driver_id", "arrival_time_sec"]).reset_index(drop=True)

    # ============================================================
    # 1. 每个 driver 的订单链 → busy/idle 段
    # ============================================================
    print("  Building driver busy/idle segments ...")

    n_windows = SECONDS_PER_DAY // DISPATCH_INTERVAL

    # 每个窗口的 idle 司机集合和位置
    window_idle_drivers = defaultdict(dict)  # wid → {driver_id: location_link}

    hv_schedules = {}
    driver_groups = orders.groupby("driver_id")

    for did, grp in driver_groups:
        did = int(did)
        grp = grp.sort_values("arrival_time_sec")

        times = grp["arrival_time_sec"].values.astype(float)
        atas = grp["ata"].values.astype(float)
        origins = grp["origin_link"].values
        dests = grp["dest_link"].values

        first_start = times[0]
        last_end = times[-1] + atas[-1]

        hv_schedules[did] = {
            "entry_sec": float(first_start),
            "exit_sec": float(last_end),
            "first_link": int(origins[0]) if pd.notna(origins[0]) else 0,
            "last_dest_link": int(dests[-1]) if pd.notna(dests[-1]) else 0,
            "n_orders": len(grp),
        }

        # 构建 idle 段
        pre_idle_start = max(first_start - 300, 0)
        idle_segments = []

        # 第一单前
        if pre_idle_start < first_start:
            idle_segments.append((
                pre_idle_start, first_start,
                int(origins[0]) if pd.notna(origins[0]) else 0
            ))

        # 两单之间的 idle
        for k in range(len(times)):
            busy_end = times[k] + atas[k]
            if k < len(times) - 1:
                next_start = times[k + 1]
                gap = next_start - busy_end
                if gap > 3600:
                    continue
                if gap > 0:
                    loc = int(dests[k]) if pd.notna(dests[k]) else 0
                    idle_segments.append((busy_end, next_start, loc))

        # 最后一单后的 idle（保留 10 分钟）
        final_end = times[-1] + atas[-1]
        final_loc = int(dests[-1]) if pd.notna(dests[-1]) else 0
        idle_segments.append((final_end, min(final_end + 600, 86400), final_loc))

        # 把 idle 段映射到窗口
        for idle_start, idle_end, loc in idle_segments:
            w_start = int(idle_start // DISPATCH_INTERVAL)
            w_end = int(idle_end // DISPATCH_INTERVAL)
            for wid in range(max(w_start, 0), min(w_end + 1, n_windows)):
                window_idle_drivers[wid][did] = loc

    # 转成标准格式
    window_active_drivers = {
        wid: list(drivers.keys())
        for wid, drivers in window_idle_drivers.items()
    }
    window_driver_locations = dict(window_idle_drivers)

    # 统计
    idle_counts = [len(window_active_drivers.get(w, [])) for w in range(n_windows)]
    print(f"  Idle drivers/window: "
          f"mean={np.mean(idle_counts):.0f} "
          f"max={max(idle_counts)} "
          f"min={min(idle_counts)}")

    # ============================================================
    # 2. AV 数量
    # ============================================================
    peak_windows = [
        w for w in range(n_windows)
        if (7 * 3600 <= w * DISPATCH_INTERVAL < 11 * 3600) or
           (17 * 3600 <= w * DISPATCH_INTERVAL < 19 * 3600)
    ]

    if peak_windows:
        peak_idle = np.mean([len(window_active_drivers.get(w, [])) for w in peak_windows])
    else:
        peak_idle = np.mean(idle_counts)

    n_av = max(1, int(peak_idle * AV_RATIO))
    av_ids = list(range(-1, -n_av - 1, -1))

    # ============================================================
    # 3. 区域需求密度（独立加载 link_to_zone，避免依赖订单列）
    # ============================================================
    ltz_path = os.path.join(DISPATCH_DATA_DIR, "link_to_zone.pkl")
    if os.path.exists(ltz_path):
        link_to_zone = pd.read_pickle(ltz_path)
        # 给 orders 临时添加 zone 列（不保存回文件）
        orders_copy = pd.read_pickle(orders_path)
        orders_copy["zone"] = orders_copy["origin_link"].map(link_to_zone).fillna(0).astype(int)
        zone_demand = orders_copy.groupby("zone")["order_id"].count().to_dict()
        print(f"[prepare_fleet] Zone demand computed: {len(zone_demand)} zones")
    else:
        print("[prepare_fleet] Warning: link_to_zone.pkl not found, zone_demand will be empty.")
        zone_demand = {}

    # ============================================================
    # 4. 组装结果（一次性定义）
    # ============================================================
    result = {
        "hv_schedules": hv_schedules,
        "av_ids": av_ids,
        "n_hv": len(hv_schedules),
        "n_av": n_av,
        "summary": {
            "total_hv": len(hv_schedules),
            "total_av": n_av,
            "avg_idle_per_window": float(np.mean(idle_counts)),
            "peak_idle": float(peak_idle),
        },
        "zone_demand": zone_demand,   # ★ 关键字段
    }

    # ============================================================
    # 5. 保存
    # ============================================================
    fleet_path = os.path.join(DISPATCH_DATA_DIR, f"fleet_schedule_day{day}.pkl")
    pd.to_pickle(result, fleet_path)
    print(f"  Saved: {fleet_path}")

    wad_path = os.path.join(DISPATCH_DATA_DIR, f"window_active_drivers_day{day}.pkl")
    pd.to_pickle({
        "active_drivers": window_active_drivers,
        "driver_locations": window_driver_locations,
    }, wad_path)
    print(f"  Saved: {wad_path}")

    print(f"  HV: {len(hv_schedules):,} | AV: {n_av:,}")
    print(f"  Peak idle HV: {peak_idle:.0f}")
    if zone_demand:
        print(f"  Zone demand: {len(zone_demand)} zones with positive orders")
    else:
        print("  Zone demand: (empty)")

    return result


if __name__ == "__main__":
    prepare_fleet()