"""
core/fleet.py — 车辆与车队状态管理

供给侧建模:
  1. Busy 状态：可观测（来自指派记录）
  2. Idle-online 状态：按目标供需比动态校准
  3. 每个窗口：eligible idle HV → 按 target 激活 → 进入匹配池
  4. AV 全天在线

关键改进:
  不再依赖"窗口级活跃司机统计"
  而是每个窗口根据当前订单数动态决定放多少 HV 进池
"""
import os
import random
import numpy as np
import pandas as pd
from collections import defaultdict

from core.config import (
    THETA_NEAR_FREE, DISPATCH_DATA_DIR, DISPATCH_INTERVAL,
    TARGET_SUPPLY_RATIOS, AV_ONLINE_RATIO, HV_MIN_ONLINE,
    SECONDS_PER_DAY,
)


class Vehicle:
    __slots__ = [
        "id", "vtype", "status",
        "location_link", "zone",
        "current_order_id",
        "busy_until", "release_link",
        "entry_sec", "exit_sec",
    ]

    def __init__(self, vid, vtype, location_link, zone,
                 entry_sec=0.0, exit_sec=86400.0):
        self.id = vid
        self.vtype = vtype
        self.status = "OFFLINE"
        self.location_link = location_link
        self.zone = zone
        self.current_order_id = None
        self.busy_until = 0.0
        self.release_link = location_link
        self.entry_sec = entry_sec
        self.exit_sec = exit_sec

    def is_available(self, current_time, theta=THETA_NEAR_FREE):
        if self.status == "IDLE":
            return True
        if self.status == "NEAR_FREE":
            return (self.busy_until - current_time) <= theta
        return False

    def is_av(self):
        return self.vtype == "AV"

    def in_work_period(self, current_time):
        """HV 是否在工作时段内"""
        return self.entry_sec <= current_time <= self.exit_sec


class FleetManager:
    """
    供给侧管理器。

    核心逻辑:
      每个窗口：
        1. 释放完成订单的车辆
        2. 找出所有 eligible idle HV（在工作时段内 + 不 busy）
        3. 按目标供需比从 eligible 中选取 N 辆上线
        4. AV 全天在线
        5. 组成当前窗口的可用池
    """

    def __init__(self, fleet_schedule, link_to_zone, day=None):
        self.link_to_zone = link_to_zone
        self.vehicles = {}
        self.total_hv = 0
        self.total_av = 0

        # 当前窗口需要的信息
        self._current_online_hv_ids = set()

        self._init_fleet(fleet_schedule)

    def _get_zone(self, link_id):
        return self.link_to_zone.get(int(link_id), 0) if link_id else 0

    def _init_fleet(self, fleet_schedule):
        # HV：全部初始化为 OFFLINE
        for driver_id, info in fleet_schedule["hv_schedules"].items():
            loc = int(info.get("first_link", 0) or 0)
            v = Vehicle(
                vid=int(driver_id),
                vtype="HV",
                location_link=loc,
                zone=self._get_zone(loc),
                entry_sec=float(info["entry_sec"]),
                exit_sec=float(info["exit_sec"]),
            )
            v.status = "OFFLINE"
            self.vehicles[v.id] = v
            self.total_hv += 1

        # AV：全天在线
        random.seed(42)
        hv_locs = [v.location_link for v in self.vehicles.values()
                    if v.location_link != 0]

        for av_id in fleet_schedule["av_ids"]:
            loc = random.choice(hv_locs) if hv_locs else 0
            v = Vehicle(
                vid=int(av_id),
                vtype="AV",
                location_link=loc,
                zone=self._get_zone(loc),
                entry_sec=0.0,
                exit_sec=86400.0,
            )
            v.status = "IDLE"
            self.vehicles[v.id] = v
            self.total_av += 1

        print(f"[FleetManager] Init: HV={self.total_hv:,} AV={self.total_av:,}")

    def _get_supply_ratio(self, current_time):
        """根据时段返回目标供需比"""
        hour = current_time / 3600.0

        # 夜间
        if hour < 6 or hour >= 22:
            return TARGET_SUPPLY_RATIOS["night"]

        # 高峰
        if (7 <= hour < 9) or (17 <= hour < 19):
            return TARGET_SUPPLY_RATIOS["peak"]

        # 平峰
        return TARGET_SUPPLY_RATIOS["normal"]

    def update(self, current_time, n_orders_this_window=0):
        """
        推进时间 + 增量式供给池管理。

        核心逻辑:
          1. 释放完成订单的车辆 → IDLE（位置更新）
          2. 即将完成 → NEAR_FREE
          3. 已经 IDLE 的 HV：如果仍在工作时段内 → 保持 IDLE（不踢下线）
          4. 已经 IDLE 的 HV：如果超出工作时段 → OFFLINE
          5. 如果当前 IDLE + NEAR_FREE 数量不够 → 从 OFFLINE eligible 中补充上线
        """
        # ============================================================
        # Step 1: 释放完成订单的车辆
        # ============================================================
        for v in self.vehicles.values():
            if v.status in ("BUSY", "NEAR_FREE"):
                if current_time >= v.busy_until:
                    v.status = "IDLE"
                    v.location_link = v.release_link
                    v.zone = self._get_zone(v.release_link)
                    v.current_order_id = None
                elif (v.busy_until - current_time) <= THETA_NEAR_FREE:
                    v.status = "NEAR_FREE"

        # ============================================================
        # Step 2: 已在线的 HV 状态维护
        # ============================================================
        # 已经 IDLE 的 HV：如果超出工作时段 → 下线
        # 已经 IDLE 的 HV：如果仍在工作时段 → 保持（不踢）
        for v in self.vehicles.values():
            if v.vtype != "HV":
                continue
            if v.status == "IDLE":
                if not v.in_work_period(current_time):
                    v.status = "OFFLINE"

        # ============================================================
        # Step 3: 统计当前供给
        # ============================================================
        n_idle_hv = 0
        n_busy_hv = 0
        n_near_free_hv = 0
        n_av_available = 0

        eligible_offline = []  # 可以被激活上线的 HV

        for v in self.vehicles.values():
            if v.is_av():
                if v.status in ("IDLE", "NEAR_FREE"):
                    n_av_available += 1
                continue

            # HV
            if v.status == "IDLE":
                n_idle_hv += 1
            elif v.status == "BUSY":
                n_busy_hv += 1
            elif v.status == "NEAR_FREE":
                n_near_free_hv += 1
            elif v.status == "OFFLINE":
                # 在工作时段内的 OFFLINE HV 可以被激活
                if v.in_work_period(current_time):
                    eligible_offline.append(v)

        current_available = n_idle_hv + n_near_free_hv + n_av_available

        # ============================================================
        # Step 4: 判断是否需要补充上线
        # ============================================================
        ratio = self._get_supply_ratio(current_time)

        # 目标可用车辆数
        target_available = max(
            int(n_orders_this_window * ratio),
            HV_MIN_ONLINE
        )

        # 还差多少
        deficit = target_available - current_available

        if deficit > 0 and eligible_offline:
            # 需要从 OFFLINE 中激活一些 HV
            n_activate = min(deficit, len(eligible_offline))

            random.seed(int(current_time))
            to_activate = random.sample(eligible_offline, n_activate)

            for v in to_activate:
                v.status = "IDLE"
                # 位置保持最后已知位置（不变）

    def get_available(self, current_time):
        """获取当前可用车辆（IDLE + NEAR_FREE）"""
        return [v for v in self.vehicles.values() if v.is_available(current_time)]

    def assign(self, vehicle, order, current_time, pickup_time):
        """
        执行指派。
        pickup_time 已包含 NEAR_FREE 的剩余等待时间。
        """
        trip_time = float(order.get("ata", order.get("simple_eta", 300)))

        vehicle.status = "BUSY"
        vehicle.current_order_id = order["order_id"]
        vehicle.busy_until = current_time + pickup_time + trip_time
        vehicle.release_link = int(
            order.get("dest_link", vehicle.location_link) or vehicle.location_link
        )
        vehicle.zone = self._get_zone(vehicle.release_link)

    def get_stats(self, current_time):
        stats = {
            "n_idle": 0, "n_busy": 0, "n_near_free": 0, "n_offline": 0,
            "n_av_idle": 0, "n_av_busy": 0, "n_av_near_free": 0,
            "n_hv_idle": 0, "n_hv_busy": 0, "n_hv_near_free": 0, "n_hv_offline": 0,
        }
        for v in self.vehicles.values():
            s = v.status.lower()
            stats[f"n_{s}"] = stats.get(f"n_{s}", 0) + 1
            prefix = "av" if v.is_av() else "hv"
            stats[f"n_{prefix}_{s}"] = stats.get(f"n_{prefix}_{s}", 0) + 1

        stats["n_available"] = stats["n_idle"] + stats["n_near_free"]
        stats["n_av_available"] = stats["n_av_idle"] + stats.get("n_av_near_free", 0)
        stats["n_hv_available"] = stats["n_hv_idle"] + stats.get("n_hv_near_free", 0)
        stats["n_eligible_idle"] = len(self._current_online_hv_ids)

        return stats