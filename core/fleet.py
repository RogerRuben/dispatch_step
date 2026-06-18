"""
core/fleet.py - 车辆与车队状态管理（修复版：动态配置 + zone_to_links 缓存 + 性能优化）
"""
import os
import random
import numpy as np
from collections import defaultdict
import core.config as cfg

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

    def is_available(self, current_time, theta=None):
        if theta is None:
            theta = cfg.THETA_NEAR_FREE
        if self.status == "IDLE":
            return True
        if self.status == "NEAR_FREE":
            return (self.busy_until - current_time) <= theta
        return False

    def is_av(self):
        return self.vtype == "AV"

    def in_work_period(self, current_time):
        return self.entry_sec <= current_time <= self.exit_sec


class FleetManager:
    # ★ 确保 __init__ 接受三个参数
    def __init__(self, fleet_schedule, link_to_zone, day=None):
        self.link_to_zone = link_to_zone
        # ★ 预建反向映射，加速 AV 初始化
        self.zone_to_links = {}
        for link, zone in link_to_zone.items():
            self.zone_to_links.setdefault(zone, []).append(link)

        self.vehicles = {}
        self.total_hv = 0
        self.total_av = 0
        self._current_online_hv_ids = set()
        self.zone_demand = fleet_schedule.get("zone_demand", {})
        if not self.zone_demand:
            print("[FleetManager] Warning: No zone_demand provided. AV will use random HV locations.")
        self._init_fleet(fleet_schedule)

    def _get_zone(self, link_id):
        return self.link_to_zone.get(int(link_id), 0) if link_id else 0

    def _init_fleet(self, fleet_schedule):
        # -------- HV 初始化 --------
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

        # -------- AV 初始化（使用 fleet_schedule 中预定义的 av_ids） --------
        random.seed(42)
        hv_locs = [v.location_link for v in self.vehicles.values() if v.location_link != 0]

        av_ids = fleet_schedule.get("av_ids", [])
        n_av = len(av_ids)
        if n_av == 0:
            # 若预定义为空，则回退到配置比例（但理论上不应发生）
            n_av = max(0, int(self.total_hv * cfg.AV_RATIO))
            av_ids = list(range(-1, -n_av - 1, -1))

        # 按需求密度采样 zone
        if self.zone_demand and n_av > 0:
            zones = list(self.zone_demand.keys())
            weights = np.array([self.zone_demand.get(z, 0) for z in zones], dtype=np.float64)
            weights = weights / max(weights.sum(), 1e-6)
            sampled_zones = np.random.choice(zones, size=n_av, p=weights, replace=True)
        else:
            sampled_zones = [None] * n_av if n_av > 0 else []

        for idx, av_id in enumerate(av_ids):
            if idx < len(sampled_zones) and sampled_zones[idx] is not None:
                target_zone = sampled_zones[idx]
                zone_links = self.zone_to_links.get(target_zone, [])
                loc = random.choice(zone_links) if zone_links else (random.choice(hv_locs) if hv_locs else 0)
            else:
                loc = random.choice(hv_locs) if hv_locs else 0
                target_zone = self._get_zone(loc)

            v = Vehicle(
                vid=int(av_id),
                vtype="AV",
                location_link=loc,
                zone=target_zone,
                entry_sec=0.0,
                exit_sec=86400.0,
            )
            v.status = "IDLE"
            self.vehicles[v.id] = v
            self.total_av += 1

        print(f"[FleetManager] Init: HV={self.total_hv} AV={self.total_av} (AV_RATIO={cfg.AV_RATIO:.2f})")

    def _get_supply_ratio(self, current_time):
        hour = current_time / 3600.0
        if hour < 6 or hour >= 22:
            return cfg.TARGET_SUPPLY_RATIOS["night"]
        if (7 <= hour < 9) or (17 <= hour < 19):
            return cfg.TARGET_SUPPLY_RATIOS["peak"]
        return cfg.TARGET_SUPPLY_RATIOS["normal"]

    def update(self, current_time, n_orders_this_window=0):
        # Step1: 释放完成订单
        for v in self.vehicles.values():
            if v.status in ("BUSY", "NEAR_FREE"):
                if current_time >= v.busy_until:
                    v.status = "IDLE"
                    v.location_link = v.release_link
                    v.zone = self._get_zone(v.release_link)
                    v.current_order_id = None
                elif (v.busy_until - current_time) <= cfg.THETA_NEAR_FREE:
                    v.status = "NEAR_FREE"

        # Step2: 在线HV下班
        for v in self.vehicles.values():
            if v.vtype != "HV":
                continue
            if v.status == "IDLE":
                if not v.in_work_period(current_time):
                    v.status = "OFFLINE"

        # Step3: 统计供给，并限制HV激活范围（防止瞬移）
        n_idle_hv = 0
        n_near_free_hv = 0
        n_av_available = 0
        eligible_offline = []

        # 确定热区（基于静态 zone_demand）
        hot_zones = set()
        if self.zone_demand:
            sorted_zones = sorted(self.zone_demand.items(), key=lambda x: -x[1])
            hot_zones = {z for z, _ in sorted_zones[:5]}  # Top 5

        for v in self.vehicles.values():
            if v.is_av():
                if v.status in ("IDLE", "NEAR_FREE"):
                    n_av_available += 1
                continue

            # HV
            if v.status == "IDLE":
                n_idle_hv += 1
            elif v.status == "NEAR_FREE":
                n_near_free_hv += 1
            elif v.status == "OFFLINE":
                if not v.in_work_period(current_time):
                    continue
                # 仅当该车所在 Zone 在热区列表中才允许激活，防止瞬移
                if not hot_zones or v.zone in hot_zones:
                    eligible_offline.append(v)

        current_available = n_idle_hv + n_near_free_hv + n_av_available
        ratio = self._get_supply_ratio(current_time)
        target_available = max(int(n_orders_this_window * ratio), cfg.HV_MIN_ONLINE)
        deficit = target_available - current_available

        if deficit > 0 and eligible_offline:
            n_activate = min(deficit, len(eligible_offline))
            random.seed(int(current_time))
            to_activate = random.sample(eligible_offline, n_activate)
            for v in to_activate:
                v.status = "IDLE"
                # 位置保持不变，防止瞬移

    def get_available(self, current_time):
        return [v for v in self.vehicles.values() if v.is_available(current_time)]

    def assign(self, vehicle, order, current_time, pickup_time):
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
        return stats