"""
simulation/simulator.py — 动态调度仿真主循环（修复版）

修复:
  1. _dispatch_by_zone 传入 current_time
  2. compute_cost_matrix 接收 current_time（NEAR_FREE 剩余时间）
  3. 日志打印改成每小时聚合
"""
import time as time_module
import numpy as np
from collections import defaultdict

from core.config import (
    DISPATCH_INTERVAL, SECONDS_PER_DAY,
    MAX_CANDIDATE_ZONES, MAX_CANDIDATES_PER_ZONE,
    TAU_SAFE_Q3, BIG_M, DEFAULT_PICKUP_TIME,RELOCATION_ENABLED
)
from core.fleet import FleetManager
from core.cost import compute_cost_matrix
from simulation.recorder import DispatchRecorder


class DispatchSimulator:

    def __init__(self, orders_df, fleet_manager, link_distances,
                 link_to_zone, zone_neighbors,
                 dispatch_interval=DISPATCH_INTERVAL):

        self.orders_df = orders_df
        self.fleet = fleet_manager
        self.link_dist = link_distances
        self.link_to_zone = link_to_zone
        self.zone_neighbors = zone_neighbors
        self.interval = dispatch_interval

        if "zone" not in self.orders_df.columns:
            self.orders_df["zone"] = (
                self.orders_df["origin_link"]
                .map(link_to_zone).fillna(0).astype(int)
            )

        self.orders_df["window_id"] = (
            self.orders_df["arrival_time_sec"] // self.interval
        ).astype(int)

        self.window_orders = {}
        for wid, grp in self.orders_df.groupby("window_id"):
            self.window_orders[int(wid)] = grp.to_dict("records")

        self.total_windows = SECONDS_PER_DAY // self.interval

        print(f"[simulator] Init: {len(self.orders_df):,} orders, "
              f"{self.total_windows} windows ({self.interval}s)")

    def run(self, dispatcher, verbose=True):
        recorder = DispatchRecorder(
            total_av_vehicles=self.fleet.total_av,
            total_simulation_seconds=self.total_windows * self.interval
        )
        strategy_name = getattr(dispatcher, "name", "unknown")

        if verbose:
            print(f"\n{'='*60}")
            print(f"Simulation: {strategy_name}")
            print(f"{'='*60}")

        sim_start = time_module.time()

        for wid in range(self.total_windows):
            current_time = float(wid * self.interval)

            window_orders = self.window_orders.get(wid, [])
            n_orders = len(window_orders)


            # ★ 把当前窗口订单数传给 fleet，用于供需比校准
            self.fleet.update(current_time, n_orders_this_window=n_orders)
            # ★ 优先级2：空闲车辆主动巡游（向热区步进）
            self._relocate_idle_vehicles(current_time, window_orders)

            # 获取可用车辆（IDLE + NEAR_FREE）
            
            available = self.fleet.get_available(current_time)
            fleet_stats = self.fleet.get_stats(current_time)

            if n_orders == 0:
                recorder.record_window(
                    window_id=wid, time_sec=current_time,
                    n_orders=0, n_matched=0,
                    n_available=len(available),
                    n_av_available=fleet_stats["n_av_available"],
                    n_hv_available=fleet_stats["n_hv_available"],
                    n_av_assigned=0, n_hv_assigned=0,
                    n_cross_zone=0, avg_pickup_time=0.0,
                    n_av_high_risk=0, n_av_extreme_risk=0,
                    fleet_stats=fleet_stats,
                )
                continue

            if len(available) == 0:
                for order in window_orders:
                    recorder.record_order(
                        order_id=order["order_id"], window_id=wid,
                        grade=order.get("grade", "G2"),
                        grade_num=order.get("grade_num", 2),
                        difficulty=order.get("difficulty", 0.0),
                        pred_risk_prob=order.get("pred_risk_prob", 0.0),
                        matched=False,
                    )
                recorder.record_window(
                    window_id=wid, time_sec=current_time,
                    n_orders=n_orders, n_matched=0,
                    n_available=0, n_av_available=0, n_hv_available=0,
                    n_av_assigned=0, n_hv_assigned=0,
                    n_cross_zone=0, avg_pickup_time=0.0,
                    n_av_high_risk=0, n_av_extreme_risk=0,
                    fleet_stats=fleet_stats,
                )
                continue

            # ★ 修复：传入 current_time
            all_matched, all_unmatched_indices = self._dispatch_by_zone(
                window_orders, available, dispatcher, current_time
            )

            # 执行指派
            n_av_assigned = 0
            n_hv_assigned = 0
            n_cross_zone = 0
            n_av_high_risk = 0
            n_av_extreme_risk = 0
            pickup_times_list = []
            matched_order_ids = set()

            for oi, vi, pt in all_matched:
                order = window_orders[oi]
                vehicle = available[vi]

                order_zone = order.get("zone", 0)
                is_cross = (order_zone != vehicle.zone)

                self.fleet.assign(vehicle, order, current_time, pt)

                if vehicle.is_av():
                    n_av_assigned += 1
                    if order.get("grade_num", 2) >= 3:
                        n_av_high_risk += 1
                    if order.get("pred_risk_prob", 0) > TAU_SAFE_Q3:
                        n_av_extreme_risk += 1
                else:
                    n_hv_assigned += 1

                if is_cross:
                    n_cross_zone += 1

                pickup_times_list.append(pt)
                matched_order_ids.add(order["order_id"])

                total_cost = pt + float(order.get("ata", order.get("simple_eta", 300)))

                recorder.record_order(
                    order_id=order["order_id"], window_id=wid,
                    grade=order.get("grade", "G2"),
                    grade_num=order.get("grade_num", 2),
                    difficulty=order.get("difficulty", 0.0),
                    pred_risk_prob=order.get("pred_risk_prob", 0.0),
                    matched=True,
                    vehicle_id=vehicle.id, vehicle_type=vehicle.vtype,
                    pickup_time=pt, total_cost=total_cost,
                    cross_zone=is_cross,
                    zone=order.get("zone", 0)  # ★ 新增
                )

            for oi in all_unmatched_indices:
                order = window_orders[oi]
                if order["order_id"] not in matched_order_ids:
                    recorder.record_order(
                        order_id=order["order_id"], window_id=wid,
                        grade=order.get("grade", "G2"),
                        grade_num=order.get("grade_num", 2),
                        difficulty=order.get("difficulty", 0.0),
                        pred_risk_prob=order.get("pred_risk_prob", 0.0),
                        matched=False,
                        zone=order.get("zone", 0)  # ★ 新增
                    )

            avg_pt = float(np.mean(pickup_times_list)) if pickup_times_list else 0.0

            recorder.record_window(
                window_id=wid, time_sec=current_time,
                n_orders=n_orders, n_matched=len(all_matched),
                n_available=len(available),
                n_av_available=fleet_stats["n_av_available"],
                n_hv_available=fleet_stats["n_hv_available"],
                n_av_assigned=n_av_assigned, n_hv_assigned=n_hv_assigned,
                n_cross_zone=n_cross_zone, avg_pickup_time=avg_pt,
                n_av_high_risk=n_av_high_risk,
                n_av_extreme_risk=n_av_extreme_risk,
                fleet_stats=fleet_stats,
            )

            # ★ 每小时聚合打印
            if verbose and (wid % 30 == 29 or wid == self.total_windows - 1):
                hour = current_time / 3600
                elapsed = time_module.time() - sim_start

                wl = recorder.get_window_log()
                recent = wl.tail(30) if len(wl) >= 30 else wl

                hour_orders = int(recent["n_orders"].sum())
                hour_matched = int(recent["n_matched"].sum())
                hour_match_rate = hour_matched / max(hour_orders, 1)
                hour_avg_pickup = float(recent["avg_pickup_time"].mean()) if len(recent) > 0 else 0.0
                hour_av = int(recent["n_av_assigned"].sum())
                hour_hv = int(recent["n_hv_assigned"].sum())

                print(
                    f"  [{strategy_name}] "
                    f"t={hour:05.2f}h | "
                    f"window: orders={n_orders} matched={len(all_matched)} | "
                    f"last_1h: orders={hour_orders:,} matched={hour_matched:,} "
                    f"({hour_match_rate:.1%}) "
                    f"avg_pickup={hour_avg_pickup:.0f}s "
                    f"AV={hour_av} HV={hour_hv} | "
                    f"avail={len(available)} "
                    f"(AV={fleet_stats['n_av_available']} "
                    f"HV={fleet_stats['n_hv_available']}) | "
                    f"elapsed={elapsed:.0f}s"
                )

        total_time = time_module.time() - sim_start

        if verbose:
            summary = recorder.get_summary()
            print(f"\n  [{strategy_name}] DONE in {total_time:.1f}s")
            print(f"  Match rate: {summary['match_rate']:.4f}")
            print(f"  Avg pickup: {summary['avg_pickup_time']:.1f}s")
            print(f"  AV util: {summary['av_utilization']:.4f}")
            print(f"  AV high risk: {summary['av_high_risk_rate']:.4f}")

        return recorder

    def _dispatch_by_zone(self, window_orders, available_vehicles,
                          dispatcher, current_time):
        """
        按 zone 分治调度。
        ★ current_time 传给 compute_cost_matrix。
        """
        zone_order_map = defaultdict(list)
        for gi in range(len(window_orders)):
            z = int(window_orders[gi].get("zone", 0))
            zone_order_map[z].append(gi)

        zone_vehicle_map = defaultdict(list)
        for gj in range(len(available_vehicles)):
            z = int(available_vehicles[gj].zone)
            zone_vehicle_map[z].append(gj)

        used_vehicles = set()
        all_matched = []
        all_unmatched = []

        for zone_id in sorted(zone_order_map.keys()):
            order_gis = zone_order_map[zone_id]
            if not order_gis:
                continue

            cand_gjs = []
            for gj in zone_vehicle_map.get(zone_id, []):
                if gj not in used_vehicles:
                    cand_gjs.append(gj)

            added = 0
            for nz in self.zone_neighbors.get(zone_id, []):
                if added >= MAX_CANDIDATE_ZONES - 1:
                    break
                for gj in zone_vehicle_map.get(nz, []):
                    if gj not in used_vehicles:
                        cand_gjs.append(gj)
                added += 1

            if not cand_gjs:
                all_unmatched.extend(order_gis)
                continue

            # ★ 候选车辆截断
            if len(cand_gjs) > MAX_CANDIDATES_PER_ZONE:
                ref_link = int(window_orders[order_gis[0]].get("origin_link", 0))

                def _approx_dist(gj):
                    v = available_vehicles[gj]
                    vl = int(v.release_link if v.status == "NEAR_FREE" else v.location_link)
                    if vl == ref_link:
                        return 0.0
                    if vl in self.link_dist and ref_link in self.link_dist[vl]:
                        return self.link_dist[vl][ref_link]
                    if ref_link in self.link_dist and vl in self.link_dist[ref_link]:
                        return self.link_dist[ref_link][vl]
                    return DEFAULT_PICKUP_TIME

                cand_gjs.sort(key=_approx_dist)
                cand_gjs = cand_gjs[:MAX_CANDIDATES_PER_ZONE]

            sub_orders = [window_orders[i] for i in order_gis]
            sub_vehicles = [available_vehicles[j] for j in cand_gjs]

            # ★ 修复：传入 current_time
            cost_mat, pickup_mat = compute_cost_matrix(
                sub_orders, sub_vehicles, self.link_dist,
                current_time=current_time,
            )

            matched, unmatched_local = dispatcher.dispatch(
                sub_orders, sub_vehicles, cost_mat, pickup_mat
            )

            for loc_oi, loc_vi, pt in matched:
                gi = int(order_gis[loc_oi])
                gj = int(cand_gjs[loc_vi])
                all_matched.append((gi, gj, pt))
                used_vehicles.add(gj)

            for loc_oi in unmatched_local:
                gi = int(order_gis[loc_oi])
                all_unmatched.append(gi)

        return all_matched, all_unmatched

    def _relocate_idle_vehicles(self, current_time: float, window_orders: list):
        """
        空闲车辆主动巡游：向需求热点步进移动，解决空间僵死问题。
        AV 优先向 G1/G2 订单区移动，HV 向所有未匹配订单区移动。
        """
        from core.config import RELOCATION_ENABLED
        if not RELOCATION_ENABLED or len(window_orders) == 0:
            return

        # 1. 统计当前窗口的需求分布（按 zone 聚合）
        zone_demand = defaultdict(int)
        zone_g12_demand = defaultdict(int)
        for order in window_orders:
            z = int(order.get("zone", 0))
            grade = order.get("grade_num", 2)
            zone_demand[z] += 1
            if grade <= 2:  # G1 或 G2
                zone_g12_demand[z] += 1

        if not zone_demand:
            return

        # 找 Top-3 热区
        top_hot_zones = [z for z, _ in sorted(zone_demand.items(), key=lambda x: -x[1])[:3]]
        top_g12_zones = [z for z, _ in sorted(zone_g12_demand.items(), key=lambda x: -x[1])[:3]]

        # 构建 zone -> link 缓存（避免重复构造）
        if not hasattr(self, '_zone_to_link_cache'):
            self._zone_to_link_cache = {}
            for link, z in self.link_to_zone.items():
                self._zone_to_link_cache.setdefault(z, []).append(link)

        def _move_vehicle_towards(v, target_zones):
            if not target_zones:
                return
            current_z = v.zone
            if current_z in target_zones:
                return
            neighbors = self.zone_neighbors.get(current_z, [])
            if not neighbors:
                return
            # 贪心：选择邻居中第一个属于热区的，否则随机选一个
            best_next_z = None
            for nz in neighbors:
                if nz in target_zones:
                    best_next_z = nz
                    break
            if best_next_z is None:
                best_next_z = neighbors[0]

            zone_links = self._zone_to_link_cache.get(best_next_z, [])
            if zone_links:
                new_link = zone_links[0]
                v.location_link = new_link
                v.zone = best_next_z

        # AV 向 G1/G2 热点移动
        for vehicle in self.fleet.vehicles.values():
            if vehicle.status == "IDLE" and vehicle.is_av():
                _move_vehicle_towards(vehicle, top_g12_zones)

        # HV 向全局热点移动
        for vehicle in self.fleet.vehicles.values():
            if vehicle.status == "IDLE" and not vehicle.is_av():
                _move_vehicle_towards(vehicle, top_hot_zones)