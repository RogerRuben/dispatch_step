"""
core/data_loader.py — 加载所有离线预处理数据
"""
import os
import pandas as pd
from core.config import DISPATCH_DATA_DIR


def load_orders(day: str) -> pd.DataFrame:
    """加载订单表"""
    path = os.path.join(DISPATCH_DATA_DIR, f"orders_day{day}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run: python prepare/prepare_all.py --day {day}"
        )
    orders = pd.read_pickle(path)
    print(f"[data_loader] Orders loaded: {len(orders):,} (day {day})")
    return orders


def load_fleet_schedule(day: str) -> dict:
    """
    加载车队配置。

    返回:
      {
        "hv_schedules": {driver_id: {"entry_sec", "exit_sec", "first_link", ...}},
        "av_ids": [av_id, ...],
        "n_hv": int,
        "n_av": int,
        "summary": {...},
      }
    """
    path = os.path.join(DISPATCH_DATA_DIR, f"fleet_schedule_day{day}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run: python prepare/prepare_all.py --day {day}"
        )
    fleet = pd.read_pickle(path)
    print(f"[data_loader] Fleet loaded: HV={fleet['n_hv']:,} AV={fleet['n_av']:,}")
    return fleet


def load_zones() -> tuple:
    """
    加载区域划分和邻接关系。

    返回:
      link_to_zone: dict[int, int]
      zone_neighbors: dict[int, list[int]]
    """
    ltz_path = os.path.join(DISPATCH_DATA_DIR, "link_to_zone.pkl")
    zn_path = os.path.join(DISPATCH_DATA_DIR, "zone_neighbors.pkl")

    if not os.path.exists(ltz_path) or not os.path.exists(zn_path):
        raise FileNotFoundError(
            "Zone files not found. Run: python prepare/prepare_all.py"
        )

    link_to_zone = pd.read_pickle(ltz_path)
    zone_neighbors = pd.read_pickle(zn_path)

    n_zones = len(set(link_to_zone.values()))
    print(f"[data_loader] Zones loaded: {n_zones} zones, "
          f"{len(link_to_zone):,} links mapped")

    return link_to_zone, zone_neighbors


def load_link_distances(day: str) -> dict:
    """
    加载 link 级代理距离表。

    返回:
      dict[src_link_id -> dict[dst_link_id -> travel_time_sec]]
    """
    path = os.path.join(DISPATCH_DATA_DIR, f"link_neighbor_dist_day{day}.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run: python prepare/prepare_all.py --day {day}"
        )
    link_dist = pd.read_pickle(path)
    total_entries = sum(len(v) for v in link_dist.values())
    print(f"[data_loader] Link distances loaded: {len(link_dist):,} source links, "
          f"{total_entries:,} total entries")
    return link_dist