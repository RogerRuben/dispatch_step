"""
prepare/prepare_distances.py
link 级代理距离表（只对活跃 link 构建）
"""
import os
import sys
import re
import glob
from collections import deque
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import *
from prepare.prepare_zones import load_topology


def prepare_distances(day=DAY):
    print(f"[prepare_distances] Building link neighbor distances for day {day} ...")

    topo = load_topology()

    # 加载订单，获取活跃 link 集合
    orders_path = os.path.join(DISPATCH_DATA_DIR, f"orders_day{day}.pkl")
    orders = pd.read_pickle(orders_path)

    active_links = set(
        orders["origin_link"].dropna().astype(int).tolist()
        + orders["dest_link"].dropna().astype(int).tolist()
    )
    print(f"  Active links: {len(active_links):,}")

    # 加载 link_time 均值
    from prepare.prepare_orders import _read_files
    link_df = _read_files("link", day)
    link_avg_time = (
        link_df.groupby("link_id")["link_time"]
        .mean()
        .to_dict()
    )
    default_time = float(np.median(list(link_avg_time.values()))) if link_avg_time else 10.0

    # BFS 构建邻居距离
    neighbors = {}
    processed = 0

    for src in active_links:
        if src not in topo:
            continue

        visited = {src: 0.0}
        queue = deque([(src, 0.0, 0)])

        while queue:
            current, dist, hops = queue.popleft()
            if hops >= TOPO_MAX_HOPS:
                continue
            for nxt in topo.get(current, []):
                lt = link_avg_time.get(nxt, default_time)
                new_dist = dist + lt
                if nxt not in visited or new_dist < visited[nxt]:
                    visited[nxt] = new_dist
                    queue.append((nxt, new_dist, hops + 1))

        neighbors[src] = visited
        processed += 1

        if processed % 5000 == 0:
            print(f"    Processed {processed:,}/{len(active_links):,} links")

    out_path = os.path.join(DISPATCH_DATA_DIR, f"link_neighbor_dist_day{day}.pkl")
    pd.to_pickle(neighbors, out_path)
    print(f"  Saved: {out_path}")
    print(f"  Total entries: {sum(len(v) for v in neighbors.values()):,}")

    return neighbors


if __name__ == "__main__":
    prepare_distances()