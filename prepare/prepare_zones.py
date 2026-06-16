"""
prepare/prepare_zones.py
拓扑区域划分 + 邻接关系
"""
import os
import sys
import random
from collections import deque, defaultdict
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import *


def load_topology():
    topo = {}
    with open(TOPO_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            src = int(parts[0])
            dsts = [int(x) for x in parts[1].split(",") if x] if len(parts) > 1 else []
            topo[src] = dsts
    print(f"  Topology: {len(topo):,} links")
    return topo


def cluster_links(topo, n_clusters=N_ZONES, seed=RANDOM_SEED):
    """BFS 多源聚类"""
    print(f"[prepare_zones] Clustering {len(topo):,} links into {n_clusters} zones ...")

    all_links = list(topo.keys())
    random.seed(seed)
    seeds = random.sample(all_links, min(n_clusters, len(all_links)))

    link_to_zone = {}
    queue = deque()

    for cid, seed_link in enumerate(seeds):
        link_to_zone[seed_link] = cid
        queue.append((seed_link, cid))

    while queue:
        current, cid = queue.popleft()
        for nxt in topo.get(current, []):
            if nxt not in link_to_zone:
                link_to_zone[nxt] = cid
                queue.append((nxt, cid))

    # 孤立 link → zone 0
    for lid in all_links:
        if lid not in link_to_zone:
            link_to_zone[lid] = 0

    # 统计
    zone_counts = defaultdict(int)
    for z in link_to_zone.values():
        zone_counts[z] += 1

    counts = list(zone_counts.values())
    print(f"  Zone sizes: mean={sum(counts)/len(counts):.0f} "
          f"max={max(counts)} min={min(counts)}")

    return link_to_zone


def build_zone_neighbors(link_to_zone, topo):
    """构建 zone 邻接关系"""
    neighbors = defaultdict(set)

    for src, dsts in topo.items():
        src_z = link_to_zone.get(src)
        if src_z is None:
            continue
        for dst in dsts:
            dst_z = link_to_zone.get(dst)
            if dst_z is not None and src_z != dst_z:
                neighbors[src_z].add(dst_z)
                neighbors[dst_z].add(src_z)

    result = {k: sorted(v) for k, v in neighbors.items()}
    print(f"  Zone neighbor pairs: {sum(len(v) for v in result.values()) // 2}")
    return result


def prepare_zones():
    topo = load_topology()
    link_to_zone = cluster_links(topo)
    zone_neighbors = build_zone_neighbors(link_to_zone, topo)

    ltz_path = os.path.join(DISPATCH_DATA_DIR, "link_to_zone.pkl")
    zn_path = os.path.join(DISPATCH_DATA_DIR, "zone_neighbors.pkl")

    pd.to_pickle(link_to_zone, ltz_path)
    pd.to_pickle(zone_neighbors, zn_path)

    print(f"  Saved: {ltz_path}")
    print(f"  Saved: {zn_path}")

    return link_to_zone, zone_neighbors


if __name__ == "__main__":
    prepare_zones()