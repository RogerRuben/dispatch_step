"""
prepare/prepare_orders.py
订单数据整合 + 秒级时间戳生成
"""
import os
import sys
import re
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import *


def _detect_sep(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        first = f.readline()
    if "\t" in first: return "\t"
    elif "," in first: return ","
    return r"\s+"


def _read_files(file_type, day):
    subdir = SPLIT_SUBDIRS[file_type]
    folder = os.path.join(DATA_DIR, subdir)
    pat = re.compile(rf"^{file_type}_?(\d{{1,2}})(?:_|\.|$)")

    files = []
    for fn in sorted(os.listdir(folder)):
        m = pat.match(fn)
        if m and m.group(1).zfill(2) == day:
            files.append(os.path.join(folder, fn))

    if not files:
        raise FileNotFoundError(f"No {file_type} files for day {day} in {folder}")

    frames = []
    for f in files:
        sep = _detect_sep(f)
        df = pd.read_csv(f, sep=sep, engine="python" if sep == r"\s+" else "c")
        df.columns = df.columns.str.strip()
        df = df.rename(columns=COLUMN_RENAME)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def prepare_orders(day=DAY, difficulty_csv=DIFFICULTY_CSV):
    print(f"[prepare_orders] Loading day {day} ...")

    # 1. Head
    head = _read_files("head", day)
    head["order_id"] = head["order_id"].astype("int64")
    print(f"  Head: {len(head):,} orders")

    # 2. Link（取首尾 link）
    link = _read_files("link", day)
    link["order_id"] = link["order_id"].astype("int64")

    origins = (
        link.groupby("order_id").first()
        .rename(columns={"link_id": "origin_link", "link_ratio": "origin_ratio"})
        [["origin_link", "origin_ratio"]]
        .reset_index()
    )
    dests = (
        link.groupby("order_id").last()
        .rename(columns={"link_id": "dest_link", "link_ratio": "dest_ratio"})
        [["dest_link", "dest_ratio"]]
        .reset_index()
    )
    n_links = (
        link.groupby("order_id")["link_id"].count()
        .rename("n_links").reset_index()
    )

    # 3. 合并
    orders = head.merge(origins, on="order_id", how="left")
    orders = orders.merge(dests, on="order_id", how="left")
    orders = orders.merge(n_links, on="order_id", how="left")

    # 4. 合并 Stage 3 输出
    if difficulty_csv and os.path.exists(difficulty_csv):
        diff = pd.read_csv(difficulty_csv)
        want = [
            "order_id", "difficulty", "grade",
            "D1_cong_exposure", "D2_cong_severity",
            "R1_max_p4", "R2_reject_ratio", "U1_path_entropy",
        ]
        # 尝试加载 Stage 1 输出列
        for col in ["pred_cong_prob", "pred_risk_prob", "pred_entropy"]:
            if col in diff.columns:
                want.append(col)

        avail = [c for c in want if c in diff.columns]
        diff = diff[avail]
        diff["order_id"] = diff["order_id"].astype("int64")
        orders = orders.merge(diff, on="order_id", how="left")
        print(f"  Merged Stage 3: {len(avail)} columns")
    else:
        print("  WARNING: No difficulty_csv, using defaults")
        orders["difficulty"] = 0.0
        orders["grade"] = "G2_AV_Moderate"
        orders["R1_max_p4"] = 0.0

    # 5. 秒级时间戳
    np.random.seed(RANDOM_SEED)
    orders["slice_start_sec"] = orders["slice_id"] * 300
    orders["arrival_time_sec"] = (
        orders["slice_start_sec"]
        + np.random.randint(0, 300, size=len(orders))
    ).astype("float64")

    orders = orders.sort_values("arrival_time_sec").reset_index(drop=True)

    # 6. 填充缺失
    orders["n_links"] = orders["n_links"].fillna(1).astype(int)
    orders["simple_eta"] = orders["simple_eta"].fillna(300).astype(float)
    orders["ata"] = orders["ata"].fillna(orders["simple_eta"]).astype(float)
    orders["distance"] = orders["distance"].fillna(1000).astype(float)

    grade_map = {
        "G1_AV_Easy": 1, "G2_AV_Moderate": 2,
        "G3_AV_Hard": 3, "G4_HV_Only": 4,
    }
    orders["grade_num"] = orders["grade"].map(grade_map).fillna(2).astype(int)

    # 填充 Stage 1/2 可能缺失的列
    for col, default in [
        ("pred_cong_prob", 0.05), ("pred_risk_prob", 0.01),
        ("pred_entropy", 0.5), ("R1_max_p4", 0.01),
        ("R2_reject_ratio", 0.0), ("difficulty", 0.0),
        ("D1_cong_exposure", 0.05), ("D2_cong_severity", 0.1),
        ("U1_path_entropy", 0.3),
    ]:
        if col not in orders.columns:
            orders[col] = default
        else:
            orders[col] = orders[col].fillna(default)

    # 7. 保存
    out_path = os.path.join(DISPATCH_DATA_DIR, f"orders_day{day}.pkl")
    orders.to_pickle(out_path)
    print(f"  Saved: {out_path} ({len(orders):,} orders)")

    # 统计
    sc = orders.groupby("slice_id").size()
    print(f"  Orders/slice: mean={sc.mean():.0f} max={sc.max()} min={sc.min()}")

    return orders


if __name__ == "__main__":
    prepare_orders()