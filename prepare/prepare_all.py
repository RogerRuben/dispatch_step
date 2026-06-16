"""
prepare/prepare_all.py
一键运行所有离线预处理
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import DAY


def main(day=None):
    if day is None:
        day = DAY

    print("=" * 60)
    print(f"Stage 4 Data Preparation — Day {day}")
    print("=" * 60)

    # 1. 订单
    from prepare.prepare_orders import prepare_orders
    prepare_orders(day)

    # 2. 车队
    from prepare.prepare_fleet import prepare_fleet
    prepare_fleet(day)

    # 3. 区域（不依赖天，只依赖拓扑）
    from prepare.prepare_zones import prepare_zones
    prepare_zones()

    # 4. 距离
    from prepare.prepare_distances import prepare_distances
    prepare_distances(day)

    print("\n" + "=" * 60)
    print("ALL PREPARATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", default=DAY)
    args = parser.parse_args()
    main(args.day)