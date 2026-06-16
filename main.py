"""
main.py — Stage 4 入口

用法:
  python main.py                    # 跑所有策略
  python main.py --strategy Full    # 只跑一个策略
  python main.py --analyze          # 只做分析（不跑仿真）
"""
import os
import sys
import argparse
import time

# 确保能找到 core/ simulation/ evaluation/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import DAY, RESULTS_DIR
from core.data_loader import (
    load_orders, load_fleet_schedule, load_zones, load_link_distances,
)
from core.fleet import FleetManager
from core.dispatcher import TieredDispatcher
from core.baselines import (
    RandomDispatcher, NearestDispatcher,
    NearestSafeDispatcher, GradeOnlyDispatcher,
    GlobalMatchDispatcher,
)
from simulation.simulator import DispatchSimulator
from evaluation.analysis import full_analysis
from evaluation.visualize import plot_all


def create_simulator(day=DAY):
    orders = load_orders(day)
    fleet_schedule = load_fleet_schedule(day)
    link_to_zone, zone_neighbors = load_zones()
    link_distances = load_link_distances(day)

    # ★ 不再传 day（不再加载 window_active_drivers）
    fleet_manager = FleetManager(fleet_schedule, link_to_zone)

    simulator = DispatchSimulator(
        orders_df=orders,
        fleet_manager=fleet_manager,
        link_distances=link_distances,
        link_to_zone=link_to_zone,
        zone_neighbors=zone_neighbors,
    )

    return simulator, fleet_schedule, link_to_zone


def get_all_dispatchers():
    """返回所有要比较的调度策略"""
    return {
        "Random": RandomDispatcher(),
        "Nearest": NearestDispatcher(),
        "NearestSafe": NearestSafeDispatcher(),
        "GradeOnly": GradeOnlyDispatcher(),
        "Full": TieredDispatcher(name="Full"),
    }


def run_single_strategy(strategy_name, day=DAY):
    """运行单个策略"""
    simulator, fleet_schedule, link_to_zone = create_simulator(day)

    dispatchers = get_all_dispatchers()
    if strategy_name not in dispatchers:
        print(f"Unknown strategy: {strategy_name}")
        print(f"Available: {list(dispatchers.keys())}")
        return

    dispatcher = dispatchers[strategy_name]
    recorder = simulator.run(dispatcher, verbose=True)
    recorder.save(strategy_name)


def run_all_strategies(day=DAY):
    """运行所有策略"""
    dispatchers = get_all_dispatchers()

    for name, dispatcher in dispatchers.items():
        # 每个策略都要重新创建仿真器（重置车队状态）
        simulator, fleet_schedule, link_to_zone = create_simulator(day)

        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")

        t0 = time.time()
        recorder = simulator.run(dispatcher, verbose=True)
        elapsed = time.time() - t0

        recorder.save(name)
        print(f"[{name}] Total time: {elapsed:.1f}s")

    print("\n" + "=" * 60)
    print("ALL STRATEGIES COMPLETE")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Dynamic Dispatch Simulation")
    parser.add_argument("--day", default=DAY, help="Day to simulate")
    parser.add_argument("--strategy", default=None,
                        help="Run only this strategy (e.g., Full, Nearest)")
    parser.add_argument("--analyze", action="store_true",
                        help="Only run analysis (no simulation)")
    parser.add_argument("--plot", action="store_true",
                        help="Only generate plots (no simulation)")
    args = parser.parse_args()

    if args.analyze:
        full_analysis()
        return

    if args.plot:
        plot_all()
        return

    if args.strategy:
        run_single_strategy(args.strategy, args.day)
    else:
        run_all_strategies(args.day)

    # 仿真完自动做分析和画图
    print("\n" + "=" * 60)
    print("ANALYSIS & VISUALIZATION")
    print("=" * 60)

    full_analysis()
    plot_all()


if __name__ == "__main__":
    main()