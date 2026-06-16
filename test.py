import pandas as pd

ol = pd.read_pickle("results/Full_orders.pkl")
wl = pd.read_pickle("results/Full_windows.pkl")

# 每小时匹配的独立车辆数
ol_matched = ol[ol["matched"]]
ol_matched["hour"] = ol_matched["window_id"] * 120 // 3600

hourly_vehicles = ol_matched.groupby("hour")["vehicle_id"].nunique()
hourly_orders = ol_matched.groupby("hour")["order_id"].count()
hourly_ratio = hourly_orders / hourly_vehicles

print("Hour | Orders | Unique Vehicles | Orders/Vehicle")
for h in sorted(hourly_vehicles.index):
    print(f"  {h:2d}  | {hourly_orders[h]:6d} | {hourly_vehicles[h]:15d} | {hourly_ratio[h]:.1f}")