"""Controlled anomalies injected into a COPY of the real orders (adlens_orders_eval). The real table is never modified.

Each scenario has a known cause (type, dimension, segment). The current day is compared with the previous day of the same
"type" (the real data alternates between short-distance and long-distance days, so same-parity days are used as baselines;
see README). Injected magnitudes move the daily mean delivery time by roughly 5-12%.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from database.connection import execute, query  # noqa: E402

REAL = "food_delivery_analytics.real_orders"
EVAL = f"{config.WORK_DB}.adlens_orders_eval"
# "short-distance" days of the real data (all orders < 10 km); consecutive entries are 2 days apart
SHORT_DAYS = ["2022-03-01", "2022-03-03", "2022-03-05", "2022-03-07", "2022-03-09", "2022-03-11", "2022-03-13", "2022-03-15", "2022-03-17",
              "2022-03-19", "2022-03-21", "2022-03-24", "2022-03-26", "2022-03-28", "2022-03-30", "2022-04-01", "2022-04-03", "2022-04-05"]

SCENARIOS = {
    # name: (cause type, dimension, segment, SQL template applied to the current day)
    "jam_slowdown": ("RATE_CHANGE", "traffic_density", "Jam",
                     "UPDATE {t} SET time_taken_min = time_taken_min + 9 WHERE DATE(order_datetime) = %(d)s AND traffic_density = 'Jam'"),
    "fog_slowdown": ("RATE_CHANGE", "weather", "Fog",
                     "UPDATE {t} SET time_taken_min = time_taken_min + 10 WHERE DATE(order_datetime) = %(d)s AND weather = 'Fog'"),
    "urban_slowdown": ("RATE_CHANGE", "city_type", "Urban",
                       "UPDATE {t} SET time_taken_min = time_taken_min + 7 WHERE DATE(order_datetime) = %(d)s AND city_type = 'Urban'"),
    "bad_vehicle_slowdown": ("RATE_CHANGE", "vehicle_condition", "0",
                             "UPDATE {t} SET time_taken_min = time_taken_min + 8 WHERE DATE(order_datetime) = %(d)s AND vehicle_condition = 0"),
    "low_rated_couriers": ("RATE_CHANGE", "courier_rating_band", "<4.5",
                           "UPDATE {t} SET time_taken_min = time_taken_min + 9 WHERE DATE(order_datetime) = %(d)s AND delivery_person_rating < 4.5"),
    "festival_surge": ("MIX_SHIFT", "festival", "Yes",
                       "INSERT INTO {t} SELECT CONCAT(order_id,'_inj'), delivery_person_id, delivery_person_age, delivery_person_rating, "
                       "TIMESTAMP(%(d)s, TIME(order_datetime)), TIMESTAMP(%(d)s, TIME(pickup_datetime)), prep_time_min, distance_km, weather, "
                       "traffic_density, vehicle_condition, order_type, vehicle_type, multiple_deliveries, festival, city_type, time_taken_min, on_time "
                       "FROM {t} WHERE festival = 'Yes' AND DATE(order_datetime) IN %(others)s AND distance_km < 10 LIMIT 160"),
    "batched_orders": ("MIX_SHIFT", "multiple_deliveries", "2",
                       "INSERT INTO {t} SELECT CONCAT(order_id,'_inj'), delivery_person_id, delivery_person_age, delivery_person_rating, "
                       "TIMESTAMP(%(d)s, TIME(order_datetime)), TIMESTAMP(%(d)s, TIME(pickup_datetime)), prep_time_min, distance_km, weather, "
                       "traffic_density, vehicle_condition, order_type, vehicle_type, multiple_deliveries, festival, city_type, time_taken_min, on_time "
                       "FROM {t} WHERE multiple_deliveries >= 2 AND DATE(order_datetime) IN %(others)s AND distance_km < 10 LIMIT 260"),
    "jam_share_up": ("MIX_SHIFT", "traffic_density", "Jam",
                     "DELETE FROM {t} WHERE DATE(order_datetime) = %(d)s AND traffic_density = 'Low' AND MOD(CRC32(order_id), 10) < 6; "
                     "INSERT INTO {t} SELECT CONCAT(order_id,'_inj'), delivery_person_id, delivery_person_age, delivery_person_rating, "
                     "TIMESTAMP(%(d)s, TIME(order_datetime)), TIMESTAMP(%(d)s, TIME(pickup_datetime)), prep_time_min, distance_km, weather, "
                     "traffic_density, vehicle_condition, order_type, vehicle_type, multiple_deliveries, festival, city_type, time_taken_min, on_time "
                     "FROM {t} WHERE traffic_density = 'Jam' AND DATE(order_datetime) IN %(others)s AND distance_km < 10 LIMIT 280"),
    "none": (None, None, None, ""),
}


def reset():
    execute(f"DROP TABLE IF EXISTS {EVAL}")
    execute(f"CREATE TABLE {EVAL} LIKE {REAL}")
    execute(f"INSERT INTO {EVAL} SELECT * FROM {REAL}")


def inject(name: str, day: str) -> dict:
    """Rebuild the eval copy and inject scenario `name` on `day`. Returns the ground truth."""
    reset()
    ctype, dim, seg, sql = SCENARIOS[name]
    others = tuple(d for d in SHORT_DAYS if d != day)
    for stmt in [s.strip() for s in sql.split(";") if s.strip()]:
        execute(stmt.format(t=EVAL), dict(d=day, others=others))
    if sql:
        execute(f"UPDATE {EVAL} SET on_time = (time_taken_min <= 30) WHERE DATE(order_datetime) = %(d)s", dict(d=day))
    return dict(scenario=name, type=ctype, dimension=dim, segment=seg, day=day, baseline_day=SHORT_DAYS[SHORT_DAYS.index(day) - 1])


def clean_value(metric_sql: str, day: str) -> float:
    return float(query(f"SELECT {metric_sql} AS v FROM {REAL} WHERE DATE(order_datetime) = %(d)s", dict(d=day)).v.iloc[0])
