"""SYNTHETIC sponsored-listing layer on top of REAL restaurants.

The real Swiggy restaurant table has rating, rating count, cost tier, cuisine and city but no time dimension and no ad data.
To demonstrate ad-style investigations (CTR, CVR, CPC), this script generates 12 weeks of synthetic weekly performance for a
sample of real restaurants. Nothing here is an observation; every generated column is labelled SYNTHETIC in the registry and README.

Generating model (seed 42), per restaurant r and week t:
  placement_r        ~ Categorical(search .50, feed .35, banner .15)
  impressions_rt     ~ Poisson(300 * (1 + log1p(rating_count_r)/5) * placement_volume * season_t)
  CTR_r              = base_ctr(placement) * exp(0.6 * (rating_r - 4.0)) * cost_tier_factor      (unrated: rating treated as 3.6)
  clicks_rt          ~ Binomial(impressions_rt, CTR_r * noise_rt)
  orders_rt          ~ Binomial(clicks_rt, 0.12 * exp(0.5 * (rating_r - 4.0)))
  spend_rt           = clicks_rt * cpc(placement) * (1 + N(0, 0.1))
No anomalies are planted here: evaluation anomalies are injected separately by evaluation/scenarios.py.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from database.connection import execute, execute_script, query  # noqa: E402

CITIES = ["Bangalore", "Delhi", "Pune", "Hyderabad", "Chennai", "Kolkata", "Mumbai", "Ahmedabad"]
PER_CITY, WEEKS, SEED = 2500, 12, 42
PLACEMENT = dict(search=(0.50, 1.0, 0.045, 6.0), feed=(0.35, 1.6, 0.025, 4.0), banner=(0.15, 2.5, 0.012, 3.0))   # p, volume, base_ctr, cpc
COST = {"Budget": 1.10, "Mid": 1.00, "Premium": 0.90, "Luxury": 0.80}


def build():
    rng = np.random.default_rng(SEED)
    execute_script(config.ROOT / "database" / "schema.sql")
    r = query("SELECT id, city, rating, rating_count, cost_tier FROM restaurant_health_monitoring.restaurants WHERE city IN %(c)s", {"c": CITIES})
    r = r.groupby("city", group_keys=False).apply(lambda g: g.sample(min(PER_CITY, len(g)), random_state=SEED))
    names = list(PLACEMENT)
    r["placement"] = rng.choice(names, size=len(r), p=[PLACEMENT[n][0] for n in names])
    rating = r.rating.astype(float).fillna(3.6).values
    rc = r.rating_count.fillna(0).astype(float).values
    vol = np.array([PLACEMENT[p][1] for p in r.placement]); base = np.array([PLACEMENT[p][2] for p in r.placement]); cpc = np.array([PLACEMENT[p][3] for p in r.placement])
    ctr = base * np.exp(0.6 * (rating - 4.0)) * r.cost_tier.map(COST).fillna(1.0).values * np.where(r.rating.isna(), 0.8, 1.0)
    cvr = np.clip(0.12 * np.exp(0.5 * (rating - 4.0)), 0, 0.6)
    weeks = pd.date_range("2024-01-01", periods=WEEKS, freq="7D")
    season = 1 + 0.05 * np.sin(np.arange(WEEKS) / WEEKS * 2 * np.pi)
    rows = []
    for t, wk in enumerate(weeks):
        imp = rng.poisson(300 * (1 + np.log1p(rc) / 5) * vol * season[t])
        clk = rng.binomial(imp, np.clip(ctr * rng.normal(1, 0.05, len(r)), 0, 1))
        ords = rng.binomial(clk, cvr)
        spend = np.round(clk * cpc * (1 + rng.normal(0, 0.1, len(r))).clip(0.5), 2)
        rows += list(zip(r.id.astype(int), [wk.date()] * len(r), r.placement, imp.astype(int), clk.astype(int), ords.astype(int), spend.astype(float)))
    execute("DELETE FROM adlens_sponsored_weekly")
    sql = "INSERT INTO adlens_sponsored_weekly (restaurant_id, week_start, placement, impressions, clicks, orders, spend) VALUES (%s,%s,%s,%s,%s,%s,%s)"
    for i in range(0, len(rows), 20000):
        execute(sql, rows[i:i + 20000], many=True)
    print(f"inserted {len(rows):,} SYNTHETIC rows for {len(r):,} REAL restaurants in {len(CITIES)} cities, {WEEKS} weeks")


if __name__ == "__main__":
    build()
