"""AdLens entry point.

Interactive:      python main.py
Non-interactive:  python main.py --dataset orders --metric avg_delivery_time --base 2022-03-13 2022-03-13 --current 2022-03-14 2022-03-14
                  python main.py --dataset restaurant_ads --metric ctr --entity Bangalore --base 2024-01-01 2024-01-07 --current 2024-01-08 2024-01-14
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import config
from agents.graph import investigate
from tools import database_tools as T
from tools import metric_tools as M

SELECTABLE = ["orders", "restaurant_ads"]


def ask(prompt, options=None, default=None):
    if options:
        for i, o in enumerate(options, 1):
            print(f"  {i}. {o}")
    v = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    if options and v.isdigit() and 1 <= int(v) <= len(options):
        return options[int(v) - 1]
    return v or default


def interactive():
    print("ADLENS: select dataset")
    ds = ask("Dataset", [f"{k}: {M.get(k)['label']}" for k in SELECTABLE], "1").split(":")[0]
    ds = ds if ds in SELECTABLE else SELECTABLE[0]
    metrics = list(M.get(ds)["metrics"])
    metric = ask("Metric", metrics, metrics[0])
    ent_dim = M.get(ds)["entity_dim"]
    ents = ["ALL"] + T.list_dimensions(ds)["dimensions"][ent_dim]["top_values"]
    ent = ask(f"Entity/campaign ({ent_dim})", ents, "ALL")
    periods = T.available_periods(ds)
    print(f"Available {M.get(ds)['grain']}s: {periods[0]} .. {periods[-1]} ({len(periods)})")
    cur = ask("Current period start (YYYY-MM-DD)", default=periods[-1])
    cur_end = ask("Current period end", default=cur)
    prev = [p for p in periods if p < cur]
    base = ask("Baseline start", default=prev[-1] if prev else periods[0])
    base_end = ask("Baseline end", default=base)
    return ds, metric, ({ent_dim: ent} if ent not in ("ALL", None, "") else {}), base, base_end, cur, cur_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=SELECTABLE)
    ap.add_argument("--metric")
    ap.add_argument("--entity", help="value of the dataset's entity dimension (city_type / city)")
    ap.add_argument("--base", nargs=2)
    ap.add_argument("--current", nargs=2)
    a = ap.parse_args()
    if a.dataset:
        ent = {M.get(a.dataset)["entity_dim"]: a.entity} if a.entity else {}
        args = (a.dataset, a.metric or next(iter(M.get(a.dataset)["metrics"])), ent, *a.base, *a.current)
    else:
        args = interactive()
    ds, metric, ent, b0, b1, c0, c1 = args
    s = investigate(ds, metric, b0, b1, c0, c1, ent)
    print("\n" + s["report"])
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    (config.RESULTS / f"investigation_{ds}_{metric}_{stamp}.txt").write_text(s["report"], encoding="utf-8")
    keep = {k: v for k, v in s.items() if k not in ("usage",)}
    (config.RESULTS / f"investigation_{ds}_{metric}_{stamp}.json").write_text(json.dumps(keep, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
