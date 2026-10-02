"""Controlled database tools. The LLM chooses which to call; MySQL + pandas/NumPy do every calculation.

All tools take a dataset name, whitelisted metric/dimension names and ISO dates, and return small JSON-able dicts
(never raw tables). Statistical tests use permutation / normal approximations implemented with NumPy.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import pandas as pd

from database.connection import run_template
from tools import metric_tools as M

RNG_SEED = 7


def _rate(num, den):
    return float(num) / float(den) if den else float("nan")


def _r(x, k=4):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), k)


def _segments(dataset, metric, dim, start, end, filters):
    ids, params = M.identifiers(dataset, metric, dim, filters)
    df = run_template("segment_metrics", ids, dict(params, start=start, end=end))
    df["segment"] = df["segment"].astype(str)
    for c in ("num", "den", "n_rows"):
        df[c] = df[c].astype(float)
    return df


def _value(m, num, den):
    return float(num) if m["kind"] == "total" else _rate(num, den)


# ------------------------------------------------------------------ discovery
@lru_cache(maxsize=32)
def list_dimensions(dataset: str) -> dict:
    d = M.get(dataset)
    out = {}
    for dim, (_, prov) in d["dims"].items():
        ids = dict(table=d["table"], base=d["base"], dim=M.dim_expr(dataset, dim))
        vals = run_template("dimension_values", ids)
        out[dim] = dict(provenance=prov, n_values=int(len(vals)), top_values=vals.segment.astype(str).head(8).tolist())
    return dict(dataset=dataset, grain=d["grain"], entity_dimension=d["entity_dim"], dimensions=out,
                metrics={k: dict(desc=v["desc"], unit=v["unit"], worse_direction=v["worse"], provenance=v["prov"]) for k, v in d["metrics"].items()})


@lru_cache(maxsize=8)
def available_periods(dataset: str) -> list:
    d = M.get(dataset)
    df = run_template("periods", dict(table=d["table"], base=d["base"], period=d["period"]))
    return [str(p)[:10] for p in df.period]


# ------------------------------------------------------------------ metric tools
def get_metric_summary(dataset: str, metric: str, start: str, end: str, filters: dict | None = None) -> dict:
    m = M.metric(dataset, metric)
    ids, params = M.identifiers(dataset, metric, None, filters)
    r = run_template("metric_summary", ids, dict(params, start=start, end=end)).iloc[0]
    return dict(metric=metric, start=start, end=end, filters=filters or {}, value=_r(_value(m, r.num or 0, r.den or 0)), n_rows=int(r.n_rows))


def get_time_series(dataset: str, metric: str, start: str, end: str, filters: dict | None = None) -> dict:
    m = M.metric(dataset, metric)
    ids, params = M.identifiers(dataset, metric, None, filters)
    df = run_template("time_series", ids, dict(params, start=start, end=end))
    pts = [dict(period=str(p)[:10], value=_r(_value(m, n or 0, d or 0)), n_rows=int(k)) for p, n, d, k in df[["period", "num", "den", "n_rows"]].itertuples(index=False)]
    return dict(metric=metric, filters=filters or {}, points=pts)


def get_segment_metrics(dataset: str, metric: str, dimension: str, start: str, end: str, filters: dict | None = None, top: int = 15) -> dict:
    m = M.metric(dataset, metric)
    df = _segments(dataset, metric, dimension, start, end, filters)
    tot = df.den.sum()
    df["value"] = [_value(m, n, d) for n, d in zip(df.num, df.den)]
    df["share"] = df.den / tot if tot else np.nan
    df = df.sort_values("den", ascending=False).head(top)
    return dict(metric=metric, dimension=dimension, start=start, end=end,
                segments=[dict(segment=s, value=_r(v), share=_r(sh), n_rows=int(n)) for s, v, sh, n in df[["segment", "value", "share", "n_rows"]].itertuples(index=False)])


def compare_periods(dataset: str, metric: str, dimension: str, base_start: str, base_end: str, cur_start: str, cur_end: str,
                    filters: dict | None = None, top: int = 10) -> dict:
    """Shift-share decomposition of the metric change into mix (composition) and rate (within-segment) effects."""
    m = M.metric(dataset, metric)
    b = _segments(dataset, metric, dimension, base_start, base_end, filters).set_index("segment")
    c = _segments(dataset, metric, dimension, cur_start, cur_end, filters).set_index("segment")
    segs = sorted(set(b.index) | set(c.index))
    b, c = b.reindex(segs).fillna(0.0), c.reindex(segs).fillna(0.0)
    if m["kind"] == "total":
        base_v, cur_v = b.num.sum(), c.num.sum()
        rows = [dict(segment=s, base=_r(b.num[s]), current=_r(c.num[s]), contribution=_r(c.num[s] - b.num[s])) for s in segs]
        rows.sort(key=lambda r: -abs(r["contribution"] or 0))
        return dict(metric=metric, dimension=dimension, kind="total", base_value=_r(base_v), current_value=_r(cur_v),
                    delta=_r(cur_v - base_v), segments=rows[:top])
    wb, wc = b.den / b.den.sum(), c.den / c.den.sum()
    rb = pd.Series([_rate(n, d) for n, d in zip(b.num, b.den)], index=segs)
    rc = pd.Series([_rate(n, d) for n, d in zip(c.num, c.den)], index=segs)
    rb_f, rc_f = rb.fillna(rc), rc.fillna(rb)          # a segment missing in one period contributes only through mix
    base_v, cur_v = _rate(b.num.sum(), b.den.sum()), _rate(c.num.sum(), c.den.sum())
    mix = (wc - wb) * rb_f
    rate = wb * (rc_f - rb_f)
    inter = (wc - wb) * (rc_f - rb_f)
    rows = []
    for s in segs:
        rows.append(dict(segment=s, base_share=_r(wb[s]), current_share=_r(wc[s]), base_value=_r(rb[s]), current_value=_r(rc[s]),
                         mix_effect=_r(mix[s]), rate_effect=_r(rate[s]), interaction=_r(inter[s]),
                         contribution=_r(mix[s] + rate[s] + inter[s]), n_base=int(b.n_rows[s]), n_current=int(c.n_rows[s])))
    rows.sort(key=lambda r: -abs(r["contribution"] or 0))
    delta = cur_v - base_v
    return dict(metric=metric, dimension=dimension, kind="ratio", base_value=_r(base_v), current_value=_r(cur_v), delta=_r(delta),
                mix_effect_total=_r(mix.sum()), rate_effect_total=_r(rate.sum()), interaction_total=_r(inter.sum()),
                mix_share_of_delta=_r(mix.sum() / delta if delta else float("nan"), 3),
                rate_share_of_delta=_r(rate.sum() / delta if delta else float("nan"), 3), segments=rows[:top])


def scan_dimensions(dataset: str, metric: str, base_start: str, base_end: str, cur_start: str, cur_end: str, filters: dict | None = None) -> dict:
    """Screen every dimension: how much of the change is composition (mix) vs within-segment (rate), and the top segment."""
    out = []
    for dim in M.get(dataset)["dims"]:
        if filters and dim in filters:
            continue
        cp = compare_periods(dataset, metric, dim, base_start, base_end, cur_start, cur_end, filters, top=1)
        if cp.get("kind") == "total":
            top = cp["segments"][0] if cp["segments"] else {}
            out.append(dict(dimension=dim, top_segment=top.get("segment"), top_contribution=top.get("contribution")))
            continue
        top = cp["segments"][0] if cp["segments"] else {}
        out.append(dict(dimension=dim, mix_share_of_delta=cp["mix_share_of_delta"], rate_share_of_delta=cp["rate_share_of_delta"],
                        top_segment=top.get("segment"), top_segment_contribution=top.get("contribution"),
                        top_segment_mix=top.get("mix_effect"), top_segment_rate=top.get("rate_effect")))
    first = compare_periods(dataset, metric, next(iter(M.get(dataset)["dims"])), base_start, base_end, cur_start, cur_end, filters, top=0)
    out.sort(key=lambda r: -abs(r.get("top_segment_contribution") or r.get("top_contribution") or 0))
    return dict(metric=metric, base_value=first["base_value"], current_value=first["current_value"], delta=first["delta"], dimensions=out)


def get_entity_metrics(dataset: str, metric: str, base_start: str, base_end: str, cur_start: str, cur_end: str, top: int = 10) -> dict:
    d = M.get(dataset)
    return compare_periods(dataset, metric, d["entity_dim"], base_start, base_end, cur_start, cur_end, None, top)


def get_historical_metrics(dataset: str, metric: str, cur_start: str, cur_end: str, lookback: int = 14, filters: dict | None = None) -> dict:
    """Where the current period stands relative to the previous `lookback` periods (robust z-score)."""
    periods = available_periods(dataset)
    prior = [p for p in periods if p < cur_start][-lookback:]
    if not prior:
        return dict(error="no history before current period")
    ts = get_time_series(dataset, metric, prior[0], cur_end, filters)["points"]
    hist = np.array([p["value"] for p in ts if p["period"] < cur_start and p["value"] is not None], float)
    cur = get_metric_summary(dataset, metric, cur_start, cur_end, filters)["value"]
    med = float(np.median(hist)); mad = float(np.median(np.abs(hist - med))) * 1.4826 or float(np.std(hist)) or 1e-9
    return dict(metric=metric, current_value=cur, history_periods=len(hist), history_median=_r(med), history_mean=_r(hist.mean()),
                history_std=_r(hist.std()), robust_z=_r((cur - med) / mad, 2), history_min=_r(hist.min()), history_max=_r(hist.max()))


# ------------------------------------------------------------------ statistics
def _rows(dataset, metric, start, end, filters, dim=None):
    d = M.get(dataset)
    ids, params = M.identifiers(dataset, metric, dim or next(iter(d["dims"])), filters)
    df = run_template("row_values", ids, dict(params, start=start, end=end))
    return df.astype({"num": float, "den": float})


def run_statistical_test(dataset: str, metric: str, base_start: str, base_end: str, cur_start: str, cur_end: str,
                         dimension: str | None = None, segment: str | None = None, test: str = "rate", n_perm: int = 2000) -> dict:
    """test='rate': permutation test that the metric (within `segment` if given) differs between periods.
    test='share': two-proportion z-test that the segment's share of volume differs between periods."""
    m = M.metric(dataset, metric)
    filters = {dimension: segment} if (dimension and segment is not None and test == "rate") else None
    if test == "share":
        if not (dimension and segment is not None):
            return dict(error="share test needs dimension and segment")
        b = _segments(dataset, metric, dimension, base_start, base_end, None)
        c = _segments(dataset, metric, dimension, cur_start, cur_end, None)
        x0 = float(b.loc[b.segment == str(segment), "den"].sum()); n0 = float(b.den.sum())
        x1 = float(c.loc[c.segment == str(segment), "den"].sum()); n1 = float(c.den.sum())
        p0, p1 = x0 / n0, x1 / n1
        pp = (x0 + x1) / (n0 + n1)
        se = math.sqrt(pp * (1 - pp) * (1 / n0 + 1 / n1)) or 1e-12
        z = (p1 - p0) / se
        return dict(test="two-proportion z", dimension=dimension, segment=segment, base_share=_r(p0), current_share=_r(p1),
                    effect=_r(p1 - p0), z=_r(z, 2), p_value=_r(math.erfc(abs(z) / math.sqrt(2)), 6), n_base=int(n0), n_current=int(n1))
    if m["kind"] == "total":
        a = get_metric_summary(dataset, metric, base_start, base_end, filters)["value"] or 0
        b = get_metric_summary(dataset, metric, cur_start, cur_end, filters)["value"] or 0
        z = (b - a) / math.sqrt(max(a + b, 1))
        return dict(test="poisson z", effect=_r(b - a), z=_r(z, 2), p_value=_r(math.erfc(abs(z) / math.sqrt(2)), 6))
    x = _rows(dataset, metric, base_start, base_end, filters)
    y = _rows(dataset, metric, cur_start, cur_end, filters)
    x, y = x[x.den > 0], y[y.den > 0]
    if len(x) < 5 or len(y) < 5:
        return dict(test="permutation", error="too few rows", n_base=len(x), n_current=len(y))
    num = np.r_[x.num.values, y.num.values]; den = np.r_[x.den.values, y.den.values]
    nb = len(x)
    obs = num[nb:].sum() / den[nb:].sum() - num[:nb].sum() / den[:nb].sum()
    rng = np.random.default_rng(RNG_SEED)
    cnt = 0
    for _ in range(n_perm):
        idx = rng.permutation(len(num))
        a, b = idx[:nb], idx[nb:]
        cnt += abs(num[b].sum() / den[b].sum() - num[a].sum() / den[a].sum()) >= abs(obs)
    boots = []
    for _ in range(400):
        ia = rng.integers(0, nb, nb); ib = rng.integers(nb, len(num), len(num) - nb)
        boots.append(num[ib].sum() / den[ib].sum() - num[ia].sum() / den[ia].sum())
    return dict(test="permutation", dimension=dimension, segment=segment, base_value=_r(num[:nb].sum() / den[:nb].sum()),
                current_value=_r(num[nb:].sum() / den[nb:].sum()), effect=_r(obs), ci95=[_r(np.percentile(boots, 2.5)), _r(np.percentile(boots, 97.5))],
                p_value=_r((cnt + 1) / (n_perm + 1), 5), n_base=int(nb), n_current=int(len(num) - nb))


def model_rows(dataset: str, metric: str, start: str, end: str, dims: list[str], filters: dict | None = None) -> pd.DataFrame:
    d = M.get(dataset)
    ids, params = M.identifiers(dataset, metric, None, filters)
    ids["dim_list"] = ", ".join(f"{M.dim_expr(dataset, k)} AS `{k}`" for k in dims)
    df = run_template("model_rows", ids, dict(params, start=start, end=end))
    return df.astype({"y": float, "w": float})
