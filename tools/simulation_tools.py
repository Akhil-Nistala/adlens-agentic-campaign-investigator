"""Deterministic counterfactual simulations. The LLM interprets these numbers; it never produces them.

simulate_segment_rate  - "what if segment S had kept its baseline value (or a given value)?"
simulate_mix           - "what if the composition of dimension D had stayed as in the baseline (or given shares)?"
simulate_composition_model - fit a scikit-learn model on baseline rows, predict the current rows:
                         how much of the change does the joint composition of several dimensions explain?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from tools import database_tools as T
from tools import metric_tools as M


def _r(x, k=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


def _tables(dataset, metric, dimension, base, cur, filters=None):
    b = T._segments(dataset, metric, dimension, base[0], base[1], filters).set_index("segment")
    c = T._segments(dataset, metric, dimension, cur[0], cur[1], filters).set_index("segment")
    segs = sorted(set(b.index) | set(c.index))
    return b.reindex(segs).fillna(0.0), c.reindex(segs).fillna(0.0), segs


def simulate_segment_rate(dataset: str, metric: str, dimension: str, segment: str, base_start: str, base_end: str, cur_start: str, cur_end: str,
                          new_value: float | None = None) -> dict:
    """Current-period metric if `segment` had its baseline value (or `new_value`), all else as observed."""
    m = M.metric(dataset, metric)
    if m["kind"] == "total":
        return dict(error="segment-rate simulation applies to ratio metrics")
    b, c, segs = _tables(dataset, metric, dimension, (base_start, base_end), (cur_start, cur_end))
    segment = str(segment)
    if segment not in segs or c.loc[segment, "den"] == 0:
        return dict(error=f"segment {segment} has no current-period volume")
    target = new_value if new_value is not None else (b.loc[segment, "num"] / b.loc[segment, "den"] if b.loc[segment, "den"] else None)
    if target is None:
        return dict(error=f"segment {segment} has no baseline volume, so no baseline value to restore")
    observed = c.num.sum() / c.den.sum()
    num_cf = c.num.sum() - c.loc[segment, "num"] + target * c.loc[segment, "den"]
    cf = num_cf / c.den.sum()
    base_v = b.num.sum() / b.den.sum()
    return dict(simulation="segment_rate", dimension=dimension, segment=segment, segment_value_set_to=_r(target),
                baseline_metric=_r(base_v), observed_metric=_r(observed), counterfactual_metric=_r(cf),
                change_removed=_r(observed - cf), share_of_change_removed=_r((observed - cf) / (observed - base_v), 3) if observed != base_v else None)


def simulate_mix(dataset: str, metric: str, dimension: str, base_start: str, base_end: str, cur_start: str, cur_end: str,
                 target_shares: dict | None = None) -> dict:
    """Current-period metric re-weighted to baseline composition of `dimension` (or to `target_shares`)."""
    m = M.metric(dataset, metric)
    if m["kind"] == "total":
        return dict(error="mix simulation applies to ratio metrics")
    b, c, segs = _tables(dataset, metric, dimension, (base_start, base_end), (cur_start, cur_end))
    rc = (c.num / c.den.replace(0, np.nan))
    rb = (b.num / b.den.replace(0, np.nan))
    if target_shares:
        w = pd.Series({s: float(target_shares.get(s, 0.0)) for s in segs}); w = w / w.sum()
    else:
        w = b.den / b.den.sum()
    missing = [s for s in segs if w[s] > 0 and np.isnan(rc[s])]
    # segments the target composition needs but that have no current-period data: use their baseline value, and say so
    rate = rc.copy()
    for s in missing:
        rate[s] = rb[s]
    covered = float(w[[s for s in segs if not np.isnan(rc[s])]].sum())
    observed = c.num.sum() / c.den.sum()
    cf = float((w * rate).sum())
    base_v = b.num.sum() / b.den.sum()
    return dict(simulation="mix", dimension=dimension, target="baseline composition" if not target_shares else "given shares",
                baseline_metric=_r(base_v), observed_metric=_r(observed), counterfactual_metric=_r(cf), change_removed=_r(observed - cf),
                share_of_change_removed=_r((observed - cf) / (observed - base_v), 3) if observed != base_v else None,
                target_weight_with_current_data=_r(covered, 3),
                caveat=(f"No current-period data for {missing}; their baseline values were used, so this counterfactual "
                        "partly assumes the answer." if missing else None))


def simulate_composition_model(dataset: str, metric: str, dims: list[str], base_start: str, base_end: str, cur_start: str, cur_end: str) -> dict:
    """Fit Ridge(one-hot dims) on BASELINE rows; predict CURRENT rows. If the prediction tracks the observed current value,
    the change is explained by composition across `dims`; the remainder is change in behaviour within segments."""
    m = M.metric(dataset, metric)
    if m["kind"] == "total":
        return dict(error="composition model applies to ratio metrics")
    for d in dims:
        M.dim_expr(dataset, d)
    tb = T.model_rows(dataset, metric, base_start, base_end, dims)
    tc = T.model_rows(dataset, metric, cur_start, cur_end, dims)
    tb, tc = tb[tb.w > 0], tc[tc.w > 0]
    if len(tb) < 30 or len(tc) < 30:
        return dict(error="too few rows for a model")
    model = make_pipeline(ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"), dims)]), Ridge(alpha=1.0))
    model.fit(tb[dims].astype(str), tb.y / tb.w, ridge__sample_weight=tb.w)
    pred_cur = model.predict(tc[dims].astype(str))
    expected = float(np.average(pred_cur, weights=tc.w))
    base_v = float(tb.y.sum() / tb.w.sum()); observed = float(tc.y.sum() / tc.w.sum())
    unseen = {d: sorted(set(tc[d].astype(str)) - set(tb[d].astype(str))) for d in dims}
    unseen_share = float(np.mean(np.any([~tc[d].astype(str).isin(set(tb[d].astype(str))) for d in dims], axis=0)))
    return dict(simulation="composition_model", model="Ridge on one-hot " + ", ".join(dims), baseline_metric=_r(base_v), observed_metric=_r(observed),
                expected_from_composition=_r(expected), explained_share=_r((expected - base_v) / (observed - base_v), 3) if observed != base_v else None,
                current_rows_with_unseen_categories=_r(unseen_share, 3), unseen_categories={k: v for k, v in unseen.items() if v},
                caveat=("Some current-period categories never appear in the baseline; the model cannot attribute their effect "
                        "(it treats them as 'average'), so composition explained may be understated.") if unseen_share > 0.05 else None)
