"""Monitor agent (deterministic): is there a real change worth investigating?

Flags an anomaly when the metric moved by >= MIN_REL_CHANGE relative to the baseline AND a permutation test
(or Poisson test for counts) gives p < ALPHA. The robust z-score versus recent history is reported as context.
No LLM is involved: detection is a calculation.
"""
from tools import database_tools as T
from tools import metric_tools as M

MIN_REL_CHANGE = 0.05
ALPHA = 0.01


def detect(dataset, metric, base_start, base_end, cur_start, cur_end, entity_filter=None) -> dict:
    m = M.metric(dataset, metric)
    b = T.get_metric_summary(dataset, metric, base_start, base_end, entity_filter)
    c = T.get_metric_summary(dataset, metric, cur_start, cur_end, entity_filter)
    if b["value"] is None or c["value"] is None:
        return dict(error="no data in one of the periods", baseline=b, current=c)
    dim, seg = (next(iter(entity_filter.items())) if entity_filter else (None, None))
    test = T.run_statistical_test(dataset, metric, base_start, base_end, cur_start, cur_end, dim, seg, "rate", n_perm=1000)
    hist = T.get_historical_metrics(dataset, metric, cur_start, cur_end, 14, entity_filter)
    delta = c["value"] - b["value"]
    rel = delta / b["value"] if b["value"] else float("inf")
    worse = (delta > 0) == (m["worse"] == "up")
    p = test.get("p_value", 1.0)
    return dict(metric=metric, metric_desc=m["desc"], unit=m["unit"], metric_provenance=m["prov"], baseline_value=b["value"],
                current_value=c["value"], delta=round(delta, 4), relative_change=round(rel, 4), direction="worse" if worse else "better",
                n_baseline=b["n_rows"], n_current=c["n_rows"], test=test, history=hist,
                anomalous=bool(abs(rel) >= MIN_REL_CHANGE and p is not None and p < ALPHA),
                rule=f"|relative change| >= {MIN_REL_CHANGE:.0%} and p < {ALPHA}")


def node(state):
    obs = detect(state["dataset"], state["metric"], state["base_start"], state["base_end"], state["cur_start"], state["cur_end"],
                 state.get("entity_filter") or None)
    trace = state.get("trace") or dict(tool_calls=0, failed_tool_calls=0, steps=[])
    trace["tool_calls"] += 4
    trace["steps"].append(dict(node="monitor", anomalous=obs.get("anomalous"), delta=obs.get("delta")))
    return dict(observation=obs, anomalous=bool(obs.get("anomalous")), trace=trace)
