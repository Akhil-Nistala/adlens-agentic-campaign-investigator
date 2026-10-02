"""Evaluation on REAL orders with controlled injected anomalies.

  python evaluation/evaluate.py detection            # deterministic monitor: precision / recall over many injected + clean days
  python evaluation/evaluate.py investigation [N]    # AdLens vs one-prompt baseline on N scenarios (uses the LLM; costs money)

Investigation scoring
  correct_cause        top-ranked SUPPORTED/WEAKLY_SUPPORTED hypothesis matches the injected (type, dimension[, segment])
  cause_supported      the injected cause appears among SUPPORTED/WEAKLY_SUPPORTED hypotheses at all
  false_rejected       share of non-matching hypotheses that ended REJECTED or INCONCLUSIVE (not falsely supported)
  evidence_quality     share of supported hypotheses backed by p < 0.01 and explained >= 0.30
  counterfactual_error |counterfactual metric for the true cause - true clean-day metric| (minutes)
  efficiency           tool calls, failed tool calls, graph steps, LLM calls, tokens, cost
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import uuid

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from agents import monitor  # noqa: E402
from database.connection import execute  # noqa: E402
from evaluation import scenarios as SC  # noqa: E402
from tools import simulation_tools as SIM  # noqa: E402

DS, METRIC = "orders_eval", "avg_delivery_time"


def detection(days=None):
    days = days or SC.SHORT_DAYS[1:]
    rows = []
    for day in days:
        for name in SC.SCENARIOS:
            gt = SC.inject(name, day)
            obs = monitor.detect(DS, METRIC, gt["baseline_day"], gt["baseline_day"], day, day)
            rows.append(dict(day=day, scenario=name, injected=name != "none", flagged=obs["anomalous"], delta=obs["delta"],
                             rel=obs["relative_change"], p=obs["test"].get("p_value")))
            print(day, name, obs["anomalous"], obs["delta"], flush=True)
    tp = sum(r["injected"] and r["flagged"] for r in rows); fp = sum((not r["injected"]) and r["flagged"] for r in rows)
    fn = sum(r["injected"] and not r["flagged"] for r in rows); tn = sum((not r["injected"]) and not r["flagged"] for r in rows)
    by = {}
    for r in rows:
        b = by.setdefault(r["scenario"], dict(n=0, flagged=0)); b["n"] += 1; b["flagged"] += r["flagged"]
    res = dict(trials=len(rows), tp=tp, fp=fp, fn=fn, tn=tn, precision=tp / (tp + fp) if tp + fp else None, recall=tp / (tp + fn) if tp + fn else None,
               false_positive_rate=fp / (fp + tn) if fp + tn else None, by_scenario=by, rule=f"|rel change| >= {monitor.MIN_REL_CHANGE:.0%} and p < {monitor.ALPHA}",
               rows=rows)
    (config.RESULTS / "eval_detection.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print({k: v for k, v in res.items() if k != "rows"})
    return res


def _match(h, gt, need_segment):
    if h["type"] != gt["type"] or h["dimension"] != gt["dimension"]:
        return False
    if gt["type"] == "MIX_SHIFT" and gt["dimension"] == "multiple_deliveries":
        return True                                   # injected rows include 2 and 3 deliveries; the dimension is the cause
    return (not need_segment) or str(h.get("segment")) == gt["segment"]


def log_run(system, gt, result, usage, tool_calls=0, failed=0, steps=0):
    execute("INSERT INTO adlens_runs (run_id, system_name, dataset, metric, scenario, base_start, base_end, cur_start, cur_end, tool_calls, "
            "failed_tool_calls, steps, llm_calls, input_tokens, output_tokens, cost_usd, result_json) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (uuid.uuid4().hex[:16], system, DS, METRIC, gt["scenario"], gt["baseline_day"], gt["baseline_day"], gt["day"], gt["day"], tool_calls, failed, steps,
             usage["llm_calls"], usage["input_tokens"], usage["output_tokens"], usage["cost_usd"], json.dumps(result, default=str)[:60000]))


def investigation(n=8):
    from agents.graph import investigate
    from evaluation.baseline import diagnose
    plan = [(name, SC.SHORT_DAYS[3 + 2 * i % (len(SC.SHORT_DAYS) - 3)]) for i, name in enumerate([s for s in SC.SCENARIOS if s != "none"])][:n]
    ckpt = config.RESULTS / "eval_investigation_rows.jsonl"       # one line per (system, scenario); reruns resume
    out = [json.loads(l) for l in ckpt.read_text(encoding="utf-8").splitlines()] if ckpt.exists() else []
    done = {r["scenario"] for r in out if r["system"] == "baseline"}

    def keep(row):
        out.append(row)
        with open(ckpt, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")

    for name, day in plan:
        if name in done:
            continue
        out = [r for r in out if r["scenario"] != name]            # drop a half-finished scenario
        gt = SC.inject(name, day)
        clean = SC.clean_value("AVG(time_taken_min)", day)
        t0 = time.time()
        s = investigate(DS, METRIC, gt["baseline_day"], gt["baseline_day"], day, day)
        vals = s.get("validations", [])
        sup = [v for v in vals if v["status"] in ("SUPPORTED", "WEAKLY_SUPPORTED")]
        top = sup[0] if sup else None
        wrong = [v for v in vals if not _match(v, gt, True)]
        cf_true = SIM.simulate_segment_rate(DS, METRIC, gt["dimension"], gt["segment"], gt["baseline_day"], gt["baseline_day"], day, day) \
            if gt["type"] == "RATE_CHANGE" else SIM.simulate_mix(DS, METRIC, gt["dimension"], gt["baseline_day"], gt["baseline_day"], day, day)
        agent_cf = next((c for c in s.get("counterfactuals", []) if top and c.get("hypothesis") == top["id"] and not c.get("error")), None)
        trace = s["trace"]
        a = dict(system="adlens", scenario=name, day=day, truth=gt, flagged=s.get("anomalous"),
                 correct_cause=bool(top and _match(top, gt, True)), cause_supported=any(_match(v, gt, True) for v in sup),
                 top_hypothesis=top and {k: top[k] for k in ("type", "dimension", "segment", "status", "confidence")},
                 n_hypotheses=len(vals), false_rejected=(sum(v["status"] in ("REJECTED", "INCONCLUSIVE") for v in wrong) / len(wrong)) if wrong else None,
                 evidence_quality=(sum((v["p_value"] or 1) < 0.01 and (v["explained"] or 0) >= 0.3 for v in sup) / len(sup)) if sup else None,
                 clean_value=clean, observed=s["observation"]["current_value"],
                 counterfactual_true_cause=cf_true.get("counterfactual_metric"), counterfactual_error_true_cause=abs((cf_true.get("counterfactual_metric") or 0) - clean),
                 agent_counterfactual=agent_cf and agent_cf.get("counterfactual_metric"),
                 agent_counterfactual_error=agent_cf and abs((agent_cf.get("counterfactual_metric") or 0) - clean),
                 tool_calls=trace["tool_calls"], failed_tool_calls=trace["failed_tool_calls"], steps=len(trace["steps"]), seconds=round(time.time() - t0, 1),
                 **trace.get("llm", {}))
        keep(a)
        log_run("adlens", gt, dict(validations=vals, recommendation=s.get("recommendation")), trace.get("llm", {}), trace["tool_calls"], trace["failed_tool_calls"], len(trace["steps"]))
        (config.RESULTS / f"report_{name}_{day}.txt").write_text(s["report"], encoding="utf-8")
        t0 = time.time()
        b = diagnose(DS, METRIC, (gt["baseline_day"], gt["baseline_day"]), (day, day))
        bd = b["diagnosis"]
        keep(dict(system="baseline", scenario=name, day=day, truth=gt, correct_cause=_match(dict(type=bd["cause_type"], dimension=bd["dimension"], segment=bd["segment"]), gt, True),
                  top_hypothesis=bd, tool_calls=b["queries"], seconds=round(time.time() - t0, 1), **b["usage"]))
        log_run("baseline", gt, bd, b["usage"], b["queries"])
        print(name, day, "adlens:", a["correct_cause"], a["top_hypothesis"], "| baseline:", out[-1]["correct_cause"], bd["cause_type"], bd["dimension"], bd["segment"], flush=True)
    summary = {}
    for system in ("adlens", "baseline"):
        rs = [r for r in out if r["system"] == system]
        avg = lambda k: round(sum(r.get(k) or 0 for r in rs) / len(rs), 2) if rs else None  # noqa: E731
        summary[system] = dict(n=len(rs), correct_cause=sum(r["correct_cause"] for r in rs), cause_supported=sum(r.get("cause_supported", False) for r in rs),
                               avg_tool_calls=avg("tool_calls"), avg_failed_tool_calls=avg("failed_tool_calls"), avg_steps=avg("steps"),
                               avg_llm_calls=avg("llm_calls"), avg_input_tokens=avg("input_tokens"), avg_output_tokens=avg("output_tokens"),
                               avg_seconds=avg("seconds"), total_cost_usd=round(sum(r.get("cost_usd") or 0 for r in rs), 4), model=config.MODEL)
    print(summary)
    (config.RESULTS / "eval_investigation.json").write_text(json.dumps(dict(summary=summary, rows=out), indent=1, default=str), encoding="utf-8")
    return out


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "detection"
    if what == "detection":
        detection()
    else:
        investigation(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
