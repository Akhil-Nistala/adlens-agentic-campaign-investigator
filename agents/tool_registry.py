"""Tools exposed to the LLM (JSON schemas) and a safe executor.

The dataset is fixed by the investigation (the model can't switch tables), period dates default to the investigation
periods, filters always include the investigation's entity filter, and every identifier is validated against the
dataset registry before any SQL runs. Failed calls come back to the model as is_error tool results.
"""
from __future__ import annotations

import json

from tools import database_tools as T
from tools import simulation_tools as SIM

PERIODS = {"base_start": {"type": "string", "description": "baseline start date YYYY-MM-DD (defaults to the investigation baseline)"},
           "base_end": {"type": "string"}, "cur_start": {"type": "string", "description": "current period start (defaults to investigation)"},
           "cur_end": {"type": "string"}}
FILTERS = {"filters": {"type": "array", "description": "optional extra segment filters",
                       "items": {"type": "object", "properties": {"dimension": {"type": "string"}, "value": {"type": "string"}},
                                 "required": ["dimension", "value"]}}}


def _tool(name, description, props, required):
    return {"name": name, "description": description, "input_schema": {"type": "object", "properties": props, "required": required}}


TOOLS = [
    _tool("get_metric_summary", "Metric value and row count for one date range (optionally filtered).",
          {"metric": {"type": "string"}, "start": {"type": "string"}, "end": {"type": "string"}, **FILTERS}, ["metric", "start", "end"]),
    _tool("get_time_series", "Metric value per period (day or week) for a date range.",
          {"metric": {"type": "string"}, "start": {"type": "string"}, "end": {"type": "string"}, **FILTERS}, ["metric", "start", "end"]),
    _tool("get_segment_metrics", "Metric value and volume share for each value of one dimension in a date range.",
          {"metric": {"type": "string"}, "dimension": {"type": "string"}, "start": {"type": "string"}, "end": {"type": "string"}, **FILTERS},
          ["metric", "dimension", "start", "end"]),
    _tool("compare_periods", "Shift-share decomposition of the metric change between baseline and current period along one dimension: "
          "per segment shares, values, mix effect (composition), rate effect (within segment) and contribution.",
          {"metric": {"type": "string"}, "dimension": {"type": "string"}, **PERIODS, **FILTERS}, ["metric", "dimension"]),
    _tool("scan_dimensions", "Screen ALL dimensions at once: mix vs rate share of the change and the top contributing segment for each. "
          "A cheap first look before drilling down.", {"metric": {"type": "string"}, **PERIODS}, ["metric"]),
    _tool("get_entity_metrics", "compare_periods on the dataset's entity dimension (city type / city).", {"metric": {"type": "string"}, **PERIODS}, ["metric"]),
    _tool("get_historical_metrics", "How unusual the current period is versus the previous N periods (robust z-score).",
          {"metric": {"type": "string"}, "lookback": {"type": "integer"}, **PERIODS}, ["metric"]),
    _tool("run_statistical_test", "test='rate': permutation test that the metric (inside dimension=segment if given) changed between periods. "
          "test='share': z-test that a segment's share of volume changed.",
          {"metric": {"type": "string"}, "test": {"type": "string", "enum": ["rate", "share"]}, "dimension": {"type": "string"},
           "segment": {"type": "string"}, **PERIODS}, ["metric", "test"]),
    _tool("simulate_segment_rate", "Counterfactual: current metric if one segment had kept its baseline value (or new_value).",
          {"metric": {"type": "string"}, "dimension": {"type": "string"}, "segment": {"type": "string"}, "new_value": {"type": "number"}, **PERIODS},
          ["metric", "dimension", "segment"]),
    _tool("simulate_mix", "Counterfactual: current metric if one dimension had the baseline composition.",
          {"metric": {"type": "string"}, "dimension": {"type": "string"}, **PERIODS}, ["metric", "dimension"]),
    _tool("simulate_composition_model", "Counterfactual (scikit-learn): train on baseline rows, predict current rows from several dimensions jointly; "
          "share of the change explained by composition.", {"metric": {"type": "string"}, "dims": {"type": "array", "items": {"type": "string"}}, **PERIODS},
          ["metric", "dims"]),
]


def execute(name: str, args: dict, state: dict) -> dict:
    ds = state["dataset"]
    per = {k: args.get(k) or state[k] for k in ("base_start", "base_end", "cur_start", "cur_end")}
    filt = dict(state.get("entity_filter") or {})
    for f in args.get("filters") or []:
        filt[f["dimension"]] = f["value"]
    filt = filt or None
    m = args.get("metric", state["metric"])
    if name == "get_metric_summary":
        return T.get_metric_summary(ds, m, args["start"], args["end"], filt)
    if name == "get_time_series":
        return T.get_time_series(ds, m, args["start"], args["end"], filt)
    if name == "get_segment_metrics":
        return T.get_segment_metrics(ds, m, args["dimension"], args["start"], args["end"], filt)
    if name == "compare_periods":
        return T.compare_periods(ds, m, args["dimension"], per["base_start"], per["base_end"], per["cur_start"], per["cur_end"], filt)
    if name == "scan_dimensions":
        return T.scan_dimensions(ds, m, per["base_start"], per["base_end"], per["cur_start"], per["cur_end"], filt)
    if name == "get_entity_metrics":
        return T.get_entity_metrics(ds, m, per["base_start"], per["base_end"], per["cur_start"], per["cur_end"])
    if name == "get_historical_metrics":
        return T.get_historical_metrics(ds, m, per["cur_start"], per["cur_end"], int(args.get("lookback") or 14), filt)
    if name == "run_statistical_test":
        return T.run_statistical_test(ds, m, per["base_start"], per["base_end"], per["cur_start"], per["cur_end"],
                                      args.get("dimension"), args.get("segment"), args.get("test", "rate"))
    if name == "simulate_segment_rate":
        return SIM.simulate_segment_rate(ds, m, args["dimension"], args["segment"], per["base_start"], per["base_end"], per["cur_start"], per["cur_end"],
                                         args.get("new_value"))
    if name == "simulate_mix":
        return SIM.simulate_mix(ds, m, args["dimension"], per["base_start"], per["base_end"], per["cur_start"], per["cur_end"])
    if name == "simulate_composition_model":
        return SIM.simulate_composition_model(ds, m, list(args["dims"]), per["base_start"], per["base_end"], per["cur_start"], per["cur_end"])
    raise ValueError(f"unknown tool {name}")


def compact(obj, limit: int = 3500) -> str:
    s = json.dumps(obj, default=str)
    return s if len(s) <= limit else s[:limit] + ' ..."(truncated)"'
