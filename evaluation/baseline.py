"""Baseline: put all relevant dataset statistics into ONE LLM prompt and ask for a diagnosis.

The statistics are the same per-dimension segment tables (volume share + metric value, both periods) that AdLens's tools can
compute, so the comparison isolates the effect of the agentic structure (tool selection, validation, counterfactuals).
"""
from __future__ import annotations

import json

from agents import llm
from tools import database_tools as T
from tools import metric_tools as M

SYSTEM = "You are a data analyst. Diagnose why the metric changed between the two periods using only the statistics provided."
SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["cause_type", "dimension", "segment", "explanation", "confidence"],
    "properties": {"cause_type": {"type": "string", "enum": ["RATE_CHANGE", "MIX_SHIFT", "BROAD_SHIFT", "OTHER"]},
                   "dimension": {"type": "string"}, "segment": {"type": "string"}, "explanation": {"type": "string"},
                   "confidence": {"type": "number"}},
}


def stats_bundle(dataset, metric, base, cur, entity_filter=None) -> dict:
    out = {"overall": dict(baseline=T.get_metric_summary(dataset, metric, *base, entity_filter),
                           current=T.get_metric_summary(dataset, metric, *cur, entity_filter)), "by_dimension": {}}
    for dim in M.get(dataset)["dims"]:
        out["by_dimension"][dim] = dict(baseline=T.get_segment_metrics(dataset, metric, dim, *base, entity_filter)["segments"],
                                        current=T.get_segment_metrics(dataset, metric, dim, *cur, entity_filter)["segments"])
    return out


def diagnose(dataset, metric, base, cur, entity_filter=None) -> dict:
    usage = llm.Usage()
    stats = stats_bundle(dataset, metric, base, cur, entity_filter)
    d = M.get(dataset)
    prompt = (f"Dataset: {d['label']}. Metric: {metric} ({M.metric(dataset, metric)['desc']}). Baseline {base[0]}..{base[1]}, current {cur[0]}..{cur[1]}.\n"
              f"Statistics (value = metric, share = share of volume) per dimension segment:\n{json.dumps(stats, default=str)}\n\n"
              "Identify the single most likely cause: its type (RATE_CHANGE = one segment's own value changed; MIX_SHIFT = composition "
              "changed; BROAD_SHIFT = everything moved; OTHER), the dimension and the segment (exact value), and explain.")
    out = llm.structured(usage, SYSTEM, prompt, SCHEMA)
    return dict(diagnosis=out, usage=usage.to_dict(), queries=1 + 2 * len(d["dims"]) * 1)
