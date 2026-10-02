"""Shared LangGraph state for one investigation."""
from __future__ import annotations

from typing import Any, TypedDict


class Trace(TypedDict, total=False):
    tool_calls: int
    failed_tool_calls: int
    steps: list
    llm: dict


class InvestigationState(TypedDict, total=False):
    dataset: str
    metric: str
    entity_filter: dict              # e.g. {"city_type": "Urban"}; {} = whole dataset
    base_start: str
    base_end: str
    cur_start: str
    cur_end: str
    observation: dict                # monitor output
    anomalous: bool
    dimensions: dict                 # list_dimensions() output given to the planner
    plan: dict
    evidence: list                   # [{id, tool, input, output}]
    evidence_summary: str
    hypotheses: list
    validations: list
    counterfactuals: list
    recommendation: dict
    report: str
    trace: dict
    usage: Any                       # agents.llm.Usage (not serialised)
