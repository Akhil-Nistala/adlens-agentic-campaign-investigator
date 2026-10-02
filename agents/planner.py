"""Investigation planner (LLM): turns the observed change + available dimensions into an ordered plan of tool calls."""
import json

from agents import llm
from tools import database_tools as T
from tools import metric_tools as M

SYSTEM = (
    "You are the planning step of AdLens, an analytics agent that investigates why a performance metric changed. "
    "You do not compute anything yourself: SQL/Python tools do all calculations. Plan which tools to call, in what order, "
    "to explain the change. Prefer a cheap screen first (scan_dimensions), then drill into the most promising dimensions with "
    "compare_periods and statistical tests. Be honest about the data: if a field is SYNTHETIC or DERIVED, keep that in mind; "
    "never assume data that does not exist."
)

SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["objective", "steps", "dimensions_to_check"],
    "properties": {
        "objective": {"type": "string"},
        "dimensions_to_check": {"type": "array", "items": {"type": "string"}},
        "steps": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["step", "tool", "purpose"],
            "properties": {"step": {"type": "integer"}, "tool": {"type": "string"}, "purpose": {"type": "string"}}}},
    },
}


def node(state):
    dims = T.list_dimensions(state["dataset"])
    d = M.get(state["dataset"])
    prompt = (
        f"DATASET: {d['label']}\nSOURCE: {d['source']}\nADAPTATION NOTE: {d['adaptation']}\n\n"
        f"OBSERVED CHANGE (computed by the monitor):\n{json.dumps(state['observation'], default=str)}\n\n"
        f"PERIODS: baseline {state['base_start']}..{state['base_end']}, current {state['cur_start']}..{state['cur_end']}; "
        f"entity filter: {state.get('entity_filter') or 'none'}\n\n"
        f"AVAILABLE DIMENSIONS AND METRICS:\n{json.dumps(dims)}\n\n"
        "TOOLS: get_metric_summary, get_time_series, get_segment_metrics, compare_periods, scan_dimensions, get_entity_metrics, "
        "get_historical_metrics, run_statistical_test, simulate_segment_rate, simulate_mix, simulate_composition_model.\n"
        "Write a plan of at most 6 steps."
    )
    usage = state["usage"]
    plan = llm.structured(usage, SYSTEM, prompt, SCHEMA)
    trace = state["trace"]
    trace["tool_calls"] += len(d["dims"])  # list_dimensions queries
    trace["steps"].append(dict(node="planner", n_steps=len(plan["steps"])))
    return dict(plan=plan, dimensions=dims, trace=trace)
