"""Evidence agent (LLM + tools): follows the plan, choosing and calling database tools in a bounded loop.

Every tool result is stored as an evidence item with an id (E1, E2, ...) so later steps can cite it.
"""
import json

import config
from agents import llm
from agents.tool_registry import TOOLS, compact, execute

SYSTEM = (
    "You are the evidence-gathering step of AdLens. Investigate why the metric changed by calling the tools; "
    "all numbers must come from tool results, never from your own arithmetic or assumptions. Use the investigation's "
    "default periods unless you deliberately compare others. Each tool result is labelled with an evidence id (E1, E2, ...). "
    "Stop calling tools once you can explain the change or have exhausted the plan, then write a short evidence summary "
    "that cites evidence ids and the exact numbers. Mention when a dimension shows no overlap between periods "
    "(segments present in only one period), because then within-segment comparisons are impossible."
)
STOP_NOTE = "Tool budget used up. Do not call any more tools: write the evidence summary now, citing evidence ids."


def _run_tools(calls, state, evidence, trace, allowed=True):
    """calls: [(call_id, name, args)] -> list of OpenAI tool-result messages."""
    msgs = []
    for cid, name, args in calls:
        if not allowed:
            msgs.append(llm.tool_result(cid, "ERROR: no more tool calls allowed"))
            continue
        trace["tool_calls"] += 1
        eid = f"E{len(evidence) + 1}"
        try:
            out = execute(name, dict(args), state)
            if isinstance(out, dict) and out.get("error"):
                trace["failed_tool_calls"] += 1
            evidence.append(dict(id=eid, tool=name, input=dict(args), output=out))
            msgs.append(llm.tool_result(cid, f"[{eid}] " + compact(out)))
        except Exception as e:  # invalid dimension/metric names etc. go back to the model as errors
            trace["failed_tool_calls"] += 1
            evidence.append(dict(id=eid, tool=name, input=dict(args), output={"error": str(e)}))
            msgs.append(llm.tool_result(cid, f"[{eid}] ERROR: {e}"))
    return msgs


def node(state):
    usage, trace = state["usage"], state["trace"]
    evidence = list(state.get("evidence") or [])
    # deterministic triage: every investigation starts from the all-dimension screen (E1), so even a small model
    # drills into dimensions the data points at instead of guessing
    triage = _run_tools([("triage", "scan_dimensions", {"metric": state["metric"]})], state, evidence, trace)[0]["content"]
    kickoff = (
        f"Metric: {state['metric']} on dataset '{state['dataset']}'. Baseline {state['base_start']}..{state['base_end']}, "
        f"current {state['cur_start']}..{state['cur_end']}. Entity filter: {state.get('entity_filter') or 'none'}.\n"
        f"Observed change: {json.dumps(state['observation'], default=str)[:1500]}\n"
        f"Plan: {json.dumps(state['plan'])}\nAvailable dimensions: {list(state['dimensions']['dimensions'])}\n"
        f"Triage already run (all dimensions screened): {triage}\n"
        f"Drill into the top-contributing dimensions from E1. You have at most {config.MAX_EVIDENCE_STEPS} more rounds of tool calls."
    )
    messages = [{"role": "user", "content": kickoff}]
    summary, rounds = "", 0
    while True:
        reply = llm.chat(usage, SYSTEM, messages, tools=TOOLS)
        messages.append(reply.message)
        if not reply.tool_calls:
            summary = reply.text
            break
        allowed = rounds < config.MAX_EVIDENCE_STEPS
        messages += _run_tools(reply.tool_calls, state, evidence, trace, allowed)
        rounds += 1
        if rounds >= config.MAX_EVIDENCE_STEPS:
            messages.append({"role": "user", "content": STOP_NOTE})
        if rounds > config.MAX_EVIDENCE_STEPS + 1:
            break
    trace["steps"].append(dict(node="evidence", rounds=rounds, evidence_items=len(evidence)))
    return dict(evidence=evidence, evidence_summary=summary, trace=trace)
