r"""The LangGraph investigation graph.

monitor -> (anomaly?) -> planner -> evidence -> hypotheses -> validation -> counterfactual -> recommendation -> report
                      \-> report (no anomaly)
"""
from langgraph.graph import END, START, StateGraph

from agents import counterfactual, hypothesis, investigator, monitor, planner, recommender, report, validator
from agents.llm import Usage
from agents.state import InvestigationState


def build():
    g = StateGraph(InvestigationState)
    g.add_node("monitor", monitor.node)
    g.add_node("planner", planner.node)
    g.add_node("evidence", investigator.node)
    g.add_node("hypotheses", hypothesis.node)
    g.add_node("validation", validator.node)
    g.add_node("counterfactual", counterfactual.node)
    g.add_node("recommendation", recommender.node)
    g.add_node("report", report.node)
    g.add_edge(START, "monitor")
    g.add_conditional_edges("monitor", lambda s: "planner" if s.get("anomalous") else "report", ["planner", "report"])
    g.add_edge("planner", "evidence")
    g.add_edge("evidence", "hypotheses")
    g.add_edge("hypotheses", "validation")
    g.add_edge("validation", "counterfactual")
    g.add_edge("counterfactual", "recommendation")
    g.add_edge("recommendation", "report")
    g.add_edge("report", END)
    return g.compile()


def investigate(dataset, metric, base_start, base_end, cur_start, cur_end, entity_filter=None) -> dict:
    app = build()
    state = dict(dataset=dataset, metric=metric, entity_filter=entity_filter or {}, base_start=base_start, base_end=base_end,
                 cur_start=cur_start, cur_end=cur_end, usage=Usage(), trace=dict(tool_calls=0, failed_tool_calls=0, steps=[]))
    return app.invoke(state)
