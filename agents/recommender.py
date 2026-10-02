"""Recommendation agent (LLM): turns validated hypotheses + counterfactual numbers into actions.

It may only use numbers present in the validation and counterfactual results, and must say when the data cannot support an action.
"""
import json

from agents import llm
from tools import metric_tools as M

SYSTEM = (
    "You are the recommendation step of AdLens. Using ONLY the validated hypotheses (statuses were set by statistical tests, not by you) "
    "and the deterministic counterfactual results, write a short diagnosis and 2-4 concrete recommendations. Quote numbers exactly as given. "
    "Tie each recommendation to a hypothesis id and, when available, the counterfactual that sizes its effect. If the data is synthetic, "
    "derived, or the cause looks like a data artifact, say so plainly instead of recommending operational changes."
)
SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["diagnosis", "recommendations", "caveats"],
    "properties": {
        "diagnosis": {"type": "string"},
        "recommendations": {"type": "array", "items": {
            "type": "object", "additionalProperties": False, "required": ["action", "hypothesis_id", "expected_effect", "rationale"],
            "properties": {"action": {"type": "string"}, "hypothesis_id": {"type": "string"}, "expected_effect": {"type": "string"},
                           "rationale": {"type": "string"}}}},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
}


def node(state):
    d = M.get(state["dataset"])
    prompt = (
        f"Dataset: {d['label']}. Adaptation note: {d['adaptation']}\nObserved change: {json.dumps(state['observation'], default=str)[:1200]}\n\n"
        f"Validated hypotheses:\n{json.dumps(state['validations'], default=str)}\n\nCounterfactuals:\n{json.dumps(state['counterfactuals'], default=str)}"
    )
    rec = llm.structured(state["usage"], SYSTEM, prompt, SCHEMA)
    state["trace"]["steps"].append(dict(node="recommendation", n=len(rec["recommendations"])))
    return dict(recommendation=rec, trace=state["trace"])
