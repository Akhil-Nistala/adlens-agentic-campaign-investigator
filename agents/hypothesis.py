"""Hypothesis agent (LLM): proposes competing, TESTABLE explanations grounded in the gathered evidence.

Each hypothesis must name a type the validator can test:
  RATE_CHANGE   one segment of a dimension got worse/better within itself (e.g. traffic=Jam deliveries got slower)
  MIX_SHIFT     the composition of a dimension changed (e.g. more long-distance orders), segments' own values unchanged
  BROAD_SHIFT   the change is spread across nearly all segments of a dimension (nothing segment-specific)
  OTHER         anything the tools cannot test (data artifact, external event...) -> will be marked INCONCLUSIVE
The LLM's own view of support is recorded but NOT used for the final status or confidence (validator decides).
"""
import json

from agents import llm

SYSTEM = (
    "You are the hypothesis step of AdLens. Propose 3 to 5 competing explanations for the observed change. "
    "Cover different mechanisms (a specific segment deteriorating, a change in composition, a broad shift, and at least one "
    "plausible alternative you expect to be wrong, so it can be rejected). Each must be testable with the given types, name an "
    "existing dimension and (for RATE_CHANGE / MIX_SHIFT) a segment value exactly as it appears in the evidence. Cite evidence ids "
    "for and against. Do not invent numbers."
)
SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["hypotheses"],
    "properties": {"hypotheses": {"type": "array", "items": {
        "type": "object", "additionalProperties": False,
        "required": ["id", "hypothesis", "type", "dimension", "segment", "supporting_evidence", "contradicting_evidence"],
        "properties": {
            "id": {"type": "string"}, "hypothesis": {"type": "string"},
            "type": {"type": "string", "enum": ["RATE_CHANGE", "MIX_SHIFT", "BROAD_SHIFT", "OTHER"]},
            "dimension": {"type": "string"}, "segment": {"type": "string"},
            "supporting_evidence": {"type": "array", "items": {"type": "string"}},
            "contradicting_evidence": {"type": "array", "items": {"type": "string"}},
        }}}},
}


def node(state):
    ev = [dict(id=e["id"], tool=e["tool"], input=e["input"], output=e["output"]) for e in state["evidence"]]
    prompt = (
        f"Observed change: {json.dumps(state['observation'], default=str)[:1200]}\n\nEvidence summary from the investigation:\n"
        f"{state['evidence_summary']}\n\nEvidence items:\n{json.dumps(ev, default=str)[:24000]}\n\n"
        f"Valid dimensions: {list(state['dimensions']['dimensions'])}. Use segment '' for BROAD_SHIFT/OTHER."
    )
    out = llm.structured(state["usage"], SYSTEM, prompt, SCHEMA)
    hyps = out["hypotheses"][:5]
    state["trace"]["steps"].append(dict(node="hypotheses", n=len(hyps)))
    return dict(hypotheses=hyps, trace=state["trace"])
