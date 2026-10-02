"""Counterfactual analysis (deterministic): quantifies how much of the change each validated hypothesis accounts for.

RATE_CHANGE -> simulate_segment_rate (segment restored to its baseline value)
MIX_SHIFT   -> simulate_mix (dimension restored to its baseline composition)
plus one joint composition model (scikit-learn) over the dimensions named by supported/weak hypotheses.
"""
from tools import simulation_tools as SIM


def node(state):
    ds, metric = state["dataset"], state["metric"]
    per = (state["base_start"], state["base_end"], state["cur_start"], state["cur_end"])
    out = []
    keep = [v for v in state["validations"] if v["status"] in ("SUPPORTED", "WEAKLY_SUPPORTED")]
    for v in keep:
        try:
            if v["type"] == "RATE_CHANGE" and v["segment"]:
                r = SIM.simulate_segment_rate(ds, metric, v["dimension"], v["segment"], *per)
            elif v["type"] == "MIX_SHIFT":
                r = SIM.simulate_mix(ds, metric, v["dimension"], *per)
            else:
                continue
            out.append(dict(hypothesis=v["id"], **r))
        except Exception as e:
            out.append(dict(hypothesis=v["id"], error=str(e)))
    dims = sorted({v["dimension"] for v in keep if v["type"] in ("RATE_CHANGE", "MIX_SHIFT")})
    if dims:
        try:
            out.append(dict(hypothesis="joint", **SIM.simulate_composition_model(ds, metric, dims, *per)))
        except Exception as e:
            out.append(dict(hypothesis="joint", error=str(e)))
    state["trace"]["tool_calls"] += len(out)
    state["trace"]["steps"].append(dict(node="counterfactual", n=len(out)))
    return dict(counterfactuals=out, trace=state["trace"])
