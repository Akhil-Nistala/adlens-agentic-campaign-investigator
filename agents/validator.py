"""Validation agent (deterministic): tests every hypothesis with SQL + statistics and assigns its status.

explained  = share of the observed change attributable to the hypothesis
             RATE_CHANGE: (rate + interaction effect of the segment) / delta
             MIX_SHIFT:   total mix effect of the dimension / delta
             BROAD_SHIFT: total rate effect of the dimension / delta, and >= 80% of volume moved in the same direction
p          = RATE_CHANGE: permutation test inside the segment; MIX_SHIFT: share z-test of the segment (or the segment whose
             share moved most); BROAD_SHIFT: permutation test on the whole population
Status     SUPPORTED          explained >= 0.30 and p < 0.01
           WEAKLY_SUPPORTED   explained >= 0.10 and p < 0.05
           REJECTED           explained < 0.05 (or pointing the other way) with >= 30 rows in each period
           INCONCLUSIVE       everything else (incl. untestable OTHER hypotheses and missing data)
Confidence = clip(explained, 0, 1) x (1.0 if p < 0.01, 0.6 if p < 0.05, else 0.2)
The LLM never sets status or confidence.
"""
from tools import database_tools as T
from tools import metric_tools as M


def pct(x):
    return "n/a" if x is None else f"{x:.0%}"


def classify(explained, p, n_ok):
    if explained is None or p is None:
        return "INCONCLUSIVE"
    if explained >= 0.30 and p < 0.01:
        return "SUPPORTED"
    if explained >= 0.10 and p < 0.05:
        return "WEAKLY_SUPPORTED"
    if explained < 0.05 and n_ok:
        return "REJECTED"
    return "INCONCLUSIVE"


def confidence(explained, p):
    if explained is None or p is None:
        return 0.0
    w = 1.0 if p < 0.01 else 0.6 if p < 0.05 else 0.2
    return round(max(0.0, min(1.0, explained)) * w, 3)


def validate_one(h, s):
    ds, metric = s["dataset"], s["metric"]
    per = (s["base_start"], s["base_end"], s["cur_start"], s["cur_end"])
    filt = s.get("entity_filter") or None
    out = dict(id=h["id"], hypothesis=h["hypothesis"], type=h["type"], dimension=h["dimension"], segment=h.get("segment") or None,
               llm_cited_support=h.get("supporting_evidence"), llm_cited_against=h.get("contradicting_evidence"))
    if h["type"] == "OTHER":
        return dict(out, status="INCONCLUSIVE", confidence=0.0, explained=None, p_value=None,
                    supporting_evidence=[], contradicting_evidence=["not testable with the available tools/data"])
    try:
        M.dim_expr(ds, h["dimension"])
    except ValueError as e:
        return dict(out, status="INCONCLUSIVE", confidence=0.0, explained=None, p_value=None, supporting_evidence=[], contradicting_evidence=[str(e)])
    cp = T.compare_periods(ds, metric, h["dimension"], *per, filters=filt, top=50)
    delta = cp.get("delta") or 0.0
    segs = {r["segment"]: r for r in cp.get("segments", [])}
    sup, con = [], []
    seg = str(h.get("segment") or "")
    if cp.get("kind") == "total":
        r = segs.get(seg)
        explained = (r["contribution"] / delta) if (r and delta) else None
        test = T.run_statistical_test(ds, metric, *per, h["dimension"], seg or None, "rate")
        p, n_ok = test.get("p_value"), True
    elif h["type"] == "RATE_CHANGE":
        r = segs.get(seg)
        if not r:
            return dict(out, status="INCONCLUSIVE", confidence=0.0, explained=None, p_value=None, supporting_evidence=[],
                        contradicting_evidence=[f"segment '{seg}' not found in {h['dimension']} (values: {list(segs)[:10]})"])
        if r["n_base"] == 0 or r["n_current"] == 0:
            return dict(out, status="INCONCLUSIVE", confidence=0.0, explained=None, p_value=None, supporting_evidence=[],
                        contradicting_evidence=[f"segment '{seg}' exists in only one period (baseline n={r['n_base']}, current n={r['n_current']}), "
                                                "so its own value cannot be compared"])
        explained = (r["rate_effect"] + r["interaction"]) / delta if delta else None
        test = T.run_statistical_test(ds, metric, *per, h["dimension"], seg, "rate")
        p = test.get("p_value"); n_ok = r["n_base"] >= 30 and r["n_current"] >= 30
        (sup if (explained or 0) > 0.05 else con).append(
            f"{h['dimension']}={seg}: {r['base_value']} -> {r['current_value']} (rate effect {r['rate_effect']}, {pct(explained)} of the change)")
        (sup if p is not None and p < 0.05 else con).append(f"permutation test inside segment: effect {test.get('effect')}, p={p}")
    elif h["type"] == "MIX_SHIFT":
        explained = cp["mix_effect_total"] / delta if delta else None
        if not seg or seg not in segs:
            seg = max(segs, key=lambda k: abs((segs[k]["current_share"] or 0) - (segs[k]["base_share"] or 0)))
        r = segs[seg]
        test = T.run_statistical_test(ds, metric, *per, h["dimension"], seg, "share")
        p = test.get("p_value"); n_ok = sum(x["n_base"] for x in segs.values()) >= 30 and sum(x["n_current"] for x in segs.values()) >= 30
        (sup if (explained or 0) > 0.05 else con).append(f"composition of {h['dimension']} explains {pct(explained)} of the change (mix effect {cp['mix_effect_total']})")
        (sup if p is not None and p < 0.05 else con).append(f"share of {seg}: {r['base_share']} -> {r['current_share']} (z-test p={p})")
        disjoint = [k for k, x in segs.items() if (x["n_base"] == 0) != (x["n_current"] == 0)]
        if disjoint:
            con.append(f"segments present in only one period: {disjoint[:6]}. The composition change is total, so within-segment effects cannot be separated")
    else:  # BROAD_SHIFT
        explained = cp["rate_effect_total"] / delta if delta else None
        moved = sum((x["base_share"] or 0) for x in segs.values() if x["base_value"] is not None and x["current_value"] is not None
                    and (x["current_value"] - x["base_value"]) * delta > 0)
        test = T.run_statistical_test(ds, metric, *per, None, None, "rate")
        p = test.get("p_value"); n_ok = True
        if moved < 0.8 and explained is not None:
            explained = explained * moved          # a "broad" shift that most volume did not share is penalised
        (sup if moved >= 0.8 else con).append(f"{moved:.0%} of baseline volume is in segments that moved in the same direction")
        (sup if (explained or 0) > 0.3 else con).append(f"within-segment (rate) effects explain {cp['rate_effect_total']} of delta {delta}")
    status = classify(explained, p, n_ok)
    return dict(out, status=status, confidence=confidence(explained, p), explained=None if explained is None else round(explained, 3),
                p_value=p, supporting_evidence=sup, contradicting_evidence=con)


def confounding_check(vals, s):
    """Simpson's-paradox guard. If a composition shift in dimension M is SUPPORTED, any within-segment explanation on another
    dimension D is re-measured inside strata of M that exist in both periods. No overlapping strata -> INCONCLUSIVE."""
    mixes = [v for v in vals if v["type"] == "MIX_SHIFT" and v["status"] == "SUPPORTED"]
    if not mixes:
        return vals
    M_dim = max(mixes, key=lambda v: v["explained"] or 0)["dimension"]
    ds, metric = s["dataset"], s["metric"]
    per = (s["base_start"], s["base_end"], s["cur_start"], s["cur_end"])
    strat = T.compare_periods(ds, metric, M_dim, *per, filters=s.get("entity_filter") or None, top=50)
    if strat.get("kind") == "total":
        return vals
    delta = strat["delta"]
    overlap = [r for r in strat["segments"] if r["n_base"] >= 20 and r["n_current"] >= 20]
    for v in vals:
        if v["type"] not in ("RATE_CHANGE", "BROAD_SHIFT") or v["dimension"] == M_dim or v["status"] not in ("SUPPORTED", "WEAKLY_SUPPORTED"):
            continue
        if not overlap:
            v["status"], v["confidence"] = "INCONCLUSIVE", 0.0
            v["contradicting_evidence"].append(f"confounded: {M_dim} composition changed completely (no {M_dim} value present in both periods), "
                                               f"so the within-{v['dimension']} change cannot be separated from the {M_dim} shift")
            continue
        adj, wsum = 0.0, 0.0
        for r in overlap:
            f = dict(s.get("entity_filter") or {}); f[M_dim] = r["segment"]
            cp = T.compare_periods(ds, metric, v["dimension"], *per, filters=f, top=50)
            if cp.get("kind") == "total":
                continue
            eff = cp["rate_effect_total"] if v["type"] == "BROAD_SHIFT" else next(
                ((x["rate_effect"] or 0) + (x["interaction"] or 0) for x in cp["segments"] if x["segment"] == v["segment"]), 0.0)
            adj += (r["current_share"] or 0) * (eff or 0); wsum += r["current_share"] or 0
        explained_adj = (adj / wsum) / delta if (wsum and delta) else None
        v["contradicting_evidence"].append(f"controlling for {M_dim} (strata present in both periods, {wsum:.0%} of current volume): "
                                           f"explained share falls from {pct(v['explained'])} to {pct(explained_adj)}")
        v["explained"] = None if explained_adj is None else round(explained_adj, 3)
        v["status"] = classify(explained_adj, v["p_value"], True)
        v["confidence"] = confidence(explained_adj, v["p_value"])
    return vals


def node(state):
    vals = [validate_one(h, state) for h in state["hypotheses"]]
    vals = confounding_check(vals, state)
    vals.sort(key=lambda v: -v["confidence"])
    state["trace"]["tool_calls"] += 2 * len(vals)
    state["trace"]["steps"].append(dict(node="validation", statuses={v["id"]: v["status"] for v in vals}))
    return dict(validations=vals, trace=state["trace"])
