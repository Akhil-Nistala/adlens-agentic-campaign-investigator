"""Final report (deterministic formatting of the state)."""
import textwrap

from tools import metric_tools as M


def _wrap(s, indent="  "):
    return textwrap.fill(str(s), 100, initial_indent=indent, subsequent_indent=indent)


def render(s) -> str:
    d = M.get(s["dataset"])
    prov = M.provenance(s["dataset"])
    o = s.get("observation", {})
    L = ["ADLENS INVESTIGATION", "====================", ""]
    L += ["DATA SOURCE", _wrap(f"{'Real' if not d['is_advertising'] else 'Real + synthetic'} dataset: {d['label']}"), _wrap(d["source"]),
          _wrap("Adaptation: " + d["adaptation"]), ""]
    for k, name in (("REAL", "REAL FIELDS USED"), ("DERIVED", "DERIVED FIELDS"), ("SYNTHETIC", "SYNTHETIC FIELDS")):
        L += [name] + ([_wrap(x) for x in prov[k]] or ["  (none)"]) + [""]
    L += ["OBSERVED ANOMALY",
          _wrap(f"{o.get('metric')} ({o.get('metric_desc')}): {o.get('baseline_value')} -> {o.get('current_value')} "
                f"(delta {o.get('delta')}, {100 * (o.get('relative_change') or 0):+.1f}%, {o.get('direction')})"),
          _wrap(f"baseline {s['base_start']}..{s['base_end']} (n={o.get('n_baseline')}), current {s['cur_start']}..{s['cur_end']} (n={o.get('n_current')}); "
                f"entity filter: {s.get('entity_filter') or 'none'}"),
          _wrap(f"permutation test p={o.get('test', {}).get('p_value')}; robust z vs last {o.get('history', {}).get('history_periods')} periods: "
                f"{o.get('history', {}).get('robust_z')}; flagged as anomaly: {o.get('anomalous')} ({o.get('rule')})"), ""]
    if not s.get("anomalous"):
        L += ["No statistically meaningful change: investigation stopped after the monitor.", ""]
    else:
        L += ["INVESTIGATION PLAN", _wrap(s["plan"]["objective"])]
        L += [_wrap(f"{st['step']}. {st['tool']}: {st['purpose']}") for st in s["plan"]["steps"]] + [""]
        L += ["HYPOTHESES", ""]
        for v in s["validations"]:
            L += [f"{v['id']}  {v['hypothesis']}", f"    type {v['type']} | {v['dimension']}={v['segment'] or '*'} | Status: {v['status']} | "
                  f"confidence {v['confidence']} | explained {v['explained']} | p {v['p_value']}"]
            L += [_wrap("+ " + x, "    ") for x in v["supporting_evidence"]] + [_wrap("- " + x, "    ") for x in v["contradicting_evidence"]] + [""]
        L += ["EVIDENCE", _wrap(s["evidence_summary"]), _wrap(f"({len(s['evidence'])} evidence items from tool calls: "
                                                           + ", ".join(f"{e['id']} {e['tool']}" for e in s["evidence"]) + ")"), ""]
        L += ["COUNTERFACTUAL ANALYSIS"]
        for c in s["counterfactuals"]:
            if c.get("error"):
                L.append(_wrap(f"{c['hypothesis']}: could not simulate ({c['error']})"))
                continue
            what = c.get("segment") and f"{c['dimension']}={c['segment']} restored to {c.get('segment_value_set_to')}" or \
                c.get("dimension") and f"{c['dimension']} restored to {c.get('target')}" or c.get("model")
            after = c.get("counterfactual_metric", c.get("expected_from_composition"))
            L.append(_wrap(f"{c['hypothesis']}: {what}: observed {c.get('observed_metric')} -> {after} "
                           f"(baseline {c.get('baseline_metric')}; share of change accounted for {c.get('share_of_change_removed', c.get('explained_share'))})"))
            if c.get("caveat"):
                L.append(_wrap("caveat: " + c["caveat"], "    "))
        L += [""]
        r = s["recommendation"]
        L += ["RECOMMENDATION", _wrap(r["diagnosis"])] + [_wrap(f"- [{x['hypothesis_id']}] {x['action']} | expected: {x['expected_effect']}") for x in r["recommendations"]]
        L += [_wrap("caveat: " + c) for c in r["caveats"]] + [""]
    t = s["trace"]
    L += ["AGENT TRACE", f"  Tool calls: {t['tool_calls']}", f"  Failed tool calls: {t['failed_tool_calls']}",
          f"  Investigation steps: {len(t['steps'])} ({' -> '.join(x['node'] for x in t['steps'])})",
          f"  LLM: {t.get('llm', {})}"]
    return "\n".join(L)


def node(state):
    state["trace"]["llm"] = state["usage"].to_dict()
    state["trace"]["steps"].append(dict(node="report"))
    return dict(report=render(state), trace=state["trace"])
