# AdLens — agentic campaign / performance investigator

AdLens investigates **why a performance metric changed**. A LangGraph agent plans an investigation, calls SQL tools over MySQL, proposes competing hypotheses, and then **Python validates each one statistically** and **simulates counterfactuals** before an LLM writes recommendations. The LLM plans, picks tools, hypothesises and explains. SQL and Python do every calculation, every statistical test and every status decision.

> **Not advertising data.** None of the available datasets contain real impressions, clicks or spend (verified by `data/profile_data.py`, see `results/data_profile.md`). The primary dataset is a **real food-delivery order log**, and AdLens investigates delivery performance with the same workflow an ad analyst uses for campaign data. A second mode adds a clearly labelled **synthetic** sponsored-listing layer on top of **real** restaurants to demonstrate ad metrics (CTR/CVR/CPC). Every synthetic number is labelled as such.

## Stack
Python · LangGraph · **MySQL** (local server) · **OpenAI Python SDK** (Chat Completions with tool calling and JSON-schema structured outputs) · pandas · NumPy · scikit-learn.

The LLM layer (`agents/llm.py`) talks to any OpenAI-compatible endpoint:
- **Local, free, default for the results below:** `qwen2.5:3b` served by **Ollama** with a 16K context (`ADLENS_LLM_BASE_URL=http://localhost:11434/v1`). Nothing leaves the machine and the cost is $0.
- **Hosted:** leave `ADLENS_LLM_BASE_URL` empty and set `OPENAI_API_KEY` and `ADLENS_MODEL` (e.g. `gpt-5.4-mini`). Token cost is then tracked per run.

No web app, API server, Docker or queues.

## Data provenance

**Real dataset used as the underlying population; synthetic variables were introduced only where required to demonstrate temporal/agentic analysis.**

### `orders` (primary)
Source: `food_delivery_analytics.real_orders`. These are real Kaggle "Food Delivery Dataset" orders, 45,593 rows, cleaned by an earlier project; 1,731 rows without a timestamp are excluded. The orders run from 2022-02-11 to 2022-04-06 (44 days).

| Status | Fields |
|---|---|
| **REAL** | order_id, delivery_person_id, courier age & rating, order/pickup timestamps, weather, traffic_density, vehicle_type, vehicle_condition, order_type, multiple_deliveries, festival, city_type, time_taken_min |
| **DERIVED** | distance_km (haversine of real coordinates), prep_time_min, on_time (≤ 30 min SLA), distance_band, hour_band, courier_rating_band, courier_age_band |
| **SYNTHETIC** | none in normal use. The evaluation copy `adlens_orders_eval` contains one injected anomaly per test scenario |

Metrics supported: average delivery time, on-time rate, courier rating, distance, order count. **CTR, CPC and conversion are not computed: the data has no impressions, clicks or spend.**

### `restaurant_ads` (demonstration of ad metrics)
| Status | Fields |
|---|---|
| **REAL** | restaurant id, name, city, rating, rating_count, cost (from `restaurant_health_monitoring.restaurants`, Kaggle Swiggy listings) |
| **DERIVED** | primary_cuisine, cost_tier (earlier project), rating_band |
| **SYNTHETIC** | week_start, placement, impressions, clicks, orders, spend (`adlens_sponsored_weekly`, `data/synthetic_layer.py`: 20,000 real restaurants × 12 weeks, generating model documented in the file) |

## Architecture
```
main.py ─▶ LangGraph (agents/graph.py)
  monitor ──(no change)──────────────────────────────────────────────▶ report
     │ anomaly: |Δ| ≥ 5% and permutation p < 0.01            [deterministic]
     ▼
  planner ─▶ evidence agent ─▶ hypothesis agent ─▶ validator ─▶ counterfactual ─▶ recommender ─▶ report
  [LLM]      [LLM + SQL tools]  [LLM, structured]   [stats]     [simulation]      [LLM]          [format]
                  │
                  ▼  tools/database_tools.py · tools/simulation_tools.py · database/queries.sql (MySQL)
```
- **Tools** (`agents/tool_registry.py`): `get_metric_summary`, `get_time_series`, `get_segment_metrics`, `compare_periods` (shift-share: mix vs rate effects), `scan_dimensions`, `get_entity_metrics`, `get_historical_metrics`, `run_statistical_test` (permutation / two-proportion z), `simulate_segment_rate`, `simulate_mix`, `simulate_composition_model` (scikit-learn Ridge trained on baseline rows).
- **Safety:** SQL identifiers come only from the dataset registry (`tools/metric_tools.py`); values are bound parameters; the model can't change the dataset; invalid calls return `is_error` results that are counted as failed tool calls.
- **Hypothesis types** the validator can test: `RATE_CHANGE` (one segment changed), `MIX_SHIFT` (composition changed), `BROAD_SHIFT` (everything moved). `OTHER` is always INCONCLUSIVE.
- **Status rules** (`agents/validator.py`, never set by the LLM):

  | Status | Rule (explained = share of the change attributable to the hypothesis) |
  |---|---|
  | SUPPORTED | explained ≥ 30% and p < 0.01 |
  | WEAKLY_SUPPORTED | explained ≥ 10% and p < 0.05 |
  | REJECTED | explained < 5% with ≥ 30 rows per period |
  | INCONCLUSIVE | otherwise |

  Confidence = clip(explained) × (1 / 0.6 / 0.2 by p-value band).
- **Simpson's-paradox guard:** if a composition shift is SUPPORTED, every within-segment explanation on another dimension is re-measured inside strata of that dimension. If the strata don't overlap between periods, it becomes INCONCLUSIVE.

## Run
```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
cp .env.example .env                        # DB password + LLM endpoint (gitignored)
printf "FROM qwen2.5:3b\nPARAMETER num_ctx 16384\n" > Modelfile && ollama create qwen2.5-3b-16k -f Modelfile
python data/profile_data.py                 # dataset discovery
python data/synthetic_layer.py              # synthetic ad layer (restaurant_ads only)
python main.py                              # select dataset, entity, periods; prints the investigation
python main.py --dataset orders --metric avg_delivery_time --base 2022-03-13 2022-03-13 --current 2022-03-14 2022-03-14
python evaluation/evaluate.py detection     # deterministic monitor precision/recall
python evaluation/evaluate.py investigation 8   # AdLens vs one-prompt baseline (uses the LLM; resumable)
```

## A real finding in the real data
Average delivery time **alternates between about 23 and 30 minutes on consecutive days** across the whole period. AdLens's tools show that the two kinds of day have **completely disjoint distance bands**: every order on a "fast" day is under 10 km and every order on a "slow" day is 10 km or more. Orders on the slow days are also placed later in the evening.

So the composition shift explains 100% of the change, and per-segment explanations (traffic, weather) are confounded and cannot be separated. That looks like an artefact of how the public dataset was constructed, not an operational event, and the report says so. The evaluation therefore compares days of the same type.

## Evaluation
### Anomaly detection (deterministic monitor) — run
153 trials on real orders: 8 injected anomaly types × 17 days + 17 clean days (`results/eval_detection.json`). Rule: |rel change| >= 5% and p < 0.01.

| | Value |
|---|---|
| Precision | **100%** (128 true / 0 false alarms) |
| Recall | **94.1%** (128 of 136 injected anomalies) |
| False-positive rate on clean days | 0% (0 of 17) |
| Weakest scenario | low-rated couriers (smallest injection, about 6% change): flagged on 12 of 17 days |

### Investigation quality, AdLens vs one-prompt baseline
Scoring (`evaluation/evaluate.py`): correct root cause (type, dimension and segment), whether the true cause was supported, share of wrong hypotheses not falsely supported, evidence quality, counterfactual error, tool calls, failed tool calls, steps, tokens and time. Runs are logged to the `adlens_runs` MySQL table and are resumable.

**Status: pilot only.** On the local `qwen2.5:3b` (CPU), one investigation takes about 15 minutes, so only the first scenario has been run; the 4-scenario run was stopped for compute. Pilot result, injected *traffic-jam slowdown*:

| | AdLens (local 3B) | One-prompt baseline |
|---|---|---|
| Correct root cause | No: named *single-delivery orders* (a correlated segment: the injected jam orders are mostly single deliveries) | No: named a *morning-order* mix shift |
| How the conclusion was reached | SUPPORTED by the validator: explains 77% of the change, p = 0.0005 | asserted by the LLM, no test |
| Wrong hypotheses not falsely supported | 4 of 5 (REJECTED / INCONCLUSIVE) | n/a |
| Tool calls / failed | 31 / 0 | 25 queries |
| LLM calls / time / cost | 5 / 15.6 min / $0 | 1 / 44 s / $0 |

What this shows: the statistics layer behaves as designed (no unsupported claim reaches the report), but a 3B model proposes too narrow a hypothesis set to name the injected dimension, even though every investigation starts with a deterministic all-dimension screen (`scan_dimensions`, evidence E1). The full comparison should be run with a larger model (`ADLENS_MODEL=gpt-5.4-mini` or a bigger local model).


## Limitations
- No real advertising data. Ad metrics exist only in the synthetic layer; the real-data investigation is delivery performance.
- Ground truth for diagnosis comes from **injected** anomalies on real data, not from naturally occurring labelled incidents.
- Shift-share decomposition is additive and order-dependent for interactions; correlated dimensions are handled by the stratification guard only for the strongest supported mix shift.
- Permutation tests treat orders as exchangeable within a day.
- LLM output varies run to run; status and confidence do not, because they are computed.
