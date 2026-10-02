-- AdLens parameterized query templates.
-- {placeholders} are identifiers/expressions taken ONLY from the whitelisted dataset registry
-- (tools/metric_tools.py). %(name)s placeholders are bound values.
-- Every metric is expressed as SUM(num) / SUM(den), so one set of queries serves means, rates and ratios.

-- name: metric_summary
SELECT SUM({num}) AS num, SUM({den}) AS den, COUNT(*) AS n_rows
FROM {table}
WHERE {base} AND {period} BETWEEN %(start)s AND %(end)s {filters};

-- name: time_series
SELECT {period} AS period, SUM({num}) AS num, SUM({den}) AS den, COUNT(*) AS n_rows
FROM {table}
WHERE {base} AND {period} BETWEEN %(start)s AND %(end)s {filters}
GROUP BY {period}
ORDER BY {period};

-- name: segment_metrics
SELECT {dim} AS segment, SUM({num}) AS num, SUM({den}) AS den, COUNT(*) AS n_rows
FROM {table}
WHERE {base} AND {period} BETWEEN %(start)s AND %(end)s {filters}
GROUP BY {dim};

-- name: row_values
SELECT {num} AS num, {den} AS den, {dim} AS segment
FROM {table}
WHERE {base} AND {period} BETWEEN %(start)s AND %(end)s {filters};

-- name: model_rows
SELECT {num} AS y, {den} AS w, {dim_list}
FROM {table}
WHERE {base} AND {period} BETWEEN %(start)s AND %(end)s {filters};

-- name: periods
SELECT DISTINCT {period} AS period
FROM {table}
WHERE {base}
ORDER BY period;

-- name: dimension_values
SELECT {dim} AS segment, COUNT(*) AS n_rows
FROM {table}
WHERE {base}
GROUP BY {dim}
ORDER BY n_rows DESC;
