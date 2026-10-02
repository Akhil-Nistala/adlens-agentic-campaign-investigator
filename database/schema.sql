-- AdLens tables (created in ADLENS_WORK_DB, default food_delivery_analytics). Real source tables are only read, never modified.

-- SYNTHETIC: weekly sponsored-listing performance for REAL Swiggy restaurants (data/synthetic_layer.py)
CREATE TABLE IF NOT EXISTS adlens_sponsored_weekly (
    restaurant_id BIGINT NOT NULL,          -- REAL id from restaurant_health_monitoring.restaurants
    week_start    DATE   NOT NULL,          -- SYNTHETIC calendar
    placement     VARCHAR(16) NOT NULL,     -- SYNTHETIC: search | feed | banner
    impressions   INT    NOT NULL,          -- SYNTHETIC
    clicks        INT    NOT NULL,          -- SYNTHETIC
    orders        INT    NOT NULL,          -- SYNTHETIC
    spend         DECIMAL(12,2) NOT NULL,   -- SYNTHETIC, INR
    PRIMARY KEY (restaurant_id, week_start),
    KEY ix_week (week_start)
);

-- Investigation log: one row per AdLens or baseline run
CREATE TABLE IF NOT EXISTS adlens_runs (
    run_id        VARCHAR(40) PRIMARY KEY,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    system_name   VARCHAR(20),              -- adlens | baseline
    dataset       VARCHAR(40),
    metric        VARCHAR(40),
    scenario      VARCHAR(60),
    base_start    DATE, base_end DATE, cur_start DATE, cur_end DATE,
    tool_calls    INT, failed_tool_calls INT, steps INT,
    llm_calls     INT, input_tokens INT, output_tokens INT, cost_usd DECIMAL(10,4),
    result_json   LONGTEXT
);
