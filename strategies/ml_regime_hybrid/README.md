# ML Regime Hybrid Strategy

Long-only swing trading strategy that combines an XGBoost binary classifier with
rule-based regime filtering and structural confirmations.

## Pipeline

1. **Regime Filter** (`regime.py`) -- classifies each bar as RISK_OFF / NEUTRAL / RISK_ON
2. **ML Scoring** -- XGBoost predicts probability of a "favorable" setup (positive forward return + regime persistence)
3. **Entry Confirmations** (`confirmations.py`) -- trend alignment, volatility, overbought, and resistance checks
4. **Exit Rules** -- regime change, trend break, stop loss, trailing stop, max hold

Only RISK_ON bars with ML probability > threshold AND passing all confirmations trigger entries.

---

## Features (from manifest.json)

| Feature | Params | Role |
|---------|--------|------|
| RSI(14) | period=14 | Overbought filter (entry gate) |
| Bollinger Bands(20, 2.0) | period=20, std_dev=2.0 | %B overbought filter + width as vol proxy |
| ATR(14) raw | period=14, normalize=none | Volatility regime classification (percentile rank) |
| ATR(14) normalized | period=14, normalize=pct_distance | Additional vol feature for ML |
| ADX(14) | period=14 | Trend strength gate (min_adx_entry) |
| EMA(20) raw | period=20, type=EMA | Short-term trend reference |
| EMA(50) raw | period=50, type=EMA | Regime trend condition (close > EMA50 > EMA200) |
| EMA(200) raw | period=200, type=EMA | Long-term trend anchor for regime + confirmations |
| EMA(50) normalized | period=50, normalize=pct_distance | Sign indicates close vs EMA50 relationship for regime |
| Yearly Cycle | sin + cos | Seasonality encoding for ML |
| Support/Resistance | Bill Williams, window=35 | Nearest resistance used as entry rejection zone |

---

## Hyperparameters

### XGBoost Model Parameters

| Parameter | Default | Bounds | Description |
|-----------|---------|--------|-------------|
| `n_estimators` | 300 | -- | Number of boosting rounds. More trees = more capacity but more overfit risk. |
| `max_depth` | 4 | -- | Maximum tree depth. Controls model complexity per tree. |
| `learning_rate` | 0.05 | -- | Shrinkage rate. Lower = more rounds needed, but better generalization. |
| `min_child_weight` | 5 | -- | Minimum sum of instance weight in a leaf. Higher = more conservative splits. |
| `subsample` | 0.7 | -- | Fraction of rows sampled per tree. Reduces variance / overfit. |
| `colsample_bytree` | 0.7 | -- | Fraction of features sampled per tree. Reduces feature co-adaptation. |
| `reg_lambda` | 2.0 | -- | L2 regularization on leaf weights. Penalizes large predictions. |
| `reg_alpha` | 0.1 | -- | L1 regularization on leaf weights. Encourages sparse trees. |
| `gamma` | 0.1 | -- | Minimum loss reduction for a split. Higher = fewer splits = simpler trees. |

### Labeling

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lookforward` | 20 | Bars ahead to measure forward return and regime persistence for label construction. A bar is FAVORABLE (1) if forward return > 0 AND regime is still RISK_ON after this many bars. |

### Regime Classification

| Parameter | Default | Bounds | Description |
|-----------|---------|--------|-------------|
| `low_vol_threshold` | 0.25 | [0.15, 0.35] | ATR percentile below which vol is "low". Required (with uptrend + vol stability) for RISK_ON. |
| `high_vol_threshold` | 0.75 | [0.65, 0.85] | ATR percentile above which vol is "high". Triggers RISK_OFF regardless of trend. |
| `atr_lookback` | 252 | -- | Rolling window for ATR percentile rank (~1 year of trading days). |
| `vol_expansion_limit` | 0.1 | -- | Max 5-bar ATR rate-of-change for vol to be considered "stable". Exceeding this blocks RISK_ON. |

### Entry Confirmations

| Parameter | Default | Bounds | Description |
|-----------|---------|--------|-------------|
| `entry_quality_threshold` | 0.35 | [0.25, 0.50] | Minimum ML p(favorable) to consider an entry. Below this the bar is skipped even if regime is RISK_ON. |
| `min_adx_entry` | 20 | -- | Minimum ADX value to confirm trend strength. Below this, trend is too weak for entry. |
| `max_rsi_entry` | 75 | -- | RSI ceiling. Bars with RSI above this are rejected as overbought. |
| `max_bb_pct_b_entry` | 0.95 | -- | Bollinger %B ceiling. Values near 1.0 mean price is at the upper band; rejected as extended. |
| `resistance_buffer_pct` | 0.02 | -- | If nearest resistance is within this % of close, entry is rejected (overhead supply). |

### Exit Rules

| Parameter | Default | Bounds | Description |
|-----------|---------|--------|-------------|
| `stop_loss` | 0.07 | [0.04, 0.12] | Hard stop: exit if unrealized loss exceeds this % from entry price. |
| `trailing_stop_pct` | 0.10 | -- | Trailing stop: exit if price drops this % from peak since entry. |
| `max_hold_days` | 60 | -- | Time stop: force exit after this many bars regardless of P&L. |

### Unused / Reserved

| Parameter | Default | Bounds | Description |
|-----------|---------|--------|-------------|
| `max_drawdown_threshold` | 0.05 | [0.03, 0.08] | Intended for portfolio-level drawdown gating. Not currently wired into model.py. |
| `min_profit_threshold` | 0.03 | [0.02, 0.06] | Intended for minimum profit target before allowing exit. Not currently wired into model.py. |

---

## Regime States

| State | Value | Condition | Behavior |
|-------|-------|-----------|----------|
| RISK_OFF | 0 | High vol OR downtrend (close < EMA50 AND EMA50 < EMA200) | No entries. Immediate exit if in position. |
| NEUTRAL | 1 | Neither RISK_ON nor RISK_OFF | No new entries. Existing positions held (but subject to exit rules). |
| RISK_ON | 2 | Low vol AND uptrend (close > EMA50 > EMA200) AND vol stable | Entries allowed if ML + confirmations pass. |

---

## Entry Flow

```
RISK_ON? --> ML p(favorable) > threshold? --> trend confirmed? --> vol ok? --> not overbought? --> not at resistance? --> ENTER LONG
```

All gates must pass. Any single failure blocks the entry.

## Exit Flow

Checked every bar while in position (first match exits):

1. Regime == RISK_OFF
2. Close < EMA(50) (trend break)
3. Unrealized loss > stop_loss
4. Drawdown from peak > trailing_stop_pct
5. Bars held > max_hold_days
