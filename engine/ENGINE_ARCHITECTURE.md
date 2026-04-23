# Research Engine — Architecture & Design

A reference document describing every component of the engine (the `engine/` package and the
top-level `strategies/` workspace). The PyQt6 charting GUI under `src/` is deliberately
excluded; this doc covers only the backtesting / training / signal-generation backend and
the strategies it executes.

---

## 1. Purpose & Design Philosophy

The engine is a modular backend for systematic trading research. It answers three questions
for a given strategy and dataset:

1. **BACKTEST** — How would this strategy have performed historically?
2. **TRAIN** — What parameters (or fitted ML model) best describe this strategy's edge?
3. **SIGNAL** — What is the current trading signal for today's data?

Every execution path flows through the same pipeline: fetch data → compute features → purge
warmup → invoke user strategy → validate/compress signals → evaluate. This uniformity is the
core design invariant — training, signal generation, and backtesting share one code path
so that validation metrics are directly comparable to backtest metrics and to live signals.

Key design choices:

- **Separation of concerns.** The engine owns the *data pipeline*. User strategies own
  *signal logic*. A strict contract (`SignalModel`) separates them.
- **Declarative strategies.** Features, hyperparameters, and training configuration live in
  a single `manifest.json` file per strategy. `context.py` is generated from the manifest
  so user code has typed, attribute-based access to every configured feature.
- **Deterministic feature naming.** Every feature produces column names derived from its
  ID and parameters, so the same manifest always yields the same DataFrame columns across
  runs and machines.
- **Safety boundaries.** Every user-generated signal is passed through `SignalValidator`
  before it hits any metric calculation, guaranteeing `[-1.0, 1.0]` bounds, `DatetimeIndex`
  alignment, and no `NaN` / `Inf` contamination.
- **Two deployment modes.** The same core can run as a CLI for iterative research or as a
  FastAPI + Redis/RQ service for queued, multi-worker job execution.

---

## 2. Top-Level Layout

```
stock_bot/
├── engine/                   # Research engine (this doc)
│   ├── main.py               # CLI entry (BACKTEST / TRAIN / SIGNAL / INIT / SYNC)
│   ├── core/                 # Engine library
│   │   ├── analytics/        # Phase 1 IC analysis (ic_analyzer, macro_fetcher, conditional_ic, report)
│   │   ├── diagnostics/      # Phase 0 diagnostic tools (trial counter, DSR)
│   │   └── regime/           # Phase 3 regime detection (rule_based, hmm, bocpd, orchestrator)
│   ├── daemon/               # FastAPI server + RQ worker
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── data/                 # SQLite caches (stocks.db, diagnostics.db)
│   └── tests/                # pytest suite (run inside research_tester container)
├── strategies/               # User strategies (the actual workspaces)
│   ├── RSI_Divergence/
│   ├── consolidation_breakout/
│   ├── ml_XGBoost/
│   ├── ml_regime_hybrid/
│   └── volitility_regime_identifier/
└── research_cli.py           # LLM-facing CLI for strategy authoring
```

The engine is mounted inside Docker containers at `/code/engine`, with `PYTHONPATH=/code`
so `import engine` resolves. `WORKDIR` is `/code/engine` so relative paths behave as on
the host.

---

## 3. The Execution Pipeline

### 3.1 Entry points

Three front-ends converge on the same `ApplicationController.execute_job()`:

- **CLI** ([engine/main.py](engine/main.py)): direct synchronous execution. Parses
  `BACKTEST | TRAIN | SIGNAL | INIT | SYNC` and a ticker/interval/timeframe, then calls the
  controller.
- **API** ([engine/daemon/main.py](engine/daemon/main.py)): `POST /api/v1/jobs` validates
  a `JobPayloadRequest` and enqueues it to Redis via RQ.
- **Worker** ([engine/daemon/worker.py](engine/daemon/worker.py)): RQ pulls the job, calls
  `process_job()` in [engine/daemon/tasks.py](engine/daemon/tasks.py), which instantiates
  `ApplicationController` and invokes `execute_job()`.

The payload schema is the same object in all three paths — a `JobPayload` Pydantic model
([engine/core/controller.py:47](engine/core/controller.py#L47)) with fields
`strategy`, `assets`, `interval`, `timeframe`, `mode`, and `multi_asset_mode`.

### 3.2 The controller

`ApplicationController` ([engine/core/controller.py:117](engine/core/controller.py#L117))
is the routing layer. `execute_job()` coerces the payload, validates that the strategy
directory exists, and dispatches to one of three handlers:

- `_handle_backtest` — increments the trial counter (`diagnostics.trial_counter`), fetches
  OHLCV for every asset, instantiates `LocalBacktester` once, and calls `run_batch()`. For
  each returned signal series, it computes a tearsheet via `Tearsheet.calculate_metrics`
  (passing `n_trials` so DSR is correctly deflated) and strips bulky time-series objects
  before returning.
- `_handle_train` — increments the trial counter (train mode), fetches data, optionally runs
  hyperparameter search (Phase A), then calls `LocalTrainer.run()` with the optimal params
  (Phase B). Accepts either a single DataFrame (legacy single-ticker) or a `{ticker: df}`
  dict for pooled multi-ticker ML training. Governed by a module-level `ENABLE_HPO` kill
  switch ([engine/core/controller.py:25](engine/core/controller.py#L25)) that currently
  skips Phase A even when bounds are defined.
- `_handle_signal_only` — identical to backtest but only returns the last signal value
  and timestamp per asset. Used for live decision-making.

### 3.3 The backtester

`LocalBacktester` ([engine/core/backtester.py:296](engine/core/backtester.py#L296))
owns a strategy directory and performs one vectorized pass per asset. Lifecycle of
`run_batch()`:

1. **Load user classes once.** Dynamically imports `model.py` and `context.py` from the
   strategy dir via `importlib.util.spec_from_file_location`, finds the `SignalModel`
   subclass in the module, and caches the class objects for the whole batch. This is why
   `model.py` must contain exactly one `SignalModel` subclass.
2. **Per-asset pipeline:**
   - `compute_all_features(raw_data, features_config)` → `(df_full, l_max)`.
   - **Warmup purge:** `df_clean = df_full.iloc[l_max:]` where `l_max` is the max lookback
     discovered from any feature parameter whose key matches `{window, period, slow, fast,
     lookback}`. This single cut guarantees every feature has valid values from row 0.
   - Optional price normalization (`log_returns` or `ffd`) via `MLBridge.apply_price_normalization`
     if the manifest's `training.price_normalization` says so.
   - NaN audit: any remaining NaN in a feature column after the purge logs a warning
     (indicates a broken indicator or bad input data).
   - **Regime context** (optional): if the manifest has `"regime_aware": true`, calls
     `_build_regime_context(df_clean)` which instantiates `RegimeOrchestrator`, fetches
     macro data, fits the requested detector, and runs BOCPD. The resulting
     `RegimeContext` is passed to `generate_signals` only when the model declares the
     parameter (checked via `inspect.signature`). Failures are caught and logged — the
     backtest continues without a regime context rather than aborting.
   - **ML / rule-based branching:**
     - **With provided artifacts** (e.g. from disk for ML strategies): apply saved
       `system_scaler` to feature columns via `MLBridge.prepare_inference_matrix`, then
       call `model.generate_signals(df, ctx, params, artifacts)`.
     - **ML without artifacts:** temporal 80/20 split — fit scaler and train on the first
       80%, generate signals over the full range, logging a warning that proper
       cross-validated training should use `TRAIN` mode.
     - **Rule-based:** call `model.train()` inline then `generate_signals()`. Rule-based
       strategies have no leakage risk from in-sample training.
   - **SignalValidator** coerces the raw output into a `pd.Series[-1..1]` aligned to
     `df_clean.index`, using the manifest's `compression_mode` (`clip`, `tanh`, or
     `probability`).

### 3.4 Metrics (Tearsheet)

`Tearsheet.calculate_metrics` ([engine/core/backtester.py:33](engine/core/backtester.py#L33))
produces two complementary views of performance:

- **Continuous model** — smooth `equity_curve` used as the optimizer's fitness surface.
  Returns are `open.pct_change().shift(-2)` (T+1 execution: signal at T, enter at T+1
  open, exit at T+2 open), multiplied by the raw signal conviction. Friction fires on
  every `signals.diff().abs()` trade.
- **Discrete simulation** — real `-1 / 0 / +1` positions based on an `entry_threshold`,
  a dollar `portfolio` series, a buy-and-hold benchmark `bh_portfolio`, and a
  per-trade `trade_log` DataFrame with entry/exit dates, prices, direction, return, and
  bars held.

Scalars include CAGR, Sharpe, **Deflated Sharpe Ratio (DSR)**, Sortino, Calmar, max
drawdown, win rate, expectancy, profit factor, and trade counts. DSR is computed
immediately after Sharpe via `diagnostics.dsr.compute_dsr`, using the per-period
(daily) return series and the `n_trials` argument passed down from the controller.
Several safeguards exist:

- Bar returns are clipped to `[-0.5, 0.5]` before compounding so gap events can't push
  the equity curve below zero.
- `days` is computed from the median bar duration × bar count so CPCV non-contiguous
  validation folds don't inflate the annualization denominator.
- CAGR exponent is capped at `±10` so short CPCV windows don't annualize tiny gains
  into absurd percentages.

### 3.5 Signal validator

`SignalValidator.validate_and_compress` ([engine/core/backtester.py:730](engine/core/backtester.py#L730))
is the final safety gate before metrics. It:

- Coerces lists, arrays, and mistyped series into `pd.Series[float]`.
- Reindexes to the DataFrame index (or forces it if the lengths match).
- Replaces `Inf/−Inf/NaN` with `0.0` (no conviction).
- Applies a compression mode: `clip` (hard), `tanh` (soft for regression outputs), or
  `probability` (maps `[0, 1]` classifier probs to `[-1, 1]`).

---

## 4. Strategy Workspace

A strategy is a directory under [strategies/](strategies/) with this layout:

```
strategies/<name>/
├── manifest.json        # Declarative config: features, hparams, bounds, training
├── context.py           # AUTO-GENERATED — typed accessor for feature columns
├── model.py             # User's SignalModel subclass
├── artifacts.joblib     # (Optional, ML strategies) persisted scaler + fitted model
├── diagnostics.json     # (Optional) fold diagnostics written by last TRAIN run
└── <helper>.py          # (Optional) extra modules the user imports from model.py
```

### 4.1 `manifest.json`

The single source of truth for a strategy's configuration. Fields:

- `features` — list of `{"id": "RSI", "params": {...}}`. `id` must match a registered
  feature class; `params` override the feature's defaults.
- `hyperparameters` — scalar values available in `model.py` as the `params` dict.
- `parameter_bounds` — optimization search space. List-of-values = grid search; a
  two-element `[lo, hi]` pair = continuous range (used by Optuna).
- `is_ml` — boolean flag switching the trainer/backtester into the ML code paths
  (label construction, pooled fitting, scaler persistence).
- `compression_mode` — `"clip"` (default), `"tanh"`, or `"probability"`.
- `regime_aware` — boolean (default `false`). When `true`, the backtester builds a
  `RegimeContext` after feature computation and passes it to `generate_signals`.
- `regime_detector` — `"vix_adx"` | `"term_structure"` | `"hmm"` (default `"vix_adx"`
  when `regime_aware` is `true`). Selects the regime detector from `REGIME_REGISTRY`.
- `training` — optional block overriding `LocalTrainer.DEFAULT_TRAINING_CONFIG`: `split_method`
  (`cpcv` or `temporal`), `n_groups`, `k_test_groups`, `embargo_pct`, `train_ratio`,
  `price_normalization`, `ffd_d`, `ffd_window`.

Example — a rule-based strategy ([strategies/RSI_Divergence/manifest.json](strategies/RSI_Divergence/manifest.json)):

```json
{
    "features": [
        {"id": "RSI",              "params": {"period": 14}},
        {"id": "Fractals",         "params": {"fractal_n": 4}},
        {"id": "SupportResistance","params": {"method": "Bill Williams", "window": 35, ...}}
    ],
    "hyperparameters": {"low_vol_threshold": 0.25, "atr_lookback": 252, "stop_loss": 0.07, ...},
    "parameter_bounds": {"fractal_n": [3, 5, 8]},
    "compression_mode": "clip",
    "training": {"n_groups": 10, "k_test_groups": 3}
}
```

Example — an ML strategy ([strategies/ml_regime_hybrid/manifest.json](strategies/ml_regime_hybrid/manifest.json)):

```json
{
    "features": [ ... many indicators ... ],
    "is_ml": true,
    "hyperparameters": {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.04, ...},
    "parameter_bounds": {"stop_loss": [0.04, 0.12], "low_vol_threshold": [0.15, 0.35], ...},
    "training": {"price_normalization": "none", "n_groups": 8, "k_test_groups": 2}
}
```

### 4.2 `context.py`

Generated by `WorkspaceManager` ([engine/core/workspace.py](engine/core/workspace.py))
from the Jinja2 template [engine/core/templates/context.py.j2](engine/core/templates/context.py.j2).

Each feature's column name is mapped to a clean Python attribute on the `Context` class so
user code can write `ctx.features.RSI_14` instead of `df["RSI_14"]`. Multi-output features
(MACD, Bollinger Bands) are expanded into one attribute per output (`MACD_LINE`,
`MACD_SIGNAL`, `MACD_HIST`).

**Never edit `context.py` by hand** — run `research_cli.py sync <name>` (or
`python main.py SYNC --strategy <name>`) after any manifest change and the file is
regenerated from scratch.

### 4.3 `model.py`

A user-authored file containing exactly one class extending `SignalModel`
([engine/core/controller.py:55](engine/core/controller.py#L55)).

The contract differs by strategy type:

- **Rule-based** (default). Implement `train()` optionally (returns a static artifacts
  dict — can be empty) and `generate_signals()`. `generate_signals` receives:
  - `df` — the warmup-purged feature DataFrame.
  - `context` — instance of the generated `Context` class (or `None`).
  - `params` — the hyperparameters dict.
  - `artifacts` — whatever `train()` returned.
  - `regime_context` *(optional keyword)* — a `RegimeContext` object if the manifest
    sets `"regime_aware": true` **and** the model's `generate_signals` signature
    declares this parameter. Strategies that omit it receive the legacy 4-argument call.

  It must return a pandas Series (or anything coercible) of conviction values; the
  SignalValidator will compress/bound them.

- **ML strategies** (`is_ml: true`). Implement `build_labels()`, `fit_model()`, and
  `generate_signals()`. The trainer calls `build_labels` once per ticker — so user code
  only ever sees a single asset — then pools X and y across tickers and calls
  `fit_model` once on the stacked matrix. The backtester calls `generate_signals`
  with the loaded `artifacts` (including the fitted scaler in `artifacts["system_scaler"]`).

### 4.4 `WorkspaceManager`

[engine/core/workspace.py](engine/core/workspace.py) handles the lifecycle of these files.

- `sync(features, hparams, bounds)` — writes the manifest, regenerates `context.py`, and
  scaffolds `model.py` from the Jinja2 template **only if it doesn't already exist**
  (existing user code is never overwritten).
- Validates that no hparam key is a reserved Python keyword.
- Computes deterministic attribute names via `_build_features_payload`, stripping
  rendering-only params (`color`, `normalize`, `overbought`, `oversold`) and sanitizing
  non-identifier characters.

---

## 5. Feature System

### 5.1 Registry pattern

Every feature class inherits from `Feature` ([engine/core/features/base.py:114](engine/core/features/base.py#L114))
and self-registers into `FEATURE_REGISTRY` via the `@register_feature("NAME")` decorator.

```python
@register_feature("RSI")
class RSI(Feature):
    name = "Relative Strength Index"
    category = "momentum"
    ...
```

At import time, [engine/core/features/features.py:16](engine/core/features/features.py#L16)
walks the `engine.core.features` package and imports every submodule, triggering the
decorators and populating the registry. Result: any `.py` file added under
`engine/core/features/<category>/` that defines a subclass with the decorator is
immediately discoverable.

Current categories and modules:

- `momentum/` — RSI, MACD, Stochastic, CCI, ROC, Candle Patterns, Fractals
- `trend/` — Moving Averages (SMA/EMA/WMA), ADX, Ichimoku, Supertrend, Linear Regression
- `volatility/` — ATR, Bollinger Bands, Keltner Channels
- `levels/` — Support/Resistance, Fibonacci, KDE-based levels
- `volume/` — OBV, Volume, VWAP, Anchored VWAP, Volume Profile, Volume Z-score
- `calendar/` — Weekly and Yearly Cycle features
- `comparison/` — Ticker Comparison (cross-asset relative strength)

### 5.2 Output schema

Every feature declares a list of `OutputSchema` objects
([engine/core/features/base.py:67](engine/core/features/base.py#L67)) describing *what
kind of data* it produces. Seven structural types:

| Type       | Shape                                     | Example          |
|------------|-------------------------------------------|------------------|
| `LINE`     | `pd.Series`, one value per bar            | MA, RSI          |
| `LEVEL`    | list of `{value, label, strength}`        | Support/Resist   |
| `BAND`     | references two LINE outputs (upper+lower) | Bollinger Bands  |
| `HISTOGRAM`| signed magnitude per bar                  | MACD histogram   |
| `MARKER`   | sparse, NaN-gapped events                 | Fractals         |
| `ZONE`     | list of `{start, end, upper, lower}`      | Volume Profile   |
| `HEATMAP`  | 2D intensity grid                         | KDE levels       |

Only `LINE`, `HISTOGRAM`, and `MARKER` contribute columns to the feature DataFrame; the
other types are consumed by the GUI for rendering (out of scope here but the engine
doesn't drop them — they flow through `FeatureResult`).

### 5.3 Computation

`FeatureOrchestrator.compute_features` ([engine/core/features/features.py:142](engine/core/features/features.py#L142))
iterates the manifest's `features` list, instantiates each class, calls `compute(df,
params, cache)`, and concatenates every resulting series into a wide DataFrame.

- **Deterministic column names.** `Feature.generate_column_name(fid, params, output)`
  produces column names like `RSI_14` or `MACD_SIGNAL`, sanitizing non-alphanumeric chars.
- **Shared cache.** A `FeatureCache` is passed to each `compute()` call so that a feature
  dependency (e.g. MACD needing two EMAs) can pull already-computed series from memory
  instead of recalculating.
- **Memory-safety invariant.** The orchestrator snapshots `len(df.columns)` before and
  after every `compute` call; if a feature mutated the input frame in place, it raises a
  `FeatureError`. All features must return new `Series` objects.
- **`l_max` discovery.** While iterating, any param key matching
  `{window, period, slow, fast, lookback}` updates a running max. This single number is
  returned alongside the DataFrame and used to purge warmup rows.

### 5.4 Non-stationarity hook

`Feature.non_stationary_outputs(params)` lets a feature declare which of its output columns
are *non-stationary* under the current parameters (e.g. a raw moving average tracks price
levels). Non-stationary columns are routed through **fractional differentiation** before
the `MinMaxScaler` sees them — this preserves long memory while keeping train/test
distributions compatible. Used by `MLBridge.collect_non_stationary_columns` during
training.

---

## 6. Data Broker

[engine/core/data_broker/data_broker.py](engine/core/data_broker/data_broker.py) wraps
yfinance behind a SQLite cache. `DataBroker.get_data(ticker, interval, start, end)` is the
single entry point.

### 6.1 Caching protocol

- **Daily / weekly intervals** are stored with `_EPOCH_START = 1900-01-01` as the lower
  bound so the entire available history is always on disk. The requested `start` only
  filters the returned slice.
- **Intraday intervals** are clamped to Yahoo Finance's hard per-interval lookback limits
  (7 days for 1m, 60 days for 2m–30m, 730 days for 60m/1h).
- **15m is bypass-only.** Too short-lived to be worth caching; always fetches direct from
  yfinance.
- `_compute_fetch_range` detects four cases: DB empty, forward gap only (cache is stale),
  backward gap only (requested earlier history than cached), or both. Only the missing
  range is fetched and merged via `INSERT ... ON CONFLICT DO NOTHING`.
- `_STALENESS_WINDOW` defines how old the last bar can be before a refresh fires (4 days
  for daily, accounting for weekends/holidays; 10 days for weekly; 2 days for hourly).

### 6.2 Schema

[engine/core/data_broker/database.py](engine/core/data_broker/database.py) declares an
`OHLCV` table keyed by `(ticker, timestamp, interval)`. Rows are inserted in batches of
124 to respect SQLite's 999-bound-parameter limit.

---

## 7. Training Pipeline

`LocalTrainer` ([engine/core/trainer.py](engine/core/trainer.py)) owns the data pipeline
around a strategy's `train()` or `build_labels()/fit_model()` methods. It accepts either
a DataFrame (single-ticker, legacy) or a `{ticker: df}` dict (multi-ticker, ML-only).

### 7.1 Single-ticker flow (`_run_single`)

1. `compute_all_features` → `(df_full, l_max)`; then `df_clean = df_full.iloc[l_max:]`.
2. Apply `price_normalization` (log returns or FFD) if set.
3. Apply FFD to any feature columns self-declared as non-stationary.
4. Load `model.py` and `context.py` dynamically (via `importlib.util`).
5. Generate splits — `temporal` (single 80/20 chronological cut) or `cpcv` (combinatorial
   purged CV, see §8).
6. For each fold: if ML, fit scaler on train only and transform val; then call
   `model.train(df_train)` (rule-based) or `build_labels → fit_model` (ML). Compute
   train/val tearsheets for the fold.
7. **Final retrain.** For CPCV (multi-fold), fit a fresh model on the full dataset so
   the persisted artifacts use all available data. For temporal, the single fold's
   artifacts are kept.
8. **Fold diagnostics (CPCV only).** `_compute_fold_diagnostics` extracts per-fold
   OOS Sharpe values and computes `fraction_positive_folds`, `fraction_above_half_folds`,
   and `spearman_is_oos` (Spearman rank correlation between IS and OOS Sharpe across
   folds). Results are written to `strategies/<name>/diagnostics.json` via
   `_save_diagnostics` and returned in the results dict under `fold_diagnostics`.
9. Persist FFD metadata and artifacts via `ArtifactManager.save_artifacts`.

### 7.2 Multi-ticker flow (`_run_multi`)

ML-only (rule-based strategies must train one ticker at a time). The flow:

1. Per-ticker: compute features, purge, apply price normalization, apply FFD per-ticker
   (FFD is path-dependent so this *must* happen on contiguous data). Assert every ticker
   produced the same feature column list.
2. `_split_cpcv_by_date` — partition the sorted union of all ticker dates into N contiguous
   groups, generate all `C(N, k)` combinations. Embargo removes train observations adjacent
   to each test block.
3. For each fold: each ticker independently calls `build_labels`. `(X, y)` pairs are pooled
   across tickers; one scaler and one `fit_model` call per fold.
4. Evaluate per-ticker within the fold, average scalar metrics across tickers; then
   aggregate mean/std across folds.
5. Final retrain on the full pooled dataset, persist.
6. Fold diagnostics written to `strategies/<name>/diagnostics.json` (same as
   single-ticker, §7.1 step 8).

### 7.3 Feature analysis

After every ML fit, `_compute_feature_analysis` produces a per-feature diagnostic table
combining:

- `feature_importances_` from the fitted model (if available).
- Pearson (linear) and Spearman (monotonic) correlations of each feature with the target.
- Mutual information (nonlinear dependence) via `sklearn.feature_selection.mutual_info_classif`.

Rows are sorted by importance (or `|pearson|` as fallback) and printed at the end of
training, so authors can see which features actually carry signal the model can exploit.

---

## 8. Cross-Validation (CPCV)

Standard k-fold CV is invalid for time series because folds are interleaved and labels
look forward. `CPCVSplitter` ([engine/core/optimization/cpcv_splitter.py](engine/core/optimization/cpcv_splitter.py))
implements Lopez de Prado's Combinatorial Purged CV:

1. Partition the dataset into `n_groups` contiguous groups.
2. Enumerate all `C(n_groups, k_test_groups)` combinations; each combination forms one
   fold's test set.
3. **Purge** — remove training rows within `l_max` bars before each test block
   (prevents feature-lookback leakage into the test set).
4. **Embargo** — remove training rows within `embargo_pct × n` bars *after* each test
   block (prevents autocorrelated-label leakage).

This produces many more folds than walk-forward (6 groups with k=2 → 15 folds) and
provides tighter error bars on validation metrics. The cost is computational — each fold
runs a full training pass.

---

## 9. ML Bridge

[engine/core/ml_bridge/orchestrator.py](engine/core/ml_bridge/orchestrator.py) is the
preprocessing boundary between raw features and ML models. Key methods:

- `prepare_training_matrix(df, feature_cols, l_max)` — purges `l_max` rows and fits a
  `MinMaxScaler(feature_range=(-1, 1))` on the feature columns. Returns
  `(scaled_df, scaler)`; the scaler **must** be persisted in artifacts.
- `prepare_inference_matrix(df, feature_cols, l_max, artifacts, is_live)` — never fits,
  only transforms via `artifacts["system_scaler"]`. Replays FFD on any
  `artifacts["ffd_columns"]` so the distribution matches what the scaler was fit on.
- `apply_price_normalization(df, method)` — `log_returns` (drops first row) or `ffd`
  (log-diff with fixed-window fractional differentiation, drops `window-1` rows).
- `apply_ffd_to_dataframe(df, columns, d, window)` — fixed-window FFD
  (Lopez de Prado Ch. 5) with default `d=0.4, window=10`. Path-dependent so must be
  applied per-ticker on contiguous data.

### 9.1 Artifacts

`ArtifactManager` ([engine/core/ml_bridge/artifact_manager.py](engine/core/ml_bridge/artifact_manager.py))
pickles the artifacts dict to `<strategy_dir>/artifacts.joblib`. The dict typically
contains:

- `model` — the fitted estimator (e.g. `XGBClassifier`).
- `system_scaler` — the fitted `MinMaxScaler`.
- `feature_cols` — list of feature column names as ordered during training.
- `feature_analysis` — the per-feature diagnostics list.
- `ffd_columns`, `ffd_d`, `ffd_window` — metadata to replay FFD at inference.

At backtest/signal time, if the strategy is `is_ml: true` and no artifacts were passed in,
the backtester checks disk first (`ArtifactManager.load_artifacts`) before falling back
to the inline 80/20 training shortcut.

---

## 10. Hyperparameter Optimization

`OptimizerCore` ([engine/core/optimization/optimizer_core.py](engine/core/optimization/optimizer_core.py))
wraps Optuna for continuous bounds and a joblib grid search for discrete bounds. Only runs
during `TRAIN` when both (a) `parameter_bounds` contains at least one multi-value entry
and (b) `controller.ENABLE_HPO` is `True`.

- Permutations are capped at `PERMUTATION_LIMIT = 5000`.
- Workers share the dataset via `joblib`'s shared-memory mechanism
  ([engine/core/optimization/local_cache.py](engine/core/optimization/local_cache.py))
  so each trial doesn't copy a large OHLCV frame.
- Objective: Sharpe ratio of a T+1-executed strategy signal over the full dataset. The
  objective matches `Tearsheet.calculate_metrics` so optimizer and evaluator agree.
- **Known limitation:** HPO currently runs on the first ticker only even when training
  is multi-ticker — marked as a TODO. HPO is also disabled by default
  (`ENABLE_HPO = False`) pending further work.

Optuna results are merged with the manifest's `hyperparameters` dict before Phase B
training begins.

---

## 11. Daemon / API Layer

### 11.1 FastAPI server

[engine/daemon/main.py](engine/daemon/main.py) exposes:

- `GET /health` — verifies Redis connectivity.
- `GET /api/v1/strategies` — scans the strategies directory and returns every valid
  manifest (with the folder name attached as `id`). Malformed manifests are logged and
  skipped rather than crashing the endpoint.
- `POST /api/v1/jobs` — accepts a `JobPayloadRequest`, creates a `JobRegistry` entry in
  Redis (status `QUEUED`), and enqueues `tasks.process_job` on the `default` RQ queue.
- `GET /api/v1/jobs/{job_id}` — polls the job's Redis hash for `status`, `progress`, and
  `artifact_path`.
- `POST /api/v1/jobs/{job_id}/cancel` — sets status to `CANCEL_REQUESTED`; workers check
  this before starting and at commit time.

### 11.2 RQ worker

[engine/daemon/worker.py](engine/daemon/worker.py) is a thin RQ launcher.
`tasks.process_job(job_id, payload)` ([engine/daemon/tasks.py](engine/daemon/tasks.py))
is the actual unit of work:

1. Check for `CANCEL_REQUESTED` before starting. If set, mark `CANCELLED` and return.
2. Set status to `RUNNING`, instantiate `ApplicationController`, call `execute_job(payload)`.
3. Call `safe_commit_results` which uses Redis `WATCH`/`MULTI` (optimistic locking) to
   atomically commit status + results, aborting cleanly if a cancel came in mid-flight.
4. Results > 1 MB are spilled to `artifacts/<job_id>.json` and Redis stores the path
   (`artifact_value = "FILE_PATH:..."`) instead of the raw payload.
5. On exception: traceback recorded in `error_log`, status set to `FAILED` (unless the
   job was cancelled concurrently).

### 11.3 State model

[engine/daemon/models.py](engine/daemon/models.py):

- `JobStatus` — `QUEUED → RUNNING → COMPLETED | FAILED | CANCELLED`.
- `JobRegistry` — Pydantic model with `job_id` (UUID), `strategy_name`, `status`,
  `progress`, `parameters`, `artifact_path`, `error_log`.

All fields are stored as a single Redis hash keyed by `job:{job_id}`.

---

## 12. Portfolio Backtester

[engine/core/portfolio_backtester.py](engine/core/portfolio_backtester.py) is an
event-driven multi-asset simulator (as opposed to the vectorized single-asset tearsheet).
Uses the same T+1 execution model but adds:

- **Position sizing** via the 2% risk rule — never risk more than
  `risk_per_trade_pct` of portfolio on a single stop-out, scaled by signal strength,
  and capped by `max_position_pct`.
- **Eviction logic** — new signals must exceed the weakest held position by
  `eviction_margin` before the weakest is kicked out.
- **Optional rebalancing on signal strength** (disabled by default).

Currently accessed through the GUI's portfolio panel; `_handle_backtest` in the controller
still raises `NotImplementedError` for `PORTFOLIO` mode via the CLI path.

---

## 13. Named Universes

[engine/core/universes.py](engine/core/universes.py) defines static ticker universes
(`MEGA_CAP_TECH`, `DOW_30`, `SECTOR_ETFS`, `LIQUID_ETFS`, `SEMICONDUCTORS`,
`FINANCIALS_LARGE`, `HEALTHCARE_PHARMA`, etc.) that can be passed via
`research_cli.py`'s `--universe <NAME>` flag to expand into a ticker list. Flagged as a
TODO because the current lists are survivor-biased — they're frozen snapshots of current
index members rather than point-in-time constituents.

---

## 14. Phase 0 Diagnostics

Tools for answering whether a strategy has genuine edge or is fitted noise.
All live under `engine/core/diagnostics/` and `data/diagnostics.db`.
Phase 1 (signal identification before strategy construction) is covered in §15.

### 14.1 Trial Counter (`diagnostics/trial_counter.py`)

SQLite table (`data/diagnostics.db :: trial_counts`) that records how many times
each strategy has been backtested or trained. The `ApplicationController` calls
`increment(strategy_name, "backtest"|"train")` at the start of every run. The count
is then passed to `Tearsheet.calculate_metrics` as `n_trials` so the DSR benchmark
grows with each additional trial.

Key functions:

- `increment(strategy_name, mode)` — upserts `backtest_count` or `train_count` (+1).
- `get_total_trials(strategy_name) → int` — returns `backtest_count + train_count`
  (returns `1` if the strategy has never been recorded, a safe DSR default).
- `get_all() → list[dict]` — all rows; used by the `trial-counts` CLI command.
- `set_counts(strategy_name, backtest_count, train_count)` — manual backfill or
  correction.

### 14.2 Deflated Sharpe Ratio (`diagnostics/dsr.py`)

Implements Bailey & López de Prado (2014). DSR adjusts the observed Sharpe ratio for:

1. **Non-normal return distribution** — uses the Mertens (2002) variance estimator:
   `var(SR̂) ≈ (1/T)(1 − γ₃·SR + (γ₄−1)/4·SR²)` where γ₄ is *raw* kurtosis
   (pandas `.kurtosis()` returns excess kurtosis — `+3.0` is added internally).
2. **Multiple trials** — the SR* benchmark (expected max SR under H0) grows
   logarithmically with `n_trials` via `sr_star(n_trials, n_obs)`.

`compute_dsr(returns, n_trials) → float [0, 1]`:
- Uses per-period (daily) SR, not annualised.
- Returns `NaN` if fewer than 20 observations or degenerate distribution.
- Threshold guidance: DSR > 0.95 = strong evidence; > 0.5 = passes basic bar;
  < 0.5 = likely lucky noise.

### 14.3 Fold Diagnostics (`LocalTrainer._compute_fold_diagnostics`)

After every multi-fold (CPCV) training run, the trainer computes:

- `fold_sharpes` — list of OOS Sharpe per fold.
- `fold_train_sharpes` — list of IS Sharpe per fold.
- `fraction_positive_folds` — fraction of folds with OOS Sharpe > 0.
- `fraction_above_half_folds` — fraction of folds with OOS Sharpe > 0.5.
- `spearman_is_oos` — Spearman rank correlation between IS and OOS Sharpe across
  folds (positive = IS performance predicts OOS; near-zero or negative = overfitting).

Results are persisted to `strategies/<name>/diagnostics.json` with a UTC timestamp
and returned in the `trainer.run()` result dict under `fold_diagnostics`. The
`diagnose` CLI command reads this file to populate Section 2 of its report.

### 14.4 Research CLI Diagnostic Commands

Four new commands added to `research_cli.py`:

| Command | Purpose |
|---------|---------|
| `trial-counts [--set STRATEGY --backtest-count N --train-count N]` | View or manually set trial counts for any strategy |
| `sensitivity <name> --tickers … --interval …` | 7-step sweep across top-3 bound params; prints ASCII Sharpe-vs-param table. Bypasses controller so trial counter is not inflated. |
| `signal-stability <name> --tickers … --interval …` | Generates signals over full period and 3 five-year subsets; reports Pearson correlation per window (< 0.7 = unstable). |
| `diagnose <name> --tickers … --interval …` | One-page per-strategy report covering core metrics+DSR, fold distribution (from diagnostics.json), signal stability, optional sensitivity. Emits KEEP / INVESTIGATE / RETIRE verdict (≥2 criteria failing → RETIRE). |

---

## 15. Phase 1: Signal Identification Framework (IC Analytics)

Tools for answering whether a candidate signal has genuine edge before it is built into a
strategy. All live under `engine/core/analytics/` and are exposed through two new
`research_cli.py` commands: `ic` and `ic-surface`.

The evaluation workflow is a mandatory gate between feature selection and strategy
construction:

```
Candidate signal
    │
    ▼
Unconditional IC analysis  (ic command)
    │
    ├─ IC < 0.02 or IC-IR < 0.3 at 5-day horizon  →  DISCARD
    │
    └─ Passes gate  →  proceed
        │
        ▼
    Conditional IC surface  (ic-surface command)
        │
        ├─ flat       →  signal works everywhere; no regime layer needed
        ├─ structured  →  build strategy with regime weighting from the surface
        └─ noisy       →  wrong state space, or signal is fragile; investigate
        │
        ▼
    Strategy construction  →  CPCV backtest + DSR
```

A signal that does not pass the unconditional IC gate is discarded. No exceptions.

### 15.1 `ICAnalyzer` (`analytics/ic_analyzer.py`)

Computes cross-sectional rank IC for a set of per-ticker signal series and price DataFrames.

**Cross-sectional IC:** at each date t, collect `(signal[ticker][t], fwd_return[ticker][t])`
pairs across all tickers in the universe, rank them, and compute Spearman rank correlation.
Averaged over all dates → mean IC. Ratio of mean to std → IC-IR.

`ICResult` carries:

- `mean_ic`, `ic_ir`, `ic_pos_frac` — scalars per horizon [1, 5, 10, 20, 60].
- `ic_series` — raw daily IC `pd.Series` per horizon (consumed by `ConditionalIC`).
- `quintile_returns`, `quintile_monotonic`, `quintile_spread` — Q5−Q1 annualized spread
  and monotonicity check at each horizon.
- `daily_turnover` — mean `|Δsignal| / 2` across tickers. > 0.5 flags that the signal
  cannot survive realistic transaction costs.
- `half_life_days` — first horizon at which mean IC drops to half its peak value. Signals
  the natural rebalancing frequency.
- `period_ic_consistency` — ratio of mean IC in first vs second half of the period
  (1.0 = consistent; lower = time-varying or fragile).

IC benchmarks: mean IC > 0.02 is detectable; > 0.05 is strong. IC-IR > 0.5 is reasonably
consistent. These thresholds govern the `passes_gate` property.

### 15.2 `MacroFetcher` and `MacroSpec` (`analytics/macro_fetcher.py`)

Fetches macro series from FRED (via the existing `DataFetcher.fetch_macro_data()`) or
yfinance, and returns a daily-frequency forward-filled `pd.Series`.

CLI format for specifying macro dimensions: `SERIES_ID:SOURCE[:N_BINS]`

```
^VIX:yf:5        CBOE VIX index via yfinance, 5 quantile bins
T10Y2Y:fred:3    10Y-2Y yield spread from FRED, 3 bins
^TNX:yf          10Y Treasury yield, default 4 bins
BAMLH0A0HYM2:fred  High Yield OAS (credit stress), default 4 bins
```

`parse_macro_spec(s)` converts the CLI string to a `MacroSpec` dataclass. `MacroFetcher`
handles timezone normalization and forward-fill to the IC series' trading-day index.

### 15.3 `ConditionalIC` (`analytics/conditional_ic.py`)

Takes the raw daily IC series at a single horizon and a list of `(macro_series, MacroSpec)`
pairs, bins each macro variable into quantile regimes, and computes IC per regime cell
(cross-product for multi-dimensional surfaces).

`ConditionalICResult` carries:

- `bins` — `List[RegimeBin]` sorted by mean IC descending. Each `RegimeBin` records
  `mean_ic`, `ic_ir`, `n_obs`, and the quantile boundaries for each macro dimension.
- `diagnosis` — `flat | structured | noisy | insufficient_data`.
  - **flat:** IC range across bins < 0.02 — signal works uniformly, no regime layer needed.
  - **structured:** best-bin IC > unconditional IC + 0.02 — regime weighting will help.
  - **noisy:** range exists but no coherent lift — macro state space may be wrong.
- `unconditional_ic` — the mean IC over the full period (for comparison with regime ICs).

Bins with fewer than `min_obs_per_bin` (default 20) observations are excluded. Falls back
from quantile-cut to equal-width cut if the macro variable has too few unique values.

### 15.4 Report renderer (`analytics/report.py`)

`render_ic_report(result)` and `render_conditional_ic_report(result, signal_name)` produce
the standardized evaluation reports. The unconditional report ends with a `VERDICT` line:

```
VERDICT: PROCEED -> CONDITIONAL IC SURFACE  (strong edge)
VERDICT: PROCEED -> CONDITIONAL IC SURFACE  (weak but detectable edge)
VERDICT: DISCARD  (IC=0.0083 < 0.02 or IC-IR=0.1412 < 0.3)
```

The conditional report identifies the best regime and includes a one-line instruction for
how to apply regime weighting in `model.py`.

### 15.5 CLI commands

```bash
# Step 1 — unconditional gate check
uv run python research_cli.py ic <strategy> \
    --tickers AAPL,MSFT,GOOGL,...  \   # 20+ tickers for reliable cross-sectional IC
    --interval 1d                  \
    --start 2015-01-01             \
    [--horizons 1 5 10 20 60]      \
    [--save]                           # writes strategies/<name>/ic_report.txt

# Step 2 — conditional surface (runs unconditional pass first, aborts if gate fails)
uv run python research_cli.py ic-surface <strategy> \
    --tickers AAPL,MSFT,GOOGL,...  \
    --interval 1d                  \
    --start 2015-01-01             \
    --macro '^VIX:yf:5' 'T10Y2Y:fred:3' \
    [--primary-horizon 5]          \   # which horizon to use for the surface
    [--min-obs 20]                 \   # min observations per regime bin
    [--force]                      \   # bypass the IC gate
    [--save]                           # writes ic_report.txt and ic_surface_<dims>.txt
```

The `ic-surface` command automatically runs the unconditional IC pass first and enforces the
gate — the surface is not computed if IC fails (override with `--force`).

---

## 16. Phase 3 — Regime Detection Subsystem

### 16.1 Architecture

`engine/core/regime/` is structured parallel to `engine/core/features/`:

```
engine/core/regime/
├── base.py         RegimeDetector ABC, REGIME_REGISTRY, register_regime(), RegimeContext
├── rule_based.py   VixAdxRegime (3-state), TermStructureRegime (2-state)
├── hmm.py          GaussianHMMRegime — causal forward algorithm over hmmlearn fit
├── bocpd.py        BayesianCPD — Adams-MacKay BOCPD for novelty scoring
├── orchestrator.py RegimeOrchestrator — builds macro features, runs detector + BOCPD
└── __init__.py     Auto-registers all three detectors on import
```

All detectors inherit `RegimeDetector` and register via `@register_regime("key")`, identical
in pattern to `@register_feature`.

### 16.2 The Three Layers (in implementation order)

**Layer 1 — Rule-based** (no training, use as validation baseline)

Two deterministic detectors produce one-hot probability vectors shaped identically to
probabilistic detectors so all downstream code is type-agnostic:

- `VixAdxRegime` (`"regime_detector": "vix_adx"`) — 3 states using ^VIX and ADX:
  - State 0 *low-vol trending*: VIX < 15 AND ADX > 25 → favour momentum
  - State 1 *normal ranging*: VIX 15–25 AND ADX < 20 → favour mean-reversion
  - State 2 *stressed*: VIX > 25 → cut size, favour contrarian
- `TermStructureRegime` (`"regime_detector": "term_structure"`) — 2 states using the
  VIX/VIX3M ratio with linear blending in the transition band [0.95, 1.00]:
  - State 0 *calm contango*: ratio < 0.95
  - State 1 *stressed backwardation*: ratio > 1.00

**Layer 2 — HMM** (`"regime_detector": "hmm"`)

`GaussianHMMRegime` fits a 3-state Gaussian HMM on four macro features (SPY log-returns,
21-day realised vol, 5-day HYG log-return inverted as HY-spread proxy, VIX level) via
`hmmlearn.GaussianHMM`. Critical implementation detail: inference uses the **forward
algorithm**, not Viterbi. Viterbi decodes the globally most-likely path using the
*entire* sequence — lookahead bias. The forward algorithm produces P(state_t | x_{1:t}),
which is causal.

Post-fit: exponential smoothing (halflife = 4 days) suppresses noisy daily flipping.
Label consistency across retrainings is maintained by matching new emission means to the
previous fit via nearest-neighbour pairing — prevents the "state 0 in January = state 1
in February" problem that would corrupt per-regime statistics.

**Layer 3 — BOCPD** (always runs alongside the selected detector)

`BayesianCPD` (in `bocpd.py`) implements Adams & MacKay (2007) with a Normal-Gamma
conjugate prior on standardised VIX. At each step it maintains a probability distribution
over run lengths and returns P(run_length < 5 days) per bar. This is the novelty score:

- P < 0.5: the process is running stably in a known regime
- P > 0.5: the data-generating process likely changed in the last 5 days ("structural break")

When a structural break fires, `RegimeContext.size_multiplier` returns 0.5 for the next
10 bars, giving strategies a simple way to implement the spec's "reduce position sizes
by 50% for 10 trading days as a confirmation buffer."

For HMM strategies, the BOCPD score becomes the `novelty` field in `RegimeContext`. For
rule-based strategies, novelty = max(0, BOCPD) (rule-based detectors return constant 0
from their own `novelty_score()` since they have no probabilistic model).

### 16.3 Macro Feature Assembly

`RegimeOrchestrator._build_macro_features(df)` fetches external data and computes ADX
internally. All columns are aligned to `df.index` via forward-fill:

| Column         | Source                                 | Used by                          |
|----------------|----------------------------------------|----------------------------------|
| `vix`          | `^VIX` yfinance                        | VixAdxRegime, HMM, BOCPD         |
| `vix3m`        | `^VIX3M` yfinance                      | TermStructureRegime              |
| `vix_vix3m`    | `vix / vix3m`                          | TermStructureRegime              |
| `adx`          | Wilder ADX from df's own OHLCV         | VixAdxRegime                     |
| `spy_ret`      | SPY log-returns yfinance               | HMM                              |
| `spy_rvol`     | 21-day rolling std of spy_ret          | HMM                              |
| `hy_spread_chg`| Inverted HYG 5-day log-return          | HMM                              |

Failures in external fetches are caught and logged; the orchestrator returns whatever
columns it could successfully assemble rather than aborting.

### 16.4 `RegimeContext` — the payload

```python
@dataclass
class RegimeContext:
    detector_name: str          # key from REGIME_REGISTRY
    proba: pd.DataFrame         # P(state=k | x_{1:t}); columns = int state IDs, index = df.index
    labels: pd.Series           # argmax regime per bar (int)
    novelty: pd.Series          # BOCPD P(run_length < 5), in [0, 1]
    n_states: int
    ic_weight: Optional[pd.Series] = None  # reserved for Phase 3.4 conditional IC weighting

    # Convenience
    def current_regime() -> int
    def current_proba() -> Dict[int, float]
    def is_novel(threshold=0.5) -> bool
    def size_multiplier -> pd.Series   # 1.0 normally; 0.5 for 10 bars after novelty > 0.5
```

### 16.5 Opt-in Pattern

**manifest.json:**
```json
{
    "regime_aware": true,
    "regime_detector": "vix_adx"
}
```

**model.py** — declare `regime_context` to receive it; omit to stay unaffected:
```python
def generate_signals(self, df, context, params, artifacts, regime_context=None):
    raw = ...  # compute raw signals
    if regime_context is not None:
        raw *= regime_context.size_multiplier          # halve size during structural breaks
        stressed = (regime_context.labels == 2).values
        raw[stressed] *= 0.5                           # further cut in stressed regime
    return raw
```

`LocalBacktester._call_generate_signals` uses `inspect.signature` to detect whether the
model accepts `regime_context` before passing it — existing strategies need no changes.

### 16.6 IC Surface Weighting (Phase 3.4 — groundwork laid)

The `ic_weight` field in `RegimeContext` is reserved for the operational use of the
conditional IC surface (computed offline via `research_cli.py ic-surface`). The design:
when the current macro state's regime bin has IC = 0.08 the signal gets full weight;
when IC = −0.02 the signal gets zero weight. This is not yet wired into the runtime
(the offline analysis tooling in `analytics/conditional_ic.py` is complete; the
lookup-at-inference step is a pending addition to `RegimeOrchestrator`).

---

## 17. Supporting Modules

- **`engine/core/logger.py`** — sets up two loggers: `logger` for engine-wide messages
  and `daemon_logger` for API/worker output. Both write to `engine/logs/`.
- **`engine/core/exceptions.py`** — `EngineError` base class with `DataError`,
  `StrategyError`, `FeatureError`, `ValidationError` subclasses. Controllers catch these
  at the top level and convert them to CLI errors or API 4xx responses.
- **`engine/core/config.py`** — environment-driven config: `ENGINE_API_HOST/PORT`,
  `REDIS_HOST/PORT/DB`, `FRED_API_KEY`, `STRATEGIES_FOLDER`. All overrideable via env
  vars; Docker Compose sets sensible defaults.
- **`engine/core/templates/`** — Jinja2 templates for generated `context.py` and
  scaffolded `model.py`. Edited only when the strategy contract evolves.
- **`engine/core/bridge.py`** / **`engine/__init__.py`** — the `ModelEngine` facade used
  by the GUI. Hides the controller behind a simpler API. Out of scope for this doc but
  worth knowing it exists.
- **`engine/transit/`** — scratch space for data exports between processes; referenced
  by the optimizer's shared-memory cache.

---

## 18. Execution Matrix

| Mode             | Features recomputed? | Training happens? | Scaler fit? | Artifacts persisted? |
|------------------|----------------------|-------------------|-------------|----------------------|
| `BACKTEST` (rule)| Yes                  | Yes (inline)      | n/a         | No                   |
| `BACKTEST` (ML, artifacts on disk) | Yes | No | No (reused)  | No                   |
| `BACKTEST` (ML, no artifacts)      | Yes | Yes (80% split) | Yes (80% only) | No              |
| `TRAIN` (rule)   | Yes                  | Yes (per-fold + final) | n/a   | Yes                  |
| `TRAIN` (ML)     | Yes                  | Yes (per-fold + final) | Yes (per-fold + final) | Yes    |
| `SIGNAL_ONLY`    | Yes (same as BACKTEST)| Same as BACKTEST | Same as BACKTEST | No                 |

---

## 19. Key Invariants to Respect

These are the load-bearing contracts of the system. Breaking any of them causes silent
correctness bugs rather than loud failures:

1. **T+1 execution.** Signal at bar T is entered at bar T+1 open; returns accrue from
   T+1 open to T+2 open. `Tearsheet` and `OptimizerCore` both encode this — any new
   execution simulator must match.
2. **Warmup purge before any downstream step.** `df_clean = df_full.iloc[l_max:]` happens
   exactly once, immediately after feature computation. Nothing downstream (including the
   user's `train`/`generate_signals`) should see pre-warmup rows.
3. **Features must not mutate the input DataFrame.** The orchestrator's column-count
   invariant catches this; violations raise `FeatureError`.
4. **The scaler is fit once, on training data only.** Every `prepare_inference_matrix`
   call is `transform`-only. Persisted in `artifacts["system_scaler"]`.
5. **Signals are bounded `[-1, 1]`.** `SignalValidator` enforces this; any user function
   that bypasses it will corrupt the tearsheet.
6. **`context.py` is generated output.** Regenerate via `SYNC` after every manifest change.
7. **FFD is path-dependent.** Apply it per-ticker on contiguous time series, before any
   splitting.
8. **Artifacts are the source of truth for inference.** Anything the backtester needs to
   replay training-time decisions (scaler, FFD columns, feature order) must live in
   `artifacts` so it's pickled along with the model.

---

## 20. Example Strategies

Illustrative examples from [strategies/](strategies/):

- **[RSI_Divergence](strategies/RSI_Divergence/)** — rule-based, uses RSI + Fractals +
  Support/Resistance. Simple `parameter_bounds` (grid-search over `fractal_n`). Custom
  CPCV config (`n_groups=10, k_test_groups=3`).
- **[consolidation_breakout](strategies/consolidation_breakout/)** — rule-based, uses
  ATR + two SMAs + Volume + `TICKER_COMPARE` (cross-asset feature comparing against SPY).
  Large continuous `parameter_bounds` — an ideal HPO candidate.
- **[volitility_regime_identifier](strategies/volitility_regime_identifier/)** —
  rule-based, volatility-regime classifier using EMAs + ATR + YearlyCycle + S/R.
- **[ml_XGBoost](strategies/ml_XGBoost/)** — `is_ml: true`. Normalized MAs + RSI + ATR
  + YearlyCycle as features; `price_normalization: log_returns`; XGBoost hyperparameters
  (`n_estimators`, `max_depth`, `min_child_weight`, etc.) live in `hyperparameters`.
- **[ml_regime_hybrid](strategies/ml_regime_hybrid/)** — `is_ml: true`. The most complex
  example: 15 features (multiple MAs at different periods, BB, ATR, ADX, ROC, OBV,
  Support/Resistance), both trained bounds and classical thresholds, helper modules
  (`regime.py`, `confirmations.py`) that `model.py` imports for clarity.

---

## 21. Testing

All tests live under [engine/tests/](engine/tests/) and run inside the `research_tester`
Docker container:

```bash
docker compose --profile test up -d pytest
docker exec research_tester uv run pytest tests/
```

Covers the data broker, backtester pipeline, API endpoints, feature blast shield
(memory-safety enforcement), and the CPCV splitter. Test discovery is configured in
[engine/pytest.ini](engine/pytest.ini).
