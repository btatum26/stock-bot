# Research CLI

`research_cli.py` is the LLM-friendly command-line interface for the research engine.  
Run all commands from the **repo root** with `uv run python research_cli.py <command>`.

It uses `ModelEngine` (the same facade the GUI uses) so strategy state is always consistent between the two.

---

## Quick Reference

| Command | Purpose |
|---|---|
| `list` | List all strategies |
| `features` | List every registered feature ID, description, and default params |
| `inspect <strategy>` | Show full manifest for a strategy |
| `init <strategy>` | Scaffold a new strategy workspace |
| `edit <strategy>` | Modify manifest and auto-sync context.py |
| `show-context <strategy>` | Print generated context.py (attribute names for model.py) |
| `show-model <strategy>` | Print current model.py |
| `data-info` | Show cached OHLCV tickers, intervals, and date ranges |
| `validate <strategy>` | Import-check model.py without running data |
| `backtest <strategy>` | Run a vectorized backtest with tearsheet output |
| `portfolio <strategy>` | Run a multi-asset portfolio backtest with full tearsheet |
| `train <strategy>` | Run hyperparameter optimisation (Optuna/CPCV) |
| `signal <strategy>` | Generate current live signals |

---

## Strategy Authoring Workflow

This is the intended loop for creating a new strategy from scratch.

### 1. Discover available features

```bash
uv run python research_cli.py features
```

Each entry shows the feature ID (used in the manifest), a description, and default parameter values.  
Use `--json` to get machine-readable output:

```bash
uv run python research_cli.py features --json
```

### 2. Check what data is cached

Before choosing a date range, confirm the ticker and interval are available:

```bash
uv run python research_cli.py data-info --ticker AAPL
```

Output shows bar count and first/last dates per interval so you can pick a valid `--start`/`--end` window.

### 3. Scaffold the strategy

```bash
uv run python research_cli.py init my_strategy
```

Creates `strategies/my_strategy/` with a default manifest, a generated `context.py`, and a blank `model.py`.

### 4. Configure features and hyperparameters

Add features one at a time. Feature params are `k=v` pairs matching the `Defaults:` shown by `features`:

```bash
# Add RSI with a custom period
uv run python research_cli.py edit my_strategy --add-feature RSI --feature-params period=14

# Add a moving average
uv run python research_cli.py edit my_strategy --add-feature MovingAverage --feature-params period=50 type=EMA

# Add a second moving average (same feature ID is fine — two entries are created)
uv run python research_cli.py edit my_strategy --add-feature MovingAverage --feature-params period=200 type=SMA

# Set hyperparameters
uv run python research_cli.py edit my_strategy --set-hparam oversold=30.0 overbought=70.0 stop_loss=0.05

# Set optimisation bounds for a param
uv run python research_cli.py edit my_strategy --set-bound oversold=20.0,40.0 overbought=60.0,80.0

# Remove a feature by ID (removes all entries with that ID)
uv run python research_cli.py edit my_strategy --remove-feature MovingAverage

# Wipe all features and start over
uv run python research_cli.py edit my_strategy --clear-features
```

Every `edit` call automatically re-syncs `context.py`.

### 5. Read context.py to get exact attribute names

This is the critical step before writing `generate_signals()`. The attribute names on `ctx.features` are derived from the feature ID and its params — they are not always obvious.

```bash
uv run python research_cli.py show-context my_strategy
```

Example output excerpt:

```python
class FeaturesContext:
    RSI_14: str = 'RSI_14'
    MovingAverage_50_EMA: str = 'MovingAverage_50_EMA'
    MovingAverage_200_SMA: str = 'MovingAverage_200_SMA'
```

Use these attribute names directly in `generate_signals()`:

```python
rsi   = df[ctx.features.RSI_14]
fast  = df[ctx.features.MovingAverage_50_EMA]
slow  = df[ctx.features.MovingAverage_200_SMA]
```

### 6. Inspect the full config before writing model code

```bash
uv run python research_cli.py inspect my_strategy
```

Shows features, hyperparameters, bounds, and the path to `model.py` in one view.

### 7. Write model.py

Read the existing scaffold first so you don't overwrite any work:

```bash
uv run python research_cli.py show-model my_strategy
```

The file is at `strategies/my_strategy/model.py`. Edit it directly.  
The class must subclass `engine.core.controller.SignalModel` and implement `generate_signals()`:

```python
import pandas as pd
from context import Context
from engine.core.controller import SignalModel


class MyStrategyModel(SignalModel):

    def generate_signals(self, df: pd.DataFrame, ctx: Context, artifacts: dict = None) -> pd.Series:
        # Access feature columns via ctx.features (attribute names from show-context)
        rsi  = df[ctx.features.RSI_14]
        fast = df[ctx.features.MovingAverage_50_EMA]
        slow = df[ctx.features.MovingAverage_200_SMA]

        # Access hyperparameters via ctx.params
        oversold  = ctx.params.oversold
        overbought = ctx.params.overbought

        # Return a Series in [-1.0, 1.0]: positive = long, negative = short, 0 = flat
        signals = pd.Series(0.0, index=df.index)
        signals[rsi < oversold]   =  1.0
        signals[rsi > overbought] = -1.0
        return signals
```

**Cross-sectional strategies** — if the manifest has `"cross_sectional": true`, both
`train` and `generate_signals` receive the full universe at once:

```python
class MyCrossSectionalModel(SignalModel):

    def train(self, data: Dict[str, pd.DataFrame], ctx: Context, params: dict) -> dict:
        return {}  # universe-level pre-computation goes here

    def generate_signals(self, data: Dict[str, pd.DataFrame], ctx: Context,
                         params: dict, artifacts: dict) -> Dict[str, pd.Series]:
        # data = {ticker: warmup-purged DataFrame with all features}
        # compute cross-sectional z-scores, rankings, etc.
        # return one signal Series per ticker, values in [-1.0, 1.0]
        ...
```

See `ENGINE_ARCHITECTURE.md §4.3` for the full contract.

### 8. Validate model.py before running

```bash
uv run python research_cli.py validate my_strategy
```

Checks: file exists → imports cleanly → concrete `SignalModel` subclass found → `generate_signals` is present → class can be instantiated. Catches syntax errors, missing imports, and wrong class structure without touching any data.

### 9. Backtest

```bash
uv run python research_cli.py backtest my_strategy --tickers AAPL --interval 1d --start 2020-01-01 --end 2024-01-01

# or with a named universe
uv run python research_cli.py backtest my_strategy --universe MEGA_CAP_TECH --interval 1d
```

Progress streams live. Output includes: Total Return, CAGR, Sharpe, Sortino, Calmar, Max Drawdown, Win Rate, Trade Count, Profit Factor, and a 5-trade log.

Add `--debug` to see the full engine traceback if the backtest errors:

```bash
uv run python research_cli.py backtest my_strategy --tickers AAPL --debug
```

### 10. Train (hyperparameter optimisation)

```bash
uv run python research_cli.py train my_strategy --tickers AAPL --interval 1d --start 2018-01-01 --end 2024-01-01

# multi-ticker ML training with a universe
uv run python research_cli.py train my_strategy --universe DOW_30 --interval 1d
```

Runs Optuna with Combinatorial Purged Cross-Validation (CPCV). Trial-by-trial log output streams live via the engine logger. The optimised params are reported at the end.

---

## Command Reference

### `list`

```bash
uv run python research_cli.py list [--json]
```

Lists all strategy directories under `strategies/` that contain a `manifest.json`.

---

### `features`

```bash
uv run python research_cli.py features [--json]
```

Dumps the full feature registry grouped by category. Each entry shows:
- **ID** — the string to use in `--add-feature` and in manifest `features[].id`
- **Description** — one-line summary
- **Defaults** — default param values (`normalize` lists show the first option)

---

### `inspect <strategy>`

```bash
uv run python research_cli.py inspect <strategy> [--json]
```

Shows the full manifest: features with all params, hyperparameters, parameter bounds, optional training config, and the path to `model.py`.

---

### `init <strategy>`

```bash
uv run python research_cli.py init <strategy>
```

`<strategy>` must be a valid Python identifier (letters, digits, underscores). Creates the workspace under `strategies/<strategy>/` with default RSI + BollingerBands features.

---

### `edit <strategy>`

```bash
uv run python research_cli.py edit <strategy> [options]
```

All flags are optional and can be combined in one call. Applied in this order: `--clear-features`, `--remove-feature`, `--add-feature`, `--set-hparam`, `--delete-hparam`, `--set-bound`.

| Flag | Description |
|---|---|
| `--add-feature ID` | Add a feature entry (validate ID against registry) |
| `--feature-params k=v ...` | Params for the feature being added |
| `--remove-feature ID [ID ...]` | Remove all entries with the given ID |
| `--clear-features` | Remove all features before other edits |
| `--set-hparam k=v [k=v ...]` | Add or overwrite hyperparameter(s) |
| `--delete-hparam key [key ...]` | Remove a hyperparameter |
| `--set-bound key=lo,hi [...]` | Set optimisation bound(s) |

---

### `show-context <strategy>`

```bash
uv run python research_cli.py show-context <strategy>
```

Prints `context.py` with line numbers. **Read this before writing `generate_signals()`** — the `FeaturesContext` class lists the exact attribute names that map to DataFrame column names.

---

### `show-model <strategy>`

```bash
uv run python research_cli.py show-model <strategy>
```

Prints `model.py` with line numbers. Read before editing to preserve existing logic.

---

### `data-info`

```bash
uv run python research_cli.py data-info [--ticker AAPL] [--interval 1d]
```

Queries the local SQLite cache (`data/stocks.db`) and reports bar count plus first/last date for every (ticker, interval) combination. Filter with `--ticker` and/or `--interval`.

---

### `validate <strategy>`

```bash
uv run python research_cli.py validate <strategy>
```

Five-step check:
1. `model.py` exists  
2. File imports without errors  
3. A concrete `SignalModel` subclass is found (same discovery logic as the backtester)  
4. `generate_signals` method is present  
5. Class can be instantiated  

Exits 0 on pass, 1 on any hard failure.

---

### `backtest <strategy>`

```bash
uv run python research_cli.py backtest <strategy> \
    --tickers AAPL,MSFT \
    --interval 1d \
    --start 2020-01-01 \
    --end 2024-01-01 \
    --capital 10000 \
    [--debug]

# or use a named universe
uv run python research_cli.py backtest <strategy> --universe MEGA_CAP_TECH --interval 1d
```

| Flag | Default | Description |
|---|---|---|
| `--tickers` | — | Comma-separated symbols (mutually exclusive with `--universe`) |
| `--universe` | — | Named universe, e.g. `DOW_30` (mutually exclusive with `--tickers`) |
| `--interval` | `1d` | Bar size: `1d`, `1h`, `4h`, `15m`, `1w` |
| `--start` | 1 year ago | Start date `YYYY-MM-DD` |
| `--end` | today | End date `YYYY-MM-DD` |
| `--capital` | `10000` | Starting capital in dollars |
| `--debug` | off | Print full engine traceback on error |

---

### `portfolio <strategy>`

```bash
uv run python research_cli.py portfolio <strategy> \
    --tickers AAPL,MSFT,NVDA,AMZN \
    --interval 1d \
    --start 2020-01-01 \
    --end 2024-01-01 \
    --capital 100000 \
    --max-positions 10 \
    [--no-short] \
    [--rebalance] \
    [--debug]

# or use a named universe
uv run python research_cli.py portfolio <strategy> --universe SECTOR_ETFS --interval 1d
```

Runs the full multi-asset portfolio simulation (T+1 execution, 2% risk rule, signal-strength-weighted sizing) and prints a five-section tearsheet:

1. **Portfolio Summary** — starting capital, final value, net P&L, and all scalar metrics
2. **Equity Curve (ASCII)** — 10-row ASCII sketch of portfolio value over time
3. **Per-Ticker Contribution** — sorted bar chart showing each ticker's net P&L contribution
4. **Exit Reason Breakdown** — trade count and total P&L grouped by exit type (STOP / SIGNAL / FLIP / EVICTED / END_OF_DATA)
5. **Trade Log** — last N trades with full entry/exit detail (use `--trades 0` for all)

| Flag | Default | Description |
|---|---|---|
| `--tickers` | — | Comma-separated symbols (mutually exclusive with `--universe`) |
| `--universe` | — | Named universe, e.g. `SECTOR_ETFS` (mutually exclusive with `--tickers`) |
| `--interval` | `1d` | Bar size: `1d`, `1h`, `4h`, `15m`, `1w` |
| `--start` | 1 year ago | Start date `YYYY-MM-DD` |
| `--end` | today | End date `YYYY-MM-DD` |
| `--capital` | `100000` | Starting capital in dollars |
| `--max-positions` | `10` | Max concurrent open positions |
| `--risk-pct` | `0.02` | Portfolio fraction risked per trade (2% rule) |
| `--stop-pct` | `0.05` | Fixed stop-loss as fraction of entry price |
| `--max-pos-pct` | `0.20` | Signal-scaled max position size as fraction of portfolio |
| `--entry-threshold` | `0.10` | Minimum \|signal\| required to open a position |
| `--eviction-margin` | `0.15` | New signal must exceed weakest current by this to evict it |
| `--friction` | `0.001` | One-way transaction cost fraction (0.1%) |
| `--rebalance` | off | Enable active resizing when signal shifts by `--rebalance-delta` |
| `--rebalance-delta` | `0.10` | Minimum \|Δsignal\| to trigger a resize (requires `--rebalance`) |
| `--no-short` | off | Long-only mode — disables short selling |
| `--trades` | `20` | Number of trades to show in the log (`0` = all) |
| `--debug` | off | Print full engine traceback on error |

---

### `train <strategy>`

```bash
uv run python research_cli.py train <strategy> \
    --tickers AAPL \
    --interval 1d \
    --start 2018-01-01 \
    --end 2024-01-01 \
    [--debug]

# multi-ticker ML training with a named universe
uv run python research_cli.py train <strategy> --universe DOW_30 --interval 1d
```

| Flag | Default | Description |
|---|---|---|
| `--tickers` | — | Comma-separated symbols (mutually exclusive with `--universe`) |
| `--universe` | — | Named universe, e.g. `DOW_30` (mutually exclusive with `--tickers`) |
| `--interval` | `1d` | Bar interval |
| `--start` | 1 year ago | Start date `YYYY-MM-DD` |
| `--end` | today | End date `YYYY-MM-DD` |
| `--debug` | off | Print full engine traceback on error |

Only parameters listed in `parameter_bounds` in the manifest are optimised. Hyperparameters with no bound entry are treated as fixed.

---

### `signal <strategy>`

```bash
uv run python research_cli.py signal <strategy> --tickers AAPL,MSFT

# or use a named universe
uv run python research_cli.py signal <strategy> --universe MEGA_CAP_TECH
```

| Flag | Default | Description |
|---|---|---|
| `--tickers` | — | Comma-separated symbols (mutually exclusive with `--universe`) |
| `--universe` | — | Named universe (mutually exclusive with `--tickers`) |

Fetches a warm-up window of recent data (size determined by interval), runs the full feature pipeline, and returns the final signal value per ticker. Output is a table with numeric signal, direction (LONG / SHORT / FLAT), and a visual strength bar.

---

### `trial-counts`

```bash
# View trial counts for all strategies
uv run python research_cli.py trial-counts

# Manually backfill counts for a strategy
uv run python research_cli.py trial-counts --set my_strategy --backtest-count 12 --train-count 3
```

| Flag | Default | Description |
|---|---|---|
| `--set` | — | Strategy name to update |
| `--backtest-count` | `0` | Backtest run count to record |
| `--train-count` | `0` | Training run count to record |

Shows how many times each strategy has been backtested and trained. These counts feed the Deflated Sharpe Ratio calculation — the DSR benchmark grows logarithmically with trial count to penalise overfitting through repeated testing.

---

### `sensitivity <strategy>`

```bash
uv run python research_cli.py sensitivity <strategy> \
    --tickers AAPL,MSFT,NVDA \
    --interval 1d \
    [--params stop_loss,min_consol_days] \
    [--steps 7]

# or use a named universe
uv run python research_cli.py sensitivity <strategy> --universe MEGA_CAP_TECH --interval 1d
```

| Flag | Default | Description |
|---|---|---|
| `--tickers` | — | Comma-separated tickers (mutually exclusive with `--universe`) |
| `--universe` | — | Named universe (mutually exclusive with `--tickers`) |
| `--interval` | `1d` | Bar interval |
| `--start` | 5 years ago | Start date `YYYY-MM-DD` |
| `--end` | today | End date `YYYY-MM-DD` |
| `--params` | top-3 by range width | Comma-separated parameter names to sweep |
| `--steps` | `7` | Number of steps per parameter |

Sweeps the top-3 bound parameters (or `--params`) across their `parameter_bounds` range in `--steps` evenly-spaced steps, holding all other params at their manifest defaults. Prints an ASCII Sharpe-vs-parameter table per param. Smooth plateau = robust signal; spiky non-monotonic curve = fitted noise. Does **not** increment the trial counter.

---

### `signal-stability <strategy>`

```bash
uv run python research_cli.py signal-stability <strategy> \
    --tickers AAPL,MSFT,NVDA \
    --interval 1d

# or use a named universe
uv run python research_cli.py signal-stability <strategy> --universe MEGA_CAP_TECH
```

| Flag | Default | Description |
|---|---|---|
| `--tickers` | — | Comma-separated tickers (mutually exclusive with `--universe`) |
| `--universe` | — | Named universe (mutually exclusive with `--tickers`) |
| `--interval` | `1d` | Bar interval |

Generates signals over the full history (default 2010–present) and three five-year subsets (2010–2015, 2015–2020, 2020–2025). Reports the Pearson correlation between the full-period signals and each subset. Correlation < 0.7 indicates the strategy is fitting period-specific noise rather than a persistent edge.

---

### `diagnose <strategy>`

```bash
uv run python research_cli.py diagnose <strategy> \
    --tickers AAPL,MSFT,NVDA \
    --interval 1d \
    [--start 2019-01-01] \
    [--sensitivity]

# or use a named universe
uv run python research_cli.py diagnose <strategy> --universe MEGA_CAP_TECH --interval 1d
```

| Flag | Default | Description |
|---|---|---|
| `--tickers` | — | Comma-separated tickers (mutually exclusive with `--universe`) |
| `--universe` | — | Named universe (mutually exclusive with `--tickers`) |
| `--interval` | `1d` | Bar interval |
| `--start` | 5 years ago | Start date `YYYY-MM-DD` |
| `--end` | today | End date `YYYY-MM-DD` |
| `--sensitivity` | off | Include parameter sensitivity sweep (adds ~21 backtests) |

One-page Phase 0 report combining:

1. **Core Metrics** — Sharpe, DSR, Total Return, CAGR, Max Drawdown, Sortino, trade count per ticker
2. **Fold Distribution** — OOS Sharpe distribution from the last training run's `diagnostics.json` (run `train` first to populate)
3. **Signal Stability** — Pearson correlation across three five-year subsets
4. **Verdict** — `KEEP` / `INVESTIGATE` / `RETIRE`

Verdict logic (Bailey & López de Prado): strategy is flagged `RETIRE` when 2 or more of these criteria fail:
- DSR > 0.5
- Positive OOS folds > 70%
- Signal stability correlation > 0.7

---

### `ic <strategy>`

```bash
uv run python research_cli.py ic <strategy> \
    --tickers AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,JPM,BAC,XOM \
    --interval 1d \
    --start 2015-01-01 \
    [--horizons 1 5 10 20 60] \
    [--save]

# a named universe is strongly recommended for cross-sectional IC
uv run python research_cli.py ic <strategy> --universe DOW_30 --interval 1d --start 2015-01-01
```

| Flag | Default | Description |
|---|---|---|
| `--tickers` | — | Comma-separated tickers, 20+ recommended (mutually exclusive with `--universe`) |
| `--universe` | — | Named universe — `DOW_30` or `TOP_200` recommended (mutually exclusive with `--tickers`) |
| `--interval` | `1d` | Bar interval |
| `--start` | 5 years ago | Start date `YYYY-MM-DD` |
| `--end` | today | End date `YYYY-MM-DD` |
| `--horizons` | `1 5 10 20 60` | Holding horizons in days |
| `--save` | off | Write report to `strategies/<name>/ic_report.txt` |

Computes cross-sectional Spearman rank IC at each horizon. At each date, ranks `(signal, forward_return)` pairs across all tickers and computes correlation. Reports mean IC, IC-IR, quintile spread, turnover, half-life, and period consistency.

**Gate thresholds:** mean IC > 0.02 and IC-IR > 0.3 at the 5-day horizon. Strategies failing the gate are `DISCARD` — do not proceed to `ic-surface` or strategy construction.

---

### `ic-surface <strategy>`

```bash
uv run python research_cli.py ic-surface <strategy> \
    --tickers AAPL,MSFT,GOOGL,... \
    --interval 1d \
    --start 2015-01-01 \
    --macro '^VIX:yf:5' 'T10Y2Y:fred:3' \
    [--primary-horizon 5] \
    [--min-obs 20] \
    [--force] \
    [--save]

# with a named universe
uv run python research_cli.py ic-surface <strategy> \
    --universe DOW_30 --interval 1d --start 2015-01-01 \
    --macro '^VIX:yf:5' 'T10Y2Y:fred:3'
```

| Flag | Default | Description |
|---|---|---|
| `--tickers` | — | Comma-separated tickers (mutually exclusive with `--universe`) |
| `--universe` | — | Named universe (mutually exclusive with `--tickers`) |
| `--interval` | `1d` | Bar interval |
| `--start` | 5 years ago | Start date `YYYY-MM-DD` |
| `--end` | today | End date `YYYY-MM-DD` |
| `--macro` | required | One or more macro dimensions: `SERIES_ID:SOURCE[:BINS]`. SOURCE is `yf` or `fred`. BINS defaults to 4. |
| `--primary-horizon` | `5` | Horizon (days) used for the conditional surface |
| `--min-obs` | `20` | Minimum observations per regime bin |
| `--force` | off | Compute surface even if the unconditional IC gate fails |
| `--save` | off | Write reports to `strategies/<name>/ic_report.txt` and `ic_surface_<dims>.txt` |

Runs the unconditional IC pass first (same as `ic`) and enforces the gate — surface is skipped unless the gate passes or `--force` is set. Then bins each macro dimension into quantile regimes and computes IC per regime cell (cross-product for multi-dimensional surfaces).

**Macro spec format:** `SERIES_ID:SOURCE[:BINS]`

```
^VIX:yf:5          CBOE VIX via yfinance, 5 quantile bins
T10Y2Y:fred:3      10Y-2Y yield spread from FRED, 3 bins
^TNX:yf            10Y Treasury yield, default 4 bins
BAMLH0A0HYM2:fred  High Yield OAS, default 4 bins
```

**Diagnosis output:** `flat` (IC uniform across regimes — no regime layer needed), `structured` (best-bin IC > unconditional + 0.02 — add regime weighting), or `noisy` (range exists but no coherent lift).

---

## Named Universes (`--universe`)

Every command that accepts `--tickers` also accepts `--universe NAME` as a mutually exclusive alternative. The name is expanded to a static ticker list before the command runs.

```bash
# Instead of a manual comma-separated list:
uv run python research_cli.py backtest my_strategy --tickers AAPL,MSFT,NVDA --interval 1d

# Use a named universe:
uv run python research_cli.py backtest my_strategy --universe MEGA_CAP_TECH --interval 1d
```

| Universe name | Size | Description |
|---|---|---|
| `MEGA_CAP_TECH` | 7 | AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA |
| `DOW_30` | 30 | Dow Jones Industrial Average components |
| `SECTOR_ETFS` | 11 | S&P sector ETFs (XLK, XLF, XLV, …) |
| `LIQUID_ETFS` | 10 | SPY, QQQ, IWM, TLT, GLD, … |
| `SEMICONDUCTORS` | 15 | NVDA, AMD, INTC, AVGO, QCOM, … |
| `FINANCIALS_LARGE` | 15 | JPM, BAC, WFC, GS, MS, … |
| `HEALTHCARE_PHARMA` | 15 | JNJ, UNH, PFE, ABBV, MRK, … |
| `CONSUMER_STAPLES` | 15 | PG, KO, PEP, COST, WMT, … |
| `ENERGY_MAJORS` | 15 | XOM, CVX, COP, SLB, EOG, … |
| `REITS` | 15 | PLD, AMT, EQIX, PSA, SPG, … |
| `VOLATILITY_PRODUCTS` | 4 | VXX, UVXY, SVXY, VIXY |
| `FIXED_INCOME_ETFS` | 15 | TLT, IEF, LQD, HYG, BND, … |
| `INTERNATIONAL_ETFS` | 15 | EFA, EEM, FXI, EWJ, EWZ, … |
| `COMMODITIES` | 10 | GLD, SLV, USO, DBA, DBC, … |
| `GROWTH_TECH` | 15 | SNOW, CRWD, DDOG, ZS, NET, … |
| `DIVIDEND_ARISTOCRATS` | 15 | JNJ, PG, KO, PEP, MMM, … |
| `SMALL_CAP_GROWTH` | 5 | IWO, VBK, SLYG, IJT, FYC |
| `LEVERAGED_ETFS` | 10 | TQQQ, SQQQ, SPXL, SOXL, … |
| `TOP_200` | 200 | Broad large/mega-cap cross-section |

**Note:** Universe lists are frozen snapshots of current index members — they have survivorship bias for historical backtests. Use with caution for pre-2015 analysis.

**Recommended universes by use case:**

| Use case | Recommended universe |
|---|---|
| Quick sanity-check | `MEGA_CAP_TECH` |
| Cross-sectional IC (`ic`, `ic-surface`) | `DOW_30` or `TOP_200` |
| Multi-ticker ML training | `DOW_30` or sector-specific |
| Portfolio backtest | `SECTOR_ETFS` or `LIQUID_ETFS` |

---

## Notes

- Strategies live in `strategies/` at the repo root (not `engine/strategies/`).
- `context.py` is auto-generated — never edit it manually. Run any `edit` command to regenerate it.
- `model.py` is only generated once (on `init`). All subsequent edits are manual.
- The `--json` flag on `list`, `features`, and `inspect` produces clean JSON for programmatic use.
- Paths and engine access mirror `gui/config.py` exactly (`WORKSPACE_DIR = ./strategies`, `DB_PATH = data/stocks.db`).
- `--universe` and `--tickers` are mutually exclusive on every command that accepts them. Passing both is an error.
