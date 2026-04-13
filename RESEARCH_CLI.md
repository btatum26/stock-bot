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

### 8. Validate model.py before running

```bash
uv run python research_cli.py validate my_strategy
```

Checks: file exists → imports cleanly → concrete `SignalModel` subclass found → `generate_signals` is present → class can be instantiated. Catches syntax errors, missing imports, and wrong class structure without touching any data.

### 9. Backtest

```bash
uv run python research_cli.py backtest my_strategy --tickers AAPL --interval 1d --start 2020-01-01 --end 2024-01-01
```

Progress streams live. Output includes: Total Return, CAGR, Sharpe, Sortino, Calmar, Max Drawdown, Win Rate, Trade Count, Profit Factor, and a 5-trade log.

Add `--debug` to see the full engine traceback if the backtest errors:

```bash
uv run python research_cli.py backtest my_strategy --tickers AAPL --debug
```

### 10. Train (hyperparameter optimisation)

```bash
uv run python research_cli.py train my_strategy --tickers AAPL --interval 1d --start 2018-01-01 --end 2024-01-01
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
```

| Flag | Default | Description |
|---|---|---|
| `--tickers` | required | Comma-separated symbols |
| `--interval` | `1d` | Bar size: `1d`, `1h`, `4h`, `15m`, `1w` |
| `--start` | 1 year ago | Start date `YYYY-MM-DD` |
| `--end` | today | End date `YYYY-MM-DD` |
| `--capital` | `10000` | Starting capital in dollars |
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
```

Only parameters listed in `parameter_bounds` in the manifest are optimised. Hyperparameters with no bound entry are treated as fixed.

---

### `signal <strategy>`

```bash
uv run python research_cli.py signal <strategy> --tickers AAPL,MSFT
```

Fetches a warm-up window of recent data (size determined by interval), runs the full feature pipeline, and returns the final signal value per ticker. Output is a table with numeric signal, direction (LONG / SHORT / FLAT), and a visual strength bar.

---

## Notes

- Strategies live in `strategies/` at the repo root (not `engine/strategies/`).
- `context.py` is auto-generated — never edit it manually. Run any `edit` command to regenerate it.
- `model.py` is only generated once (on `init`). All subsequent edits are manual.
- The `--json` flag on `list`, `features`, and `inspect` produces clean JSON for programmatic use.
- Paths and engine access mirror `gui/config.py` exactly (`WORKSPACE_DIR = ./strategies`, `DB_PATH = data/stocks.db`).
