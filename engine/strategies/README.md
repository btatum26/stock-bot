# Strategy Development Guide

This is the main entry point for building trading strategies on the engine.
If you're new to the system, read this document end to end before writing your
first strategy.

---

## Table of Contents

1. [What is a Strategy?](#what-is-a-strategy)
2. [The Three-File Contract](#the-three-file-contract)
3. [Quick Start: Your First Strategy](#quick-start-your-first-strategy)
4. [manifest.json — Declaring What Your Strategy Needs](#manifestjson--declaring-what-your-strategy-needs)
5. [context.py — The Auto-Generated Bridge](#contextpy--the-auto-generated-bridge)
6. [model.py — Where Your Logic Lives](#modelpy--where-your-logic-lives)
7. [Signals: The Output Contract](#signals-the-output-contract)
8. [The Execution Pipeline](#the-execution-pipeline)
9. [Available Features](#available-features)
10. [Training and Optimization](#training-and-optimization)
11. [Tearsheet Metrics](#tearsheet-metrics)
12. [ML Strategies](#ml-strategies)
13. [Common Patterns](#common-patterns)
14. [Troubleshooting](#troubleshooting)

---

## What is a Strategy?

A strategy is a self-contained folder that tells the engine:

1. **What data it needs** — which technical indicators (features) to compute
2. **What knobs it has** — hyperparameters that control its behavior
3. **What it does with that data** — signal generation logic that outputs
   buy/sell/hold decisions as numbers between -1.0 and 1.0

The engine handles everything else: fetching market data, computing indicators,
managing warmup periods, validating your output, and calculating performance
metrics.

---

## The Three-File Contract

Every strategy is a folder containing exactly three files:

```
strategies/my_strategy/
├── manifest.json    ← You edit this — declares features, params, and bounds
├── context.py       ← Auto-generated — never edit by hand
└── model.py         ← You write this — your trading logic
```

| File | Who writes it | When it changes |
|------|--------------|-----------------|
| `manifest.json` | You | Whenever you add/remove features or change params |
| `context.py` | The engine (via SYNC) | Regenerated after every manifest change |
| `model.py` | You | Whenever you change your trading logic |

The relationship between them:

```
manifest.json ──(SYNC)──→ context.py ──(imported by)──→ model.py
     │                        │                            │
     │  "I need RSI with      │  class FeaturesContext:    │  rsi = df[ctx.features.RSI_14]
     │   period 14"           │      RSI_14 = 'RSI_14'     │  if rsi < threshold: buy
     │                        │                            │
     │  "threshold = 30"      │  class ParamsContext:      │  threshold = ctx.params.threshold
     │                        │      threshold = 30.0      │
```

---

## Quick Start: Your First Strategy

### 1. Create the scaffold

From the GUI, click the **+ New** button in the control bar and type a name.

Or from the CLI:
```bash
cd engine
uv run python main.py INIT --strategy my_first_strategy
```

This creates the three files with sensible defaults (RSI + Bollinger Bands).

### 2. Edit manifest.json

Open `strategies/my_first_strategy/manifest.json` and define what you want:

```json
{
    "features": [
        {"id": "RSI", "params": {"period": 14}},
        {"id": "SMA", "params": {"period": 50}}
    ],
    "hyperparameters": {
        "rsi_oversold": 30.0,
        "rsi_overbought": 70.0
    },
    "parameter_bounds": {
        "rsi_oversold": [20.0, 40.0],
        "rsi_overbought": [60.0, 80.0]
    }
}
```

### 3. Sync to regenerate context.py

```bash
uv run python main.py SYNC --strategy my_first_strategy
```

This reads your manifest and generates a typed `context.py` with attribute names
that match the exact DataFrame column names the engine will produce.

### 4. Write your logic in model.py

```python
import pandas as pd
from context import Context
from engine.core.controller import SignalModel

class MyStrategy(SignalModel):
    def train(self, df, context, params):
        # Rule-based strategies don't need training
        return {}

    def generate_signals(self, df, context, params, artifacts):
        rsi = df[context.features.RSI_14]
        sma = df[context.features.SMA_50]
        close = df['Close'] if 'Close' in df.columns else df['close']

        oversold = context.params.rsi_oversold
        overbought = context.params.rsi_overbought

        signals = pd.Series(0.0, index=df.index)
        signals[(rsi < oversold) & (close > sma)] = 1.0    # Buy
        signals[(rsi > overbought) & (close < sma)] = -1.0  # Sell
        return signals
```

### 5. Backtest

```bash
uv run python main.py BACKTEST --strategy my_first_strategy --ticker AAPL --interval 1d
```

The engine will:
1. Fetch AAPL daily data
2. Compute RSI(14) and SMA(50)
3. Purge warmup NaN rows
4. Call your `generate_signals()` with the enriched DataFrame
5. Validate and compress your signals to [-1.0, 1.0]
6. Calculate Sharpe, CAGR, max drawdown, win rate, and print a report

---

## manifest.json — Declaring What Your Strategy Needs

This is the single source of truth for your strategy's configuration.

### Required fields

#### `features` — What indicators to compute

An array of objects, each with:

| Key | Type | Description |
|-----|------|-------------|
| `id` | `string` | The feature's registry ID (e.g. `"RSI"`, `"BollingerBands"`) |
| `params` | `object` | Parameters passed to the feature's `compute()` method |

```json
"features": [
    {"id": "RSI", "params": {"period": 14}},
    {"id": "BollingerBands", "params": {"period": 20, "std_dev": 2.0}},
    {"id": "MACD", "params": {"fast_period": 12, "slow_period": 26, "signal_period": 9}}
]
```

Each feature ID must exist in the engine's feature registry.
See [Available Features](#available-features) for the full list.

#### `hyperparameters` — Tunable knobs for your logic

A flat dictionary of parameter names → default values. These are the values your
strategy reads from `context.params.*` and that the optimizer sweeps during training.

```json
"hyperparameters": {
    "stop_loss": 0.05,
    "take_profit": 0.10,
    "lookback_window": 20
}
```

**Rules:**
- Names must be valid Python identifiers (letters, digits, underscores)
- Names cannot be Python keywords (`if`, `def`, `class`, etc.)
- Types are inferred automatically: `int`, `float`, `str`, `bool`, `list`, `dict`

#### `parameter_bounds` — Ranges for the optimizer

Defines the search space for hyperparameter optimization. Each entry is a
parameter name → `[min, max]` array.

```json
"parameter_bounds": {
    "stop_loss": [0.01, 0.10],
    "lookback_window": [10, 50]
}
```

Parameters not listed here are held constant at their `hyperparameters` default
during optimization. You only need bounds for parameters you want the optimizer
to tune.

### Optional fields

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `is_ml` | `bool` | `false` | Enable ML pipeline (model serialization, scaling) |
| `compression_mode` | `string` | `"clip"` | How raw signals are squashed to [-1, 1] |

### Compression modes

Your `generate_signals()` can return values outside [-1, 1]. The engine
compresses them before backtesting:

| Mode | Formula | Best for |
|------|---------|----------|
| `"clip"` | Hard cap at ±1.0 | Rule-based strategies that already output ±1 |
| `"tanh"` | `tanh(signal)` | Regression models that output unbounded values |
| `"probability"` | `signal * 2 - 1` | Binary classifiers that output [0.0, 1.0] |

---

## context.py — The Auto-Generated Bridge

**Never edit this file.** It is regenerated every time you run SYNC.

When you run SYNC, the engine reads your `manifest.json` and generates a
`context.py` containing three frozen dataclasses:

### FeaturesContext — Column name mapping

For each feature in your manifest, the engine computes a deterministic
DataFrame column name and exposes it as a typed attribute:

```python
@dataclass(frozen=True)
class FeaturesContext:
    RSI_14: str = 'RSI_14'
    SMA_50: str = 'SMA_50'
    BollingerBands_20_2_0_upper: str = 'BollingerBands_20_2.0_upper'
    BollingerBands_20_2_0_lower: str = 'BollingerBands_20_2.0_lower'
```

This gives you autocomplete and catches typos at development time instead of
at runtime.

### ParamsContext — Hyperparameter defaults

```python
@dataclass(frozen=True)
class ParamsContext:
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
```

During optimization, the engine overrides these values with trial candidates.
Your model code reads `context.params.rsi_oversold` and automatically gets
either the default or the optimizer's trial value — no code changes needed.

### Context — Master container

```python
@dataclass(frozen=True)
class Context:
    features: FeaturesContext = field(default_factory=FeaturesContext)
    params: ParamsContext = field(default_factory=ParamsContext)
```

### Column naming algorithm

The column name that ends up in the DataFrame is deterministic and follows
these rules:

1. Non-core params are filtered out (`color`, `normalize`, `overbought`, `oversold`)
2. Single core param (`period` or `window`): `{ID}_{value}` → `RSI_14`
3. No core params: `{ID}` → `VWAP`
4. Multiple core params: `{ID}_{v1}_{v2}` (sorted by key) → `MACD_12_26`
5. Multi-output features append `_{suffix}` → `BollingerBands_20_2.0_upper`
6. Normalized features prepend `Norm_` → `Norm_RSI_14`

You never need to memorize these rules — SYNC generates the exact names for you
in `context.py`.

---

## model.py — Where Your Logic Lives

Your model file must contain a class that extends `SignalModel` from the engine.

### The two methods you implement

```python
from engine.core.controller import SignalModel

class MyStrategy(SignalModel):

    def train(self, df, context, params):
        """
        Called once before signal generation.

        Args:
            df:       DataFrame with OHLCV + all computed feature columns
            context:  Your strategy's Context object (features + params)
            params:   Dict of hyperparameters (same as context.params, but as a raw dict)

        Returns:
            dict: Artifacts to pass to generate_signals().
                  For rule-based strategies, return {}.
                  For ML strategies, return trained model, scaler, etc.
        """
        return {}

    def generate_signals(self, df, context, params, artifacts):
        """
        The core of your strategy. Called after train().

        Args:
            df:        DataFrame with OHLCV + feature columns. READ-ONLY.
            context:   Context object — use context.features.X and context.params.Y
            params:    Raw hyperparameters dict
            artifacts: Whatever train() returned

        Returns:
            pd.Series: Signal values. Must have the same index as df.
                       -1.0 = maximum short conviction
                        0.0 = no conviction (flat / no trade)
                        1.0 = maximum long conviction
        """
        signals = pd.Series(0.0, index=df.index)
        # your logic here
        return signals
```

### Rules

1. **Never modify `df`** — The engine checks that you didn't add columns to it.
   If you do, it raises a `FeatureError` and kills your backtest.

2. **Access features through context** — Use `df[context.features.RSI_14]`,
   not `df['RSI_14']`. The context gives you autocomplete and catches typos.

3. **Access params through context** — Use `context.params.threshold`, not
   `params['threshold']`. During optimization, the engine swaps in trial values
   through the same context interface.

4. **Handle both column casings** — OHLCV columns may be `'Close'` or `'close'`
   depending on where the data came from:
   ```python
   close = df['Close'] if 'Close' in df.columns else df['close']
   ```

5. **Return the right shape** — Your Series must have the same length and index
   as the input DataFrame. The engine will try to fix mismatches, but it's
   better to get it right.

6. **Class name doesn't matter** — The engine finds your class by checking which
   class in the file is a subclass of `SignalModel`. Name it whatever you want.

---

## Signals: The Output Contract

Your `generate_signals()` returns a pandas Series of conviction weights:

```
 1.0  ████████████  Maximum long conviction
 0.5  ██████        Moderate long conviction
 0.0  ─────────────  No conviction (flat / cash)
-0.5  ██████        Moderate short conviction
-1.0  ████████████  Maximum short conviction
```

### What happens to your signals

1. **Type coercion** — Lists and numpy arrays are converted to pandas Series
2. **Index alignment** — If your Series has a wrong index but correct length,
   the engine forces the target index onto it
3. **NaN/Inf cleanup** — `NaN` and `±Inf` are replaced with `0.0` (no conviction)
4. **Compression** — Values outside [-1, 1] are compressed using the mode
   specified in your manifest (default: `clip`)
5. **Naming** — The Series is renamed to `"conviction_signal"`

### The T+1 execution model

Signals are backtested with realistic execution timing:

- Signal generated at bar **T** (using close price)
- Order executes at bar **T+1** (using open price)
- Return measured from **T+1 open** to **T+2 open**

This prevents look-ahead bias. You can't trade on information you wouldn't have
had at the time.

---

## The Execution Pipeline

Here's what happens when you run a backtest, step by step:

```
manifest.json
     │
     ▼
┌─────────────────────┐
│ 1. Load manifest     │  Read features config, hyperparams, compression mode
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 2. Fetch market data │  OHLCV from Yahoo Finance → cached in SQLite
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 3. Compute features  │  FeatureOrchestrator runs each feature's compute()
│                      │  Shared FeatureCache prevents redundant computation
│                      │  Memory safety: verifies df was not mutated
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 4. Warmup purge      │  Removes first l_max rows (indicator NaN warmup)
│                      │  e.g., SMA(50) needs 50 bars before producing values
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 5. NaN audit         │  Checks for unexpected NaNs after purge
│                      │  Logs warnings if found (indicates bad data/features)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 6. Load model.py     │  Dynamic import of your SignalModel subclass
│                      │  Fresh instance per run (no state leakage)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 7. Train             │  model.train(df, context, params) → artifacts
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 8. Generate signals  │  model.generate_signals(df, ctx, params, artifacts)
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 9. Signal validation │  SignalValidator.validate_and_compress()
│                      │  Coerce type, align index, kill NaN/Inf, compress
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ 10. Tearsheet        │  Tearsheet.calculate_metrics(df, signals)
│                      │  Total return, CAGR, Sharpe, drawdown, win rate...
└─────────────────────┘
```

---

## Available Features

Every feature listed here can be used in your `manifest.json` by its ID.

### Oscillators (Momentum)

| ID | Name | Key Params |
|----|------|------------|
| `RSI` | Relative Strength Index | `period` |
| `MACD` | MACD | `fast_period`, `slow_period`, `signal_period` |
| `Stochastic` | Stochastic Oscillator | `k_period`, `d_period` |
| `CCI` | Commodity Channel Index | `period` |
| `ROC` | Rate of Change | `period` |
| `RSI_Divergence_Features` | RSI Divergence | `rsi_period`, `fractal_n` |

### Trend

| ID | Name | Key Params |
|----|------|------------|
| `SMA` | Simple Moving Average | `period` |
| `EMA` | Exponential Moving Average | `period` |
| `MovingAverage` | Moving Average (SMA/EMA/WMA) | `period`, `type` |
| `ADX` | Average Directional Index | `period` |
| `Ichimoku` | Ichimoku Cloud | `conversion_period`, `base_period`, `lagging_span2_period` |
| `Supertrend` | Supertrend | `period`, `multiplier` |
| `LinReg` | Linear Regression Channel | `lookback`, `std_dev` |

### Volatility

| ID | Name | Key Params |
|----|------|------------|
| `ATR` | Average True Range | `period` |
| `BollingerBands` | Bollinger Bands | `period`, `std_dev` |
| `KeltnerChannels` | Keltner Channels | `ema_period`, `atr_period`, `multiplier` |

### Volume

| ID | Name | Key Params |
|----|------|------------|
| `OBV` | On-Balance Volume | *(none)* |
| `VWAP` | VWAP | *(none)* |
| `AnchoredVWAP` | Anchored VWAP | `anchor_bars_back` |
| `Volume` | Raw Volume | *(none)* |
| `VolumeProfile` | Volume Profile (VPVR) | `bins`, `lookback` |
| `VolumeZScore` | Volume Z-Score | `period` |

### Price Levels

| ID | Name | Key Params |
|----|------|------------|
| `SupportResistance` | Support & Resistance | `method`, `window`, `clustering_pct` |
| `Fibonacci` | Fibonacci Retracement | `lookback` |
| `KDE` | Kernel Density Heatmap | `bandwidth`, `source`, `resolution` |

### Patterns

| ID | Name | Key Params |
|----|------|------------|
| `CandlePatterns` | Candlestick Patterns | `doji_threshold`, `hammer_ratio` |

All features also accept a `normalize` param (`"none"`, `"pct_distance"`,
`"price_ratio"`, `"z_score"`) for ML-friendly output.

---

## Training and Optimization

When you run TRAIN mode, the engine finds the best hyperparameters for your
strategy automatically.

### How it works

1. The optimizer reads `parameter_bounds` from your manifest
2. It picks a search algorithm:
   - **Grid Search** if total permutations ≤ 5,000 (exhaustive, parallelized)
   - **Optuna Bayesian** if > 5,000 (smart sampling, 100 trials)
3. For each candidate set of hyperparameters:
   - Compute features
   - Call your `train()` and `generate_signals()`
   - Calculate the **Sharpe Ratio** as the fitness score
4. The best parameters are returned along with a final confirmatory backtest

### Running optimization

```bash
cd engine
uv run python main.py TRAIN --strategy my_strategy --ticker AAPL --interval 1d
```

### Tips

- Start with wide bounds, then narrow them once you see which ranges work
- The optimizer maximizes Sharpe Ratio — strategies with low volatility score well
- Keep `parameter_bounds` lean. Every extra parameter multiplies the search space
- Parameters not in `parameter_bounds` are held at their manifest defaults
- Grid search runs on all CPU cores via Joblib

---

## Tearsheet Metrics

After every backtest, the engine calculates these performance metrics:

| Metric | What It Means |
|--------|---------------|
| **Total Return (%)** | Cumulative strategy return over the period |
| **CAGR (%)** | Compound Annual Growth Rate — annualized return |
| **Max Drawdown (%)** | Worst peak-to-trough decline (always negative) |
| **Win Rate (%)** | Percentage of trade entries that were profitable |
| **Profit Factor** | Total gains / total losses (>1.0 is profitable) |
| **Sharpe Ratio** | Risk-adjusted return (>1.0 is good, >2.0 is excellent) |
| **Total Trades** | Number of position changes (signal direction flips) |

### Friction model

A transaction cost of **0.1% (10 bps)** is applied on every position change.
This simulates slippage and commissions. Strategies that trade frequently pay
more friction.

---

## ML Strategies

For strategies that use machine learning models:

### 1. Set `is_ml: true` in manifest.json

```json
{
    "is_ml": true,
    "compression_mode": "probability",
    ...
}
```

### 2. Use `train()` to fit your model

```python
def train(self, df, context, params):
    from sklearn.ensemble import RandomForestClassifier

    X = df[[context.features.RSI_14, context.features.SMA_50]]
    y = (df['Close'].shift(-1) > df['Close']).astype(int)  # next-day direction

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X.dropna(), y[X.dropna().index])

    return {"model": model}
```

### 3. Use artifacts in `generate_signals()`

```python
def generate_signals(self, df, context, params, artifacts):
    model = artifacts["model"]
    X = df[[context.features.RSI_14, context.features.SMA_50]].fillna(0)

    probabilities = model.predict_proba(X)[:, 1]  # P(up)
    return pd.Series(probabilities, index=df.index)
```

### 4. Set compression mode to `"probability"`

With `compression_mode: "probability"`, the engine maps your [0.0, 1.0]
classifier output to [-1.0, 1.0]:

- 0.0 (100% bearish) → -1.0
- 0.5 (neutral) → 0.0
- 1.0 (100% bullish) → +1.0

---

## Common Patterns

### Combining multiple features

```python
def generate_signals(self, df, context, params, artifacts):
    rsi = df[context.features.RSI_14]
    bb_upper = df[context.features.BollingerBands_20_2_0_upper]
    bb_lower = df[context.features.BollingerBands_20_2_0_lower]
    close = df['Close'] if 'Close' in df.columns else df['close']

    signals = pd.Series(0.0, index=df.index)
    # Buy when RSI oversold AND price at lower Bollinger Band
    signals[(rsi < 30) & (close <= bb_lower)] = 1.0
    # Sell when RSI overbought AND price at upper Bollinger Band
    signals[(rsi > 70) & (close >= bb_upper)] = -1.0
    return signals
```

### Graduated conviction (not just ±1)

```python
def generate_signals(self, df, context, params, artifacts):
    rsi = df[context.features.RSI_14]
    signals = pd.Series(0.0, index=df.index)

    # Stronger conviction at more extreme RSI levels
    signals[rsi < 20] = 1.0     # Strong buy
    signals[(rsi >= 20) & (rsi < 30)] = 0.5  # Moderate buy
    signals[(rsi > 70) & (rsi <= 80)] = -0.5 # Moderate sell
    signals[rsi > 80] = -1.0    # Strong sell
    return signals
```

### Using train() for rule-based pre-computation

Even without ML, `train()` is useful for computing things once:

```python
def train(self, df, context, params):
    close = df['Close'] if 'Close' in df.columns else df['close']
    return {
        "median_volume": df['Volume'].median(),
        "avg_range": (df['High'] - df['Low']).mean(),
    }

def generate_signals(self, df, context, params, artifacts):
    volume_filter = df['Volume'] > artifacts["median_volume"]
    # ... use volume_filter in your logic
```

---

## Troubleshooting

### "Missing or invalid manifest"

Your `manifest.json` is missing, has a syntax error, or is not valid JSON.
Open it and check for trailing commas, missing brackets, or unquoted strings.

### "Feature 'X' not found in registry"

The feature ID in your manifest doesn't match any registered feature. IDs are
case-sensitive. Check the [Available Features](#available-features) table for
the exact spelling. Common mistakes: `"rsi"` (should be `"RSI"`),
`"bollinger"` (should be `"BollingerBands"`).

### "No valid SignalModel subclass found"

Your `model.py` doesn't contain a class that inherits from `SignalModel`.
Make sure you have:
```python
from engine.core.controller import SignalModel
class MyStrategy(SignalModel):
    ...
```

### "attempted to mutate the input DataFrame in place"

You wrote `df['my_col'] = ...` inside `compute()` or `generate_signals()`.
Never assign columns to `df`. Create a new Series and return it.

### "Signal length does not match input length"

Your `generate_signals()` returned a Series with a different number of rows
than the input DataFrame. Make sure you initialize with `pd.Series(0.0, index=df.index)`
and use vectorized operations that preserve the index.

### NaN warnings after warmup purge

Your features are producing NaN values even after the warmup period was removed.
This usually means:
- Your feature has a bug (dividing by zero, accessing out-of-bounds)
- Your data has gaps (missing trading days)
- Your feature's lookback is longer than what the engine detected

### Training takes too long

- Reduce the number of parameters in `parameter_bounds`
- Narrow the bounds (fewer permutations)
- Use fewer features (less computation per trial)
- The optimizer uses all CPU cores — closing other applications helps

### "Strategy initialization failed"

Your `model.py` has a Python syntax error, a broken import, or an exception
in your class body. Run it standalone to check:
```bash
cd strategies/my_strategy
python -c "import model"
```
