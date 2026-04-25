# Feature Engineering System

This document explains how the feature pipeline works end-to-end and how to build new features.

---

## Table of Contents

1. [What is a Feature?](#what-is-a-feature)
2. [Architecture Overview](#architecture-overview)
3. [The 7 Output Types](#the-7-output-types)
4. [Building a New Feature](#building-a-new-feature)
5. [Feature Base Class Reference](#feature-base-class-reference)
6. [The Compute Pipeline](#the-compute-pipeline)
7. [Feature Caching](#feature-caching)
8. [Output Schema and the GUI](#output-schema-and-the-gui)
9. [Using a Feature in a Strategy](#using-a-feature-in-a-strategy)
10. [Complete Example: Building a CCI Feature](#complete-example-building-a-cci-feature)
11. [Rules and Constraints](#rules-and-constraints)

---

## What is a Feature?

A feature is a self-contained technical indicator that:
- Takes raw OHLCV market data in
- Produces one or more computed time series out
- Declares its **output schema** so consumers (GUI, strategies, ML) know what to expect
- Self-registers into a global registry so the system can find it by name

Features are the building blocks that strategies consume. A strategy's `manifest.json`
lists which features it needs, and the engine computes them automatically before
handing the enriched DataFrame to the strategy's signal logic.

---

## Architecture Overview

```
manifest.json                        FeatureOrchestrator
┌─────────────────┐                  ┌──────────────────────────┐
│ features:       │                  │ 1. Validate config       │
│  - id: "RSI"    │───────────────→  │ 2. Instantiate features  │
│    params:      │                  │ 3. Call compute() each   │
│      window: 14 │                  │ 4. Concat into DataFrame │
│  - id: "MACD"   │                  │ 5. Return (df, l_max)    │
│    params: ...  │                  └──────────┬───────────────┘
└─────────────────┘                             │
                                                ▼
                                   ┌──────────────────────────┐
                                   │ df with new columns:     │
                                   │   Open, High, Low, Close │
                                   │   RSI_14                 │
                                   │   MACD_12_26_SIGNAL      │
                                   │   MACD_12_26_HIST        │
                                   │   ...                    │
                                   └──────────────────────────┘
                                                │
                                   ┌────────────┴────────────┐
                                   │                         │
                                   ▼                         ▼
                           Strategy model.py            GUI renderer
                           reads columns via            reads output_schema
                           ctx.features.RSI_14          to pick draw method
```

### Pipeline Steps

1. **Strategy declares features** in `manifest.json`
2. **`compute_all_features(df, config)`** is called by the backtester
3. **`FeatureOrchestrator`** validates the config against the registry
4. For each feature in the config:
   - Instantiate the feature class from `FEATURE_REGISTRY`
   - Call `feature.compute(df, params, cache)` with a shared `FeatureCache`
   - Memory safety check: verify the input DataFrame was not mutated
   - Collect all output Series into a dict
5. **Concat** all computed Series onto the original DataFrame
6. **Return** `(enriched_df, l_max)` where `l_max` is the max lookback window needed for warmup
7. The backtester **purges** the first `l_max` rows (NaN warmup period)
8. The purged DataFrame is passed to the strategy's `generate_signals()` method

---

## The 7 Output Types

Every feature output falls into one of these structural data shapes. These describe
**data structure only** — the GUI decides colors, widths, and styles independently.

See [output_types.md](output_types.md) for the full reference with examples.

| Type | Data Shape | Typical Pane | Example |
|---|---|---|---|
| `LINE` | `pd.Series` — one value per bar | overlay or new | Moving average, RSI, ATR |
| `LEVEL` | `[{value, label, strength?}, ...]` | overlay or new | Support/resistance, Fibonacci, RSI 30/70 |
| `BAND` | References two LINE outputs | overlay | Bollinger fill, Keltner fill, Ichimoku cloud |
| `HISTOGRAM` | `pd.Series` — signed magnitude | new | MACD histogram, volume delta |
| `MARKER` | `pd.Series` — sparse, NaN-gapped | overlay | Candlestick patterns, divergence signals |
| `ZONE` | `[{start, end, upper, lower}, ...]` | overlay | Supply/demand zones, fair value gaps |
| `HEATMAP` | `{price_grid, time_index, intensity}` | overlay | Volume profile, KDE |

---

## Building a New Feature

### Step 1: Choose a category

Features are organized by category under `engine/core/features/`:

```
features/
├── momentum/       # RSI, MACD, Stochastic, ROC
├── trend/          # Moving averages, ADX
├── volatility/     # Bollinger Bands, ATR, Keltner Channels
├── volume/         # VWAP, OBV, Volume Z-Score
├── levels/         # Support/Resistance
├── macro/          # FRED series, VIX term structure
├── options/        # Options-chain implied signals (live only)
└── alternative/    # Insider flow (EDGAR), Google Trends
```

Place your file in the appropriate subdirectory. If you're making a new category,
create a new directory with an `__init__.py`.

### Step 2: Create the feature file

```python
# engine/core/features/momentum/cci.py

from typing import Dict, Any, List
import pandas as pd
from ..base import (
    Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature
)


@register_feature("CCI")
class CCI(Feature):
    """Commodity Channel Index — measures price deviation from its statistical mean."""

    @property
    def name(self) -> str:
        return "CCI"

    @property
    def description(self) -> str:
        return "Commodity Channel Index — measures deviation from statistical mean."

    @property
    def category(self) -> str:
        return "Oscillators (Momentum)"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"period": 20, "normalize": "none"}

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name=None, output_type=OutputType.LINE, pane=Pane.NEW),
            OutputSchema(name="overbought", output_type=OutputType.LEVEL, pane=Pane.NEW),
            OutputSchema(name="oversold", output_type=OutputType.LEVEL, pane=Pane.NEW),
        ]

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache=None) -> FeatureResult:
        period = int(params.get("period", 20))
        norm = params.get("normalize", "none")

        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: (x - x.mean()).abs().mean())
        cci = (tp - sma_tp) / (0.015 * mad)

        col = self.generate_column_name("CCI", params)
        final = self.normalize(df, cci, norm)

        return FeatureResult(
            data={col: final},
            levels=[
                {"value": 100, "label": "Overbought"},
                {"value": -100, "label": "Oversold"},
            ],
        )
```

That's it. The feature is now:
- Registered as `"CCI"` in the global registry
- Discoverable by any strategy that lists `{"id": "CCI"}` in its manifest
- Renderable by the GUI (reads output_schema, sees LINE + two LEVELs)

### Step 3: Verify it loads

Features are auto-discovered at import time. The `load_features()` function in
`features.py` walks all subdirectories under `engine/core/features/` and imports
every `.py` file, which triggers the `@register_feature` decorators.

You don't need to edit any import lists or `__init__.py` files. Just create the file
in the right directory and it's registered.

To verify:
```python
from engine.core.features.features import load_features
from engine.core.features.base import FEATURE_REGISTRY

load_features()
assert "CCI" in FEATURE_REGISTRY
```

---

## Feature Base Class Reference

Every feature extends `Feature` (ABC) from `engine/core/features/base.py`.

### Required (abstract) properties

| Property | Type | Description |
|---|---|---|
| `name` | `str` | Display name shown in the GUI and logs |
| `description` | `str` | One-line explanation of what the feature computes |
| `category` | `str` | Grouping label (e.g., `"Oscillators (Momentum)"`, `"Volatility"`) |

### Required (abstract) method

| Method | Signature | Description |
|---|---|---|
| `compute` | `(df, params, cache?) → FeatureResult` | The core math. See [Compute Pipeline](#the-compute-pipeline) |

### Optional overridable properties

| Property | Type | Default | Description |
|---|---|---|---|
| `parameters` | `Dict[str, Any]` | `{}` | Default parameter values. These are used when a strategy doesn't specify params |
| `parameter_options` | `Dict[str, Dict]` | `{}` | Metadata about parameters: allowed values, bounds, types. Used by the GUI for dropdowns/sliders |
| `output_schema` | `List[OutputSchema]` | `[OutputSchema(None, LINE, NEW)]` | Declares the structural outputs. See [Output Schema](#output-schema-and-the-gui) |
| `outputs` | `List[Optional[str]]` | *derived from output_schema* | Column suffixes. You don't need to override this — it's auto-derived from `output_schema` |

### Helper methods

| Method | Signature | Description |
|---|---|---|
| `generate_column_name` | `(feature_id, params, output_name?) → str` | Deterministic column naming. Call this to name your output Series. See [Column Naming](#column-naming) |
| `normalize` | `(df, series, method, window?) → pd.Series` | Normalizes raw data for ML. Methods: `"none"`, `"pct_distance"`, `"price_ratio"`, `"z_score"` |

---

## The Compute Pipeline

### Input: `compute(df, params, cache)`

| Argument | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Raw OHLCV market data. Columns: `Open`, `High`, `Low`, `Close`, `Volume`. **Read-only** — do not assign new columns to this DataFrame. |
| `params` | `Dict[str, Any]` | The parameters for this computation, merged from the feature's defaults and the strategy's manifest config. Common keys: `period`, `window`, `normalize`, `source`. |
| `cache` | `FeatureCache` (optional) | Shared cache for fetching pre-computed dependencies. Use this when your feature depends on another (e.g., Bollinger Bands depends on SMA). |

### Output: `FeatureResult`

```python
@dataclass
class FeatureResult:
    data:     Dict[str, pd.Series]       # Time-series columns (lines, histograms, markers)
    levels:   List[Dict[str, Any]]       # Horizontal thresholds [{value, label, strength?}]
    zones:    List[Dict[str, Any]]       # Rectangular regions [{start, end, upper, lower, label?}]
    heatmaps: Dict[str, Dict[str, Any]]  # 2D grids {name: {price_grid, time_index, intensity}}
```

All fields default to `None`. Populate only the ones your feature produces.

**The `data` dict is the critical field** — these are the Series that get concatenated
onto the DataFrame and become available to strategies via `ctx.features`. The keys
must be deterministic column names generated by `generate_column_name()`.

The `levels`, `zones`, and `heatmaps` fields carry structured data that doesn't fit
into a time-series column. The orchestrator passes them through to the GUI separately.

### Column Naming

Use `self.generate_column_name(feature_id, params, output_name)` to create column names.

The naming algorithm:
1. Filters out non-core params (`color`, `normalize`, `overbought`, `oversold`, `color_*`)
2. If only one core param (`period` or `window`): `"{ID}_{value}"` → `RSI_14`
3. If no core params: `"{ID}"` → `VWAP`
4. Otherwise: `"{ID}_{v1}_{v2}_{...}"` (sorted by key) → `MACD_12_26`
5. If `output_name` is set: append `"_{OUTPUT}"` → `MACD_12_26_SIGNAL`
6. If `normalize != "none"`: prepend `"Norm_"` → `Norm_RSI_14`

**This naming must be deterministic** because the workspace manager uses the exact same
logic to generate the `context.py` attribute names. If your column name doesn't match,
the strategy can't access the data.

---

## Feature Caching

The `FeatureCache` prevents redundant computation when multiple features share a
dependency. For example, both Bollinger Bands and Keltner Channels need a Simple
Moving Average. Without caching, the SMA would be computed twice.

### Using the cache as a consumer

```python
def compute(self, df, params, cache=None):
    period = int(params.get("period", 20))

    # This either returns a cached SMA or computes + caches one
    if cache:
        sma = cache.get_series("SMA", {"period": period}, df)
    else:
        close = df["Close"]
        sma = close.rolling(window=period).mean()
```

`cache.get_series(feature_id, params, df)`:
1. Checks if the feature with those exact params was already computed
2. If yes: returns the cached Series immediately
3. If no: looks up the feature in `FEATURE_REGISTRY`, calls `compute()`, caches all
   output Series, and returns the primary one

### Being a good cache citizen

- Always call `cache.get_series()` for dependencies rather than computing them inline
- Never mutate the DataFrame — the cache and orchestrator verify this with column-count
  checks before and after your `compute()` call
- Your outputs are automatically cached by column name after computation

---

## Output Schema and the GUI

The `output_schema` property tells consumers (primarily the GUI) what shape each
output takes and where it belongs structurally.

### OutputSchema fields

```python
@dataclass(frozen=True)
class OutputSchema:
    name:        Optional[str]   # Suffix label (None for single-output features)
    output_type: OutputType      # One of the 7 data shapes
    pane:        Pane            # OVERLAY (price space) or NEW (own y-axis)
    band_pair:   Optional[tuple] # For BAND: (upper_suffix, lower_suffix)
    y_range:     Optional[tuple] # Fixed y-axis bounds, e.g., (0, 100) for RSI
```

### Pane: OVERLAY vs NEW

- **`Pane.OVERLAY`**: The data is in the same coordinate space as price. Moving
  averages, Bollinger Bands, support/resistance lines — these overlay the candlestick
  chart.
- **`Pane.NEW`**: The data has its own y-axis scale. RSI (0-100), MACD, ATR — these
  get their own sub-chart below the price chart.

### Examples

**Single line on a new pane (simplest case — RSI, CCI, ATR):**
```python
@property
def output_schema(self):
    return [
        OutputSchema(name=None, output_type=OutputType.LINE, pane=Pane.NEW, y_range=(0, 100)),
        OutputSchema(name="overbought", output_type=OutputType.LEVEL, pane=Pane.NEW),
        OutputSchema(name="oversold", output_type=OutputType.LEVEL, pane=Pane.NEW),
    ]
```

**Multiple lines with a band fill (Bollinger Bands, Keltner Channels):**
```python
@property
def output_schema(self):
    return [
        OutputSchema(name="upper", output_type=OutputType.LINE, pane=Pane.OVERLAY),
        OutputSchema(name="mid",   output_type=OutputType.LINE, pane=Pane.OVERLAY),
        OutputSchema(name="lower", output_type=OutputType.LINE, pane=Pane.OVERLAY),
        OutputSchema(name="width", output_type=OutputType.LINE, pane=Pane.NEW),
        OutputSchema(name="fill",  output_type=OutputType.BAND, pane=Pane.OVERLAY,
                     band_pair=("upper", "lower")),
    ]
```

**Line + histogram on same pane (MACD):**
```python
@property
def output_schema(self):
    return [
        OutputSchema(name=None,      output_type=OutputType.LINE, pane=Pane.NEW),
        OutputSchema(name="signal",  output_type=OutputType.LINE, pane=Pane.NEW),
        OutputSchema(name="hist",    output_type=OutputType.HISTOGRAM, pane=Pane.NEW),
    ]
```

**Lines + computed levels (Support/Resistance):**
```python
@property
def output_schema(self):
    return [
        OutputSchema(name="dist_to_support",    output_type=OutputType.LINE, pane=Pane.NEW),
        OutputSchema(name="dist_to_resistance", output_type=OutputType.LINE, pane=Pane.NEW),
        OutputSchema(name="last_support_level",    output_type=OutputType.LINE, pane=Pane.OVERLAY),
        OutputSchema(name="last_resistance_level", output_type=OutputType.LINE, pane=Pane.OVERLAY),
        OutputSchema(name="levels", output_type=OutputType.LEVEL, pane=Pane.OVERLAY),
    ]
```

### How the GUI reads this

The GUI does not need per-feature rendering code. For each feature:

1. Call `compute()` to get the `FeatureResult`
2. Read `output_schema` to get the list of `OutputSchema` entries
3. For each entry, dispatch to the appropriate renderer based on `output_type`:
   - `LINE` → draw a continuous line from the Series in `result.data`
   - `LEVEL` → draw horizontal lines from `result.levels`
   - `BAND` → shade the region between two LINE outputs
   - `HISTOGRAM` → draw signed bars from the Series
   - `MARKER` → draw scatter points where Series is non-NaN
   - `ZONE` → draw filled rectangles from `result.zones`
   - `HEATMAP` → draw a color-mapped grid from `result.heatmaps`
4. Colors, widths, and styles come from the GUI's own theme config — never from the feature

---

## Using a Feature in a Strategy

### 1. Add it to manifest.json

```json
{
    "features": [
        {
            "id": "CCI",
            "params": {"period": 20}
        }
    ],
    "hyperparameters": {
        "cci_threshold": 100
    },
    "parameter_bounds": {
        "cci_threshold": [50, 200]
    }
}
```

### 2. Run SYNC to regenerate context.py

```bash
uv run python main.py SYNC --strategy my_strategy
```

This generates a typed `context.py`:

```python
@dataclass(frozen=True)
class FeaturesContext:
    CCI_20: str = 'CCI_20'

@dataclass(frozen=True)
class ParamsContext:
    cci_threshold: float = 100

@dataclass(frozen=True)
class Context:
    features: FeaturesContext
    params: ParamsContext
```

### 3. Use it in model.py

```python
class SignalModel:
    def generate_signals(self, df, ctx, artifacts=None):
        cci = df[ctx.features.CCI_20]
        threshold = ctx.params.cci_threshold

        signals = pd.Series(0.0, index=df.index)
        signals[cci > threshold] = -1.0   # Overbought → short
        signals[cci < -threshold] = 1.0   # Oversold → long
        return signals
```

### 4. Backtest

```bash
uv run python main.py BACKTEST --strategy my_strategy --ticker AAPL --interval 1d
```

The engine will:
1. Fetch AAPL daily data
2. Compute all features listed in manifest.json (CCI in this case)
3. Purge warmup NaN rows
4. Call your `generate_signals()` with the enriched DataFrame
5. Validate signals are in [-1.0, 1.0]
6. Calculate tearsheet metrics (Sharpe, CAGR, max drawdown, etc.)

---

## Complete Example: Building a CCI Feature

Here's a full walkthrough from empty file to working backtest.

### The feature file

Create `engine/core/features/momentum/cci.py`:

```python
from typing import Dict, Any, List
import pandas as pd
from ..base import (
    Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature
)


@register_feature("CCI")
class CCI(Feature):
    @property
    def name(self) -> str:
        return "CCI"

    @property
    def description(self) -> str:
        return "Commodity Channel Index — measures deviation from statistical mean."

    @property
    def category(self) -> str:
        return "Oscillators (Momentum)"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "period": 20,
            "normalize": "none",
        }

    @property
    def parameter_options(self) -> Dict[str, Dict[str, Any]]:
        return {
            "period": {"min": 5, "max": 100, "step": 1},
        }

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name=None, output_type=OutputType.LINE, pane=Pane.NEW),
            OutputSchema(name="overbought", output_type=OutputType.LEVEL, pane=Pane.NEW),
            OutputSchema(name="oversold", output_type=OutputType.LEVEL, pane=Pane.NEW),
        ]

    def compute(
        self,
        df: pd.DataFrame,
        params: Dict[str, Any],
        cache=None,
    ) -> FeatureResult:
        period = int(params.get("period", 20))
        norm_method = params.get("normalize", "none")

        # Typical Price
        high = df["High"] if "High" in df.columns else df["high"]
        low = df["Low"] if "Low" in df.columns else df["low"]
        close = df["Close"] if "Close" in df.columns else df["close"]
        tp = (high + low + close) / 3

        # CCI = (TP - SMA(TP)) / (0.015 * Mean Absolute Deviation)
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(
            lambda x: (x - x.mean()).abs().mean(), raw=False
        )
        cci = (tp - sma_tp) / (0.015 * mad)

        # Name the output column deterministically
        col_name = self.generate_column_name("CCI", params)
        final_data = self.normalize(df, cci, norm_method)

        return FeatureResult(
            data={col_name: final_data},
            levels=[
                {"value": 100, "label": "Overbought"},
                {"value": -100, "label": "Oversold"},
            ],
        )
```

### Checklist

- [x] File placed in correct category directory (`momentum/`)
- [x] `@register_feature("CCI")` decorator with unique ID
- [x] All abstract properties implemented (`name`, `description`, `category`)
- [x] `parameters` returns sensible defaults
- [x] `output_schema` declares the data shapes
- [x] `compute()` uses `generate_column_name()` for the output key
- [x] `compute()` never mutates `df`
- [x] `compute()` returns a `FeatureResult`
- [x] Works with `normalize` param for ML compatibility

---

## External Data Features

The `macro/`, `options/`, and `alternative/` categories fetch data from external sources
rather than computing from the OHLCV DataFrame passed to `compute()`. They follow the
same interface contract as all other features, with a few additional patterns.

### Macro features (`macro/`)

**FRED series** — `NFCI`, `ANFCI`, `HYSpread`, `T10Y2Y`, `T10Y3M`, `VIXCLS`, `ICSA`, `DFF`

Each FRED feature fetches the series via `DataFetcher.fetch_macro_data()` using the
date range of the price df's index, then forward-fills onto that index (weekly releases
fill daily gaps). Each produces three output columns:

| Suffix | Description | Stationary? |
|---|---|---|
| `_LEVEL` | Raw FRED value | Non-stationary — auto-routed through FFD by trainer |
| `_ROC5` | 5-day percent change | Stationary |
| `_ZSCORE` | 252-day rolling z-score (min 63 obs) of the **level** | Stationary |

The `non_stationary_outputs()` method declares `_LEVEL` so the ML trainer automatically
routes it through fractional differentiation before scaling.

**FRED series quick reference:**

| Feature ID | FRED Series | Description |
|---|---|---|
| `NFCI` | NFCI | Chicago Fed National Financial Conditions Index (weekly, ffilled) |
| `ANFCI` | ANFCI | Adjusted NFCI — removes business-cycle variation |
| `HYSpread` | BAMLH0A0HYM2 | ICE BofA High-Yield OAS (leading credit stress indicator) |
| `T10Y2Y` | T10Y2Y | 10Y minus 2Y Treasury spread (yield curve, recession predictor) |
| `T10Y3M` | T10Y3M | 10Y minus 3M Treasury spread (St. Louis Fed recession model input) |
| `VIXCLS` | VIXCLS | VIX (FRED daily close, overlaps with yfinance ^VIX) |
| `ICSA` | ICSA | Initial jobless claims (weekly labor market stress) |
| `DFF` | DFF | Effective fed funds rate |

**Important column naming gotcha**: `HYSpread` uses `COLUMN_PREFIX = "HYSpread"` (not
`"BAMLH0A0HYM2"`). All other FRED features use their SERIES_ID as the column prefix.

**Important distinction**: `_ZSCORE` is the 252-day z-score of the **raw level**, not of
`_ROC5`. If you need the z-score of the 5-day momentum, compute it yourself in model.py
from `_ROC5` — the feature only gives you the level's z-score.

**VIX term structure** — `VIXTermStructure`

Fetches `^VIX` and `^VIX3M` from yfinance and computes `VIX / VIX3M`. Values below
0.90 indicate deep contango (calm); above 1.00 indicates backwardation (stress). Outputs
the raw ratio and its 252-day z-score.

| Output attr | Column name | Description |
|---|---|---|
| `VIXTERMSTRUCTURE` | `VIXTermStructure` | Raw VIX/VIX3M ratio |
| `VIXTERMSTRUCTURE_ZSCORE` | `VIXTermStructure_ZSCORE` | 252-day rolling z-score of ratio |

Example manifest entry (no parameters needed):

```json
{"id": "NFCI"}
{"id": "T10Y2Y"}
{"id": "VIXTermStructure"}
```

**Macro strategy pattern — using features without fetching data in model.py:**

The correct way to build a macro signal is to declare the features in `manifest.json`
and consume the pre-computed columns in `model.py`. Do not call FRED or yfinance directly
from `generate_signals()`.

```python
# manifest.json
{"id": "HYSpread", "params": {}}   # gives HYSPREAD_ROC5, HYSPREAD_LEVEL, HYSPREAD_ZSCORE
{"id": "NFCI", "params": {}}       # gives NFCI_LEVEL, NFCI_ROC5, NFCI_ZSCORE
{"id": "VIXTermStructure", "params": {}}  # gives VIXTERMSTRUCTURE, VIXTERMSTRUCTURE_ZSCORE

# model.py — all pandas, no external calls
def generate_signals(self, df, ctx, artifacts=None):
    hy_roc5 = df[ctx.features.HYSPREAD_ROC5]   # 5-day ROC of HY OAS
    nfci    = df[ctx.features.NFCI_LEVEL]       # raw NFCI index
    ratio   = df[ctx.features.VIXTERMSTRUCTURE] # VIX/VIX3M
```

See `strategies/macro_regime_rotation/model.py` for a complete example implementing
a three-component composite macro score.

### Options features (`options/`)

**`OptionsFlow`** — requires `"ticker"` in params.

Fetches the live options chain from yfinance and computes:

| Output | Description |
|---|---|
| `_PCR` | Put/call volume ratio (30-day expiry chain) |
| `_PCR_CHG5` | 5-day change in P/C ratio |
| `_IV_SKEW` | OTM put IV (≈25-delta) minus ATM IV |
| `_IV_TS` | Short-term ATM IV divided by medium-term ATM IV (~30d / ~90d) |
| `_VOL_UNUSUAL` | Total options volume divided by its 20-day rolling median |

**Important**: yfinance only exposes current options snapshots, not historical data.
All output columns return NaN for historical backtesting rows. A live value is stamped
at the last row only when the price df ends within 5 calendar days of today. These
features are useful for signal generation, not backtesting.

```json
{"id": "OptionsFlow", "params": {"ticker": "AAPL"}}
```

### Alternative features (`alternative/`)

**`InsiderFlow`** — requires `"ticker"` in params. Data cached in `data/insider.db`.

Fetches SEC EDGAR Form 4 filings for the ticker via the EDGAR submissions API,
filters to open-market purchases by officers and directors (excluding pure 10%-owner
filers), and caches results in a SQLite database. Subsequent runs are served from cache.

Outputs a rolling purchase count over `window` calendar days and a sparse cluster
marker where the count reaches `min_cluster` (default 3).

```json
{"id": "InsiderFlow", "params": {"ticker": "AAPL", "window": 14, "min_cluster": 3}}
```

**`GoogleTrends`** — requires `"ticker"` in params. Requires `uv add pytrends`.

Fetches weekly Google Search interest, forward-fills to the daily index, and outputs
the interest as a ratio to its rolling `median_window`-week median plus a z-score.
Rate-limited by Google (~1 req/sec); best suited for weekly-refreshed signal universes.

```json
{"id": "GoogleTrends", "params": {"ticker": "AAPL", "median_window": 8}}
```

### The `ticker` parameter pattern

Three features (`OptionsFlow`, `InsiderFlow`, `GoogleTrends`) require the ticker symbol
as an explicit `params` entry because `compute()` only receives the price DataFrame
(which carries no ticker metadata). This mirrors the `TickerComparison` feature's
`compare_ticker` param.

Note that including `ticker` in params affects the generated column name:

```
InsiderFlow_AAPL_14_3_PURCHASE_COUNT
```

This is expected behaviour — it makes multi-ticker comparisons in a single strategy
unambiguous.

---

## Rules and Constraints

### Memory Safety

The orchestrator enforces strict immutability on the input DataFrame:

```python
# ❌ NEVER do this — the orchestrator will raise FeatureError
df["my_column"] = computed_series

# ✅ Always return new Series in the FeatureResult
return FeatureResult(data={"my_column": computed_series})
```

Before and after calling your `compute()`, the orchestrator counts the DataFrame's
columns. If the count changes, your feature is killed with a `FeatureError`. This
prevents subtle bugs where one feature's mutation silently affects another.

### Signal Bounds

Features produce raw data — they can output any numerical range. The **strategy's
signal output** (from `generate_signals()`) is what gets bounded to `[-1.0, 1.0]` by
the `SignalValidator`. Features themselves are not bounded.

### Deterministic Column Names

Column names **must** be generated by `self.generate_column_name()`. The workspace
manager uses the same algorithm to generate `context.py`. If you hand-code a column
name, strategies won't be able to access it through the typed context.

### Normalization

Every feature should accept a `normalize` parameter (defaulting to `"none"`) and call
`self.normalize(df, series, method)` on its output. This allows strategies to request
normalized data for ML without changing the feature's core math.

Available methods:
- `"none"` — raw values (default)
- `"pct_distance"` — `(close - series) / series`
- `"price_ratio"` — `series / close`
- `"z_score"` — `(series - rolling_mean) / rolling_std`

### No Rendering Information

Features must never include colors, widths, line styles, or any visual config in their
output. The `output_schema` describes data shape and structural pane placement only.
The GUI owns all rendering decisions.

```python
# ❌ Don't do this
return {"color": "#ff0000", "width": 2, "data": series}

# ✅ Do this
return FeatureResult(data={col_name: series})
```
