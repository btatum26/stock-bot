# Feature Output Types

Every feature in the engine produces data that falls into one of 7 shapes.
These types describe **data structure only** — no colors, widths, or rendering info.
The GUI interprets these shapes and decides how to draw them.

---

## 1. `line` — A time series

The most common. A `pd.Series` indexed by datetime with one value per bar.
This is the default — if a feature doesn't declare an output type, it's a line.

**Used by:** Moving averages (SMA/EMA/WMA/HMA/VWMA/DEMA/TEMA), RSI, MACD signal/main,
Stochastic %K/%D, ADX, CCI, ROC, Williams %R, OBV, ATR, Chaikin Money Flow,
Accumulation/Distribution, VWAP, linear regression, Bollinger mid/upper/lower,
Keltner mid/upper/lower, Ichimoku Tenkan/Kijun/Chikou/Senkou A/Senkou B, SuperTrend

---

## 2. `level` — Horizontal price thresholds

A list of `{value, label, strength?}`. Can be static (RSI's 30/70) or computed
(Fibonacci retracements, pivot points, support/resistance).

```python
# Static — known before compute
[{"value": 70, "label": "Overbought"}, {"value": 30, "label": "Oversold"}]

# Computed — returned in FeatureResult
[{"value": 423.50, "label": "S1", "strength": 0.85},
 {"value": 445.20, "label": "R1", "strength": 0.72}]
```

**Used by:** Support/resistance, Fibonacci retracements, pivot points (Standard,
Camarilla, Woodie), volume-weighted levels, RSI/Stochastic/CCI overbought-oversold
boundaries, VWAP standard deviation bands

---

## 3. `band` — A filled region between two boundaries

A pair of `pd.Series` (upper, lower) that form a shaded region. The feature declares
which two of its line outputs form the band.

```python
{"upper": "upper", "lower": "lower"}  # references output suffixes
```

**Used by:** Bollinger Bands, Keltner Channels, Donchian Channels, Ichimoku Cloud
(Senkou A/B), linear regression channel, VWAP bands, standard deviation envelopes,
Darvas boxes

---

## 4. `histogram` — Positive/negative bar values

A `pd.Series` where each value is a signed magnitude. The type declaration tells the
GUI to render bars rather than a continuous line.

```python
# Just a regular pd.Series — the output type distinguishes it from a line
{"MACD_HIST": pd.Series([0.5, -0.3, 0.1, ...])}
```

**Used by:** MACD histogram, Awesome Oscillator, volume delta (buying vs selling
pressure), Chaikin Oscillator, Elder Force Index, accumulation/distribution delta

---

## 5. `marker` — Discrete events at specific bars

A sparse `pd.Series` (mostly NaN) where non-NaN values indicate an event. The value
can encode direction/magnitude (e.g. 1.0 = bullish, -1.0 = bearish).

```python
# 1.0 = bullish, -1.0 = bearish, NaN = nothing
{"doji": pd.Series([NaN, NaN, 1.0, NaN, -1.0, ...])}
```

**Used by:** Candlestick patterns (doji, engulfing, hammer, shooting star, morning
star, etc.), buy/sell signals, divergence points, Parabolic SAR dots, pivot reversals,
Elliott Wave labels, breakout/breakdown alerts

---

## 6. `zone` — Rectangular price-time regions

A list of `{start, end, upper, lower, label?}` where start/end are timestamps and
upper/lower are prices. Represents a 2D box on the chart.

```python
[{"start": "2024-01-15", "end": "2024-01-22",
  "upper": 155.30, "lower": 152.80, "label": "Demand Zone"},
 {"start": "2024-02-01", "end": "2024-02-05",
  "upper": 161.00, "lower": 159.50, "label": "Supply Zone"}]
```

**Used by:** Supply/demand zones, fair value gaps (FVG), order blocks, consolidation
ranges, earnings gaps, opening range, initial balance (for intraday), session ranges
(Asian/London/NY), volume gaps

---

## 7. `heatmap` — 2D intensity grid

A dict with `{price_grid, time_index, intensity}` — values at the intersection of
price levels and time bins. This is the most complex output type.

```python
{"price_grid": [100, 101, 102, ...],     # y-axis price buckets
 "time_index": [...timestamps...],        # x-axis time bins
 "intensity":  [[0.1, 0.3, ...], ...]}    # 2D array of values
```

**Used by:** Volume profile (per-bar or session), KDE (kernel density estimation),
order flow / footprint charts, delta-at-price, time-at-price, market profile (TPO)

---

## Type-to-data shape mapping

```
output_type  | data shape                          | default pane
-------------|-------------------------------------|-------------
line         | pd.Series                           | overlay / new
level        | list of {value, label, strength?}   | overlay / new
band         | references two line outputs         | overlay / new
histogram    | pd.Series (signed)                  | new
marker       | pd.Series (sparse, NaN-gapped)      | overlay
zone         | list of {start, end, upper, lower}  | overlay
heatmap      | {price_grid, time_index, intensity} | overlay
```

The `pane` hint is structural, not graphical — `"overlay"` means the data is in the
price coordinate space, `"new"` means the data has its own y-axis range. A distributed
worker needs this to organize data correctly without knowing anything about rendering.

---

## Feature inventory by output type

| Feature | line | level | band | histogram | marker | zone | heatmap |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Moving Averages (SMA/EMA/WMA/HMA/VWMA/DEMA/TEMA) | x | | | | | | |
| Bollinger Bands | x | | x | | | | |
| Keltner Channels | x | | x | | | | |
| Donchian Channels | x | | x | | | | |
| Ichimoku Cloud | x | | x | | | | |
| SuperTrend | x | | | | | | |
| VWAP / Anchored VWAP | x | | | | | | |
| Linear Regression Channel | x | | x | | | | |
| Parabolic SAR | | | | | x | | |
| RSI | x | x | | | | | |
| MACD | x | | | x | | | |
| Stochastic | x | x | | | | | |
| ADX (+DI/-DI) | x | x | | | | | |
| CCI | x | x | | | | | |
| ROC | x | x | | | | | |
| Williams %R | x | x | | | | | |
| MFI | x | x | | | | | |
| Awesome Oscillator | | | | x | | | |
| ATR | x | | | | | | |
| OBV | x | | | | | | |
| Volume Z-Score | x | x | | | | | |
| Chaikin Money Flow | x | x | | | | | |
| Support/Resistance | | x | | | | | |
| Fibonacci Retracements | | x | | | | | |
| Pivot Points | | x | | | | | |
| Candlestick Patterns | | | | | x | | |
| RSI Divergence | x | | | | x | | |
| Supply/Demand Zones | | | | | | x | |
| Fair Value Gaps | | | | | | x | |
| Order Blocks | | | | | | x | |
| Volume Profile | | | | | | | x |
| KDE | | | | | | | x |
| Market Profile (TPO) | | | | | | | x |

Every feature maps to a combination of these 7 types. No special cases needed — even
the most complex indicators (Ichimoku, Elliott Wave, volume profile) decompose into
these primitives.
