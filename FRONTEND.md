# Frontend Architecture

The GUI layer is a PyQt6 desktop app that connects to the research engine for data, backtesting, and ML inference.

## Entry Point

```
stock_bot.py → src/gui.py (ChartWindow)
```

## Directory Layout

```
gui/
├── config.py                 # Paths, timings, layout constants (edit here, not in code)
├── colors.py                 # Centralized color palette (all hex values live here)
├── gui.py                    # ChartWindow + SignalPreviewWorker
├── chart/                    # ChartWindow sub-managers (no Qt UI construction)
│   ├── data_manager.py       # ChartDataManager — fetch OHLCV, pre-compute label arrays
│   ├── manifest_manager.py   # ManifestManager — read/write manifest.json + gui_prefs.json
│   └── feature_manager.py    # FeatureManager — add/remove/update features, overlay routing
├── gui_components/           # Reusable UI panels and rendering primitives
│   ├── axes.py               # Date/time axis tick formatter
│   ├── backtest_panel.py     # Full backtest UI (strategy config, leaderboard, drill-down)
│   ├── candles.py            # Candlestick bar renderer (pyqtgraph)
│   ├── controls.py           # Top control bar (ticker, interval, period)
│   ├── feature_panel.py      # Feature selector + per-feature parameter widgets
│   ├── models_panel.py       # Container for signals + training panels
│   ├── plots.py              # Overlay system and UnifiedPlot
│   ├── score_panel.py        # Scoring function selector + appearance controls
│   ├── signals_panel.py      # Signal model browser (set active, rename, delete)
│   ├── signals_tab.py        # [Incomplete] Strategy signal preview tab
│   ├── styling.py            # Dark theme palette + global stylesheet
│   ├── training_panel.py     # ML training trigger + settings
│   └── volume.py             # Volume bar renderer
└── features/
    ├── base.py               # FeatureOutput types (Line/Level/Marker/Heatmap) + Feature ABC
    ├── engine_adapter.py     # Adapts engine Feature objects → GUI visual outputs
    ├── feature_set.py        # [Unused] FeatureSet container
    ├── loader.py             # Thin wrapper around engine_adapter
    └── signals.py            # [Unused] Legacy hardcoded MA/RSI signal detection
```

## Component Responsibilities

### `gui.py` — ChartWindow
The top-level window. Owns both tabs (Chart and Backtest) and wires up the three manager classes. UI construction, crosshair, and signal preview live here; everything else is delegated.

| Manager | Owned by | Responsibility |
|---|---|---|
| `ChartDataManager` | `ChartWindow._data_manager` | Fetch OHLCV, compute label arrays |
| `ManifestManager` | `ChartWindow._manifest_manager` | Read/write manifest.json + gui_prefs.json |
| `FeatureManager` | `ChartWindow._feature_manager` | Add/remove/render features; build manifest payload |

`FeatureManager` emits `sync_needed` whenever a parameter changes or a feature is removed. `ChartWindow` connects to this signal and calls `ManifestManager.sync()` (guarded by `_suppress_sync` during batch strategy loads).

`write_context_py()` now lives on `ModelEngine` in `engine/bridge.py` and is called by both `ManifestManager` and `BacktestPanel` via the engine reference they already hold.

### `gui_components/plots.py` — Overlay System
Central rendering layer. All chart content is an overlay on a `UnifiedPlot`.

| Class | Z-value | Purpose |
|---|---|---|
| `CandleOverlay` | 1000 | OHLC candlestick bars |
| `VolumeOverlay` | -10 | Volume bars (separate ViewBox) |
| `LineOverlay` | 10 | Indicator lines |
| `LevelOverlay` | 100 | Horizontal support/resistance zones |
| `ScoreOverlay` | 50 | Signal strength heatmap |

All overlays inherit `BaseOverlay` and expose `add_to_plot`, `remove_from_plot`, `update`, and `get_y_range`.

### `gui_components/backtest_panel.py` — BacktestPanel
Self-contained three-panel backtest UI:
- Right sidebar: strategy selector, hyperparameter editor, batch config, run/cancel
- Left top: sortable leaderboard with color-coded metrics
- Left bottom: drill-down tabs (portfolio chart, trade log, signal chart)

Runs `engine.batch_backtest()` via a background `EngineWorker` (QThread).

### `src/features/engine_adapter.py` — Engine → GUI Bridge
Wraps engine `Feature` objects for GUI consumption. Translates engine `OutputSchema` output types into `FeatureOutput` subtypes:

| Engine OutputType | GUI Output |
|---|---|
| `LINE` | `LineOutput` |
| `LEVEL` | `LevelOutput` |
| `MARKER` | `MarkerOutput` |
| `HISTOGRAM` | `LineOutput` (workaround) |
| `HEATMAP` | Not yet implemented |
| `ZONE` | Not yet implemented |

## Data Flow

```
ControlBar (ticker/interval)
       ↓ load_chart()
  ModelEngine.fetch_ohlcv()
       ↓ pd.DataFrame
  ChartWindow loads overlays
       ↓ for each active feature
  AdaptedFeature.compute()
       ↓ FeatureResult
  UnifiedPlot.add/update overlay
```

Signal preview adds a second path:
```
  SignalPreviewWorker (QThread)
       ↓ engine.backtester
       ↓ pd.Series in [-1, 1]
  Scatter markers overlaid on main plot
```

## Known Issues

### Remaining items

**`signals_tab.py` is incomplete** — References a `strategy.Strategy` class that doesn't exist. The tab is rendered but non-functional.

**`OutputType.HISTOGRAM` rendered as a line** — Noted as a workaround in `engine_adapter.py`. Will look incorrect for features that return histogram data.

**Unused code** — `SimpleCandleItem`, `SignalEngine`, `FeatureSet`, and `signals_tab.py` are not integrated. They add noise and implied capabilities that don't exist.

## Remaining work

- Implement missing overlay types (`ZONE`, `HEATMAP`, proper `HISTOGRAM` bars).
- Remove or complete `signals_tab.py`, `SignalEngine`, and `FeatureSet`.
