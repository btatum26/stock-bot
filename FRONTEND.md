# Frontend Architecture

The GUI is a PyQt6 desktop app backed by `engine/bridge.py`.

## Entry Point

```text
stock_bot.py -> gui/gui.py (ChartWindow)
```

## Directory Layout

```text
gui/
  config.py
  colors.py
  gui.py
  chart/
    data_manager.py
    manifest_manager.py
    feature_manager.py
  gui_components/
    backtest_panel.py
    portfolio_panel.py
    training_panel.py
    plots.py
    controls.py
  features/
    engine_adapter.py
    loader.py
```

## Responsibilities

- `ChartWindow` owns the main tabs and wires managers/panels.
- `ChartDataManager` fetches OHLCV via `ModelEngine`.
- `ManifestManager` reads/writes `manifest.json` and `gui_prefs.json`.
- `FeatureManager` owns active feature widgets and chart overlays.
- `gui/features/engine_adapter.py` adapts engine `Feature` outputs into GUI visual outputs.
- Backtest, training, and portfolio panels call the same `ModelEngine` facade used by `CLI.py`.

## Data Flow

```text
ControlBar
  -> ChartDataManager.load()
  -> ModelEngine.get_historical_data()
  -> FeatureManager.update_all()
  -> engine Feature.compute()
  -> GUI overlays
```

Feature changes in the chart tab update `strategies/<name>/manifest.json` and
regenerate `context.py`.
