# Stock Bot Pro

Quantitative finance workspace for market visualization, strategy development,
backtesting, training, and signal generation.

## Current Layout

- `stock_bot.py` starts the PyQt6 desktop GUI in `gui/`.
- `CLI.py` is the primary command-line interface for strategy work.
- `engine/` contains the model engine, FastAPI/RQ daemon, data broker, features,
  backtester, trainer, and tests.
- `strategies/<name>/` contains each strategy workspace:
  `manifest.json`, generated `context.py`, user-authored `model.py`, and optional
  local helper modules.
- `data/` contains local SQLite market/cache state and is not committed.

## Quick Start

```bash
uv sync
uv run python stock_bot.py
```

## CLI

Run CLI commands from the repo root:

```bash
uv run python CLI.py features
uv run python CLI.py list
uv run python CLI.py validate consolidation_breakout
uv run python CLI.py backtest consolidation_breakout --tickers AAPL --interval 1d --start 2020-01-01
uv run python CLI.py train ml_regime_hybrid --tickers AAPL,MSFT --interval 1d
uv run python CLI.py signal ml_regime_hybrid --tickers AAPL
```

See [CLI.md](CLI.md) for the full command reference.

## Strategy Development

Create and edit strategies through `CLI.py` or the GUI:

```bash
uv run python CLI.py init my_strategy
uv run python CLI.py edit my_strategy --add-feature RSI --feature-params period=14
uv run python CLI.py show-context my_strategy
uv run python CLI.py validate my_strategy
```

After changing `manifest.json`, regenerate `context.py`:

```bash
uv run python CLI.py sync my_strategy
```

`manifest.json` is source and should be tracked. `context.py` is generated but
tracked so strategies remain readable. `artifacts.joblib`, `diagnostics.json`,
`gui_prefs.json`, logs, caches, and databases are generated outputs.

## Docker Engine Stack

Run from `engine/`:

```bash
docker compose up -d
docker compose logs -f
docker compose down
```

The compose file mounts root `strategies/` at `/code/strategies` and root
`data/` at `/code/data`, so API/worker jobs use the same strategy workspace and
database as the CLI/GUI.

## Testing

The canonical test path is Docker from `engine/`:

```bash
docker compose --profile test up -d pytest
docker exec research_tester uv run pytest tests/
```

Local smoke checks can use the existing virtualenv or `uv run` from the repo
root, but Docker is the source of truth for CI-like validation.
