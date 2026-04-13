# Stock Bot Pro

A quantitative finance platform for market visualization, strategy development, backtesting, and optimization.

## Overview

The project has two layers that work together:

- **GUI** (`src/`, `stock_bot.py`) — PyQt6 desktop app for charting, indicator overlays, and visual signal exploration
- **Research Engine** (`engine/`) — FastAPI + Redis/RQ backend for systematic strategy backtesting, hyperparameter optimization, and signal generation

The GUI connects to the engine via the `ModelEngine` facade class. The engine can also run standalone via CLI or as a Docker service.

## Quick Start

### Desktop Application
```bash
uv sync
uv run python stock_bot.py
```

### Research Engine (CLI)
```bash
cd engine
uv sync

# Backtest a strategy
uv run python main.py BACKTEST --strategy rsi_divergence --ticker AAPL --interval 1d

# Optimize hyperparameters
uv run python main.py TRAIN --strategy rsi_divergence --ticker AAPL --interval 1d

# Generate latest signal
uv run python main.py SIGNAL --strategy rsi_divergence --ticker AAPL
```

### Research Engine (Docker — full stack)
```bash
cd engine
docker compose up -d          # starts Redis + FastAPI + RQ Worker
docker compose logs -f        # tail logs
docker compose down           # stop
```

## Strategy Development

Strategies live in `engine/strategies/<name>/` and follow a three-file contract:

1. **`manifest.json`** — declare which features you need and what hyperparameters exist
2. **`context.py`** — auto-generated typed dataclass from the manifest (never edit manually)
3. **`model.py`** — your logic: implement `generate_signals()` returning a `pd.Series` in `[-1.0, 1.0]`

Scaffold a new strategy:
```bash
cd engine
uv run python main.py INIT --strategy my_strategy
# edit engine/strategies/my_strategy/model.py
uv run python main.py BACKTEST --strategy my_strategy --ticker SPY --interval 1d
```

After changing `manifest.json`, regenerate the context:
```bash
uv run python main.py SYNC --strategy my_strategy
```

## Technical Stack

| Layer | Technology |
|---|---|
| GUI | PyQt6, pyqtgraph |
| Data | Pandas, NumPy, SQLAlchemy (SQLite), yfinance |
| API | FastAPI, uvicorn |
| Job queue | Redis, RQ |
| Optimization | Optuna, CPCV (Combinatorial Purged Cross-Validation) |
| ML | scikit-learn, XGBoost, SciPy |
| Environment | Python 3.13+, uv |

## Project Structure

```
stock_bot/
├── stock_bot.py              # GUI entry point
├── CLI.py                    # legacy data-sync CLI (bulk yfinance downloads)
├── src/
│   ├── gui.py                # ChartWindow (main PyQt6 app)
│   ├── gui_components/       # modular UI panels (charts, controls, signals)
│   ├── features/             # indicator implementations for GUI overlays
│   └── signals/              # signal event models
├── engine/                   # research engine (separate package: model-engine)
│   ├── __init__.py           # ModelEngine facade + public API
│   ├── main.py               # CLI entry point
│   ├── core/                 # backtester, controller, features, metrics, workspace
│   ├── daemon/               # FastAPI server + RQ worker
│   ├── strategies/           # user strategy workspaces
│   └── tests/                # pytest suite (run via Docker)
└── data/                     # SQLite market database
```

## Installation

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone <repo>
cd stock_bot
uv sync           # installs GUI + engine deps
```

For the Docker stack, also install [Docker Desktop](https://www.docker.com/products/docker-desktop/).