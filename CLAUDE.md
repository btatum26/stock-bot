# CLAUDE.md

Canonical project guidance for code agents working in this repository.

## Testing

Run tests via Docker from inside `engine/`:

```bash
docker compose --profile test up -d pytest
docker exec research_tester uv run pytest tests/
docker exec research_tester uv run pytest tests/integration/test_api.py
```

Docker is the source of truth because it matches the API/worker environment.

## CLI

`CLI.py` is the primary strategy authoring and execution tool. Run it from the
repo root:

```bash
uv run python CLI.py features
uv run python CLI.py list
uv run python CLI.py data-info --ticker AAPL
uv run python CLI.py init <name>
uv run python CLI.py edit <name> --add-feature RSI --feature-params period=14
uv run python CLI.py show-context <name>
uv run python CLI.py validate <name>
uv run python CLI.py backtest <name> --tickers AAPL --interval 1d --start 2020-01-01
uv run python CLI.py train <name> --tickers AAPL --interval 1d
uv run python CLI.py signal <name> --tickers AAPL
```

See `CLI.md` for the full command reference.

## Running The Stack

```bash
# Full stack, from engine/
docker compose up -d

# GUI, from repo root
uv sync
uv run python stock_bot.py
```

Docker mounts root `strategies/` to `/code/strategies` and root `data/` to
`/code/data`, so API/worker jobs use the same workspace as CLI/GUI runs.

## Architecture

This repo has two layers:

1. GUI layer: `stock_bot.py` -> `gui/gui.py` (`ChartWindow`)
   - UI components: `gui/gui_components/`
   - Chart managers: `gui/chart/`
   - GUI feature adapters: `gui/features/`
   - Uses `engine/bridge.py` (`ModelEngine`) as its backend facade

2. Research engine: `engine/`
   - CLI facade: root `CLI.py`
   - Programmatic facade: `engine/bridge.py`
   - API: `engine/daemon/main.py`
   - Worker: `engine/daemon/worker.py`
   - Core: `engine/core/`
   - Tests: `engine/tests/`

Strategies live at root `strategies/<name>/`:

- `manifest.json`: tracked source config for features, hyperparameters, bounds, and training settings
- `context.py`: generated typed context, tracked for readability
- `model.py`: user-authored strategy logic
- helper `.py` files: optional local support modules
- `artifacts.joblib`, `diagnostics.json`, `gui_prefs.json`: generated outputs, ignored

## Logging

All engine operational logging goes through `engine/core/logger.py`.

Logger hierarchy:

```text
model-engine
model-engine.data
model-engine.daemon
model-engine.core
model-engine.features.*
```

Use:

```python
from engine.core.logger import logger, daemon_logger, data_logger
```

If a subsystem needs its own child logger, use `logging.getLogger("model-engine.<subsystem>")`.
Do not use `logging.basicConfig()` in engine code. Do not use `print()` for engine
operational output; CLI/UI formatting is the exception.

Logs write to `engine/logs/` by default, or `LOG_DIR` if set. Console is INFO+.
`engine.log` is DEBUG+, `errors.log` is WARNING+. In Docker, API and worker are
separate processes, so file rotation is best-effort; prefer `docker compose logs`
for operational tailing.

## Conventions

- Dependencies are managed with `uv`; do not use `pip install` directly.
- `PYTHONPATH=/code` is set in Docker.
- `FRED_API_KEY` loads from `engine/.env`; commit only `engine/.env.example`.
- `manifest.json` files are source and should be committed.
- Commits must use the repo/user Git author only. Do not add AI, assistant, or
  co-author attribution lines.
