# Research Engine

Backend package for Stock Bot Pro. It owns feature computation, cached market
data access, backtesting, model training, signal generation, and the optional
FastAPI + Redis/RQ job daemon.

## Docker Stack

Run from this `engine/` directory:

```bash
docker compose up -d
docker compose logs -f
docker compose down
```

The stack starts Redis, the FastAPI service, and the worker. It mounts:

- `../strategies` -> `/code/strategies`
- `../data` -> `/code/data`
- `.` -> `/code/engine`

Those mounts keep Docker behavior aligned with the root GUI and `CLI.py`.

## CLI And GUI

The command-line interface lives at the repo root:

```bash
uv run python CLI.py list
uv run python CLI.py features
uv run python CLI.py backtest consolidation_breakout --tickers AAPL --interval 1d
uv run python CLI.py train ml_regime_hybrid --tickers AAPL,MSFT --interval 1d
uv run python CLI.py signal ml_regime_hybrid --tickers AAPL
```

The desktop GUI also runs from the repo root:

```bash
uv run python stock_bot.py
```

## Testing

Run the test suite inside Docker:

```bash
docker compose --profile test up -d pytest
docker exec research_tester uv run pytest tests/
```

Single-file example:

```bash
docker exec research_tester uv run pytest tests/integration/test_api.py
```
