# Stock Trading Bot Framework

A modular Python framework for gathering historical stock data, managing a local market database, and executing trading algorithms in both backtesting and live simulation environments.

## Overview

This tool provides a foundation for quantitative analysis and automated swing trading. It handles the complexities of data ingestion and storage, allowing developers to focus on strategy logic.

## Key Features

- **Local Data Persistence**: Stores OHLCV (Open, High, Low, Close, Volume) data in a SQLite database to minimize API calls and improve backtesting performance.
- **Custom Timeframes**: Supports standard intervals (1h, 1d) and custom resampled intervals such as 4-hour bars.
- **Backtesting Engine**: Iterates through historical data to validate strategy logic against past market conditions.
- **Live Simulation**: Periodically fetches the latest market data to test strategy behavior in real-time.
- **Extensible Strategy Base**: Provides a structured class-based approach for implementing custom technical indicators and signals.

## Technical Stack

- **Python 3.13+**
- **uv**: Project and dependency management.
- **SQLAlchemy**: Database ORM for SQLite.
- **Pandas**: Data manipulation and analysis.
- **yfinance**: Market data retrieval.

## Installation

Ensure you have `uv` installed on your system.

1. Clone the repository.
2. Install dependencies:
   ```bash
   uv sync
   ```

## Usage

The application is controlled via a CLI interface with three primary modes:

### 1. Sync Data
Download historical data for a specific ticker and save it to the local database.
```bash
uv run python main.py --mode sync --ticker AAPL --interval 4h --period 1y
```

### 2. Snapshot (Bulk Sync)
The snapshot mode is designed for building a comprehensive local database for the Russell 1000 or top market companies.
- **Source**: Prioritizes `data/tickers.txt` if available; otherwise, it fetches from S&P 500, S&P 400, and NASDAQ 100 sources.
- **Decade History**: Automatically pulls the last 10 years for weekly/daily bars.
- **Multi-Interval**: Syncs `1w, 1d, 4h, 1h, 30m, 15m` intervals for every ticker.
- **Smart Skip**: Checks the database and skips any ticker/interval that is already up to date (within the last 6 months).
- **Auto-Period Scaling**: Respects `yfinance` limits (e.g., 2 years for 1h/4h, 60 days for 15m/30m).

```bash
uv run python main.py --mode snapshot
```

### 3. Ticker Extraction
If you have an iShares Russell 1000 ETF holdings Excel file (`.xls`), you can extract the exact tickers for the snapshot:
1. Place the file at `data/iShares-Russell-1000-ETF_fund.xls`.
2. Run the extraction script:
   ```bash
   uv run python extract_tickers.py
   ```
3. This creates `data/tickers.txt`, which the snapshot mode will use automatically.

### 4. Backtest
Run a strategy against the data stored in the local database.
```bash
uv run python main.py --mode backtest --ticker AAPL --interval 4h
```

### 3. Live Simulation
Run a strategy against real-time data updates.
```bash
uv run python main.py --mode live --ticker AAPL --interval 1h
```

## Strategy Development

To create a new strategy, inherit from the `Strategy` base class in `src/engine.py` and implement the `on_bar` method:

```python
class MyStrategy(Strategy):
    def on_bar(self, ticker, bar, history):
        # Your logic here
        pass
```

## Project Structure

- `src/database.py`: Handles SQLite schema and data persistence.
- `src/fetcher.py`: Manages yfinance API calls and data resampling.
- `src/engine.py`: Contains the core logic for backtesting and live execution.
- `main.py`: CLI entry point and sample strategy implementation.
