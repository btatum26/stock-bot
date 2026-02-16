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

### 2. Backtest
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
