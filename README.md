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

The application is controlled via a unified CLI and a desktop GUI.

### Desktop GUI
The full interactive experience with charts, indicators, and strategy training.
```bash
uv run python run_gui.py
```

### Unified CLI (`CLI.py`)

All command-line operations are now unified under `CLI.py`.

#### 1. Data Synchronization
Sync historical data for a specific ticker to the local database.
```bash
uv run python CLI.py --mode sync --ticker AAPL --interval 4h --period 1y
```

#### 2. Bulk Data Collection (Snapshot)
Build a comprehensive local database for the top 1000 market companies.
```bash
uv run python CLI.py --mode bulk_sync --period 10y
```

#### 3. Maintenance (Reset & Clean)
Clear data for a specific ticker or remove duplicates from the database.
```bash
# Reset a specific ticker/interval
uv run python CLI.py --mode reset --ticker AAPL --interval 4h

# Remove duplicate records across the entire DB
uv run python CLI.py --mode clean
```

#### 4. Backtesting & Live Simulation
Run the sample SMA Crossover strategy against historical or real-time data.
```bash
# Backtest
uv run python CLI.py --mode backtest --ticker AAPL --interval 4h

# Live Simulation
uv run python CLI.py --mode live --ticker AAPL --interval 1h
```

## Project Structure

- `run_gui.py`: Main entry point for the desktop application.
- `CLI.py`: Unified CLI for data management, backtesting, and maintenance.
- `utils/`: Core utility logic (imported by CLI.py).
  - `clean_db.py`: Logic for removing duplicates.
  - `reset_ticker.py`: Logic for clearing specific ticker data.
- `src/`: Core framework logic.
  - `engine.py`: Backtesting and live execution engines.
  - `database.py`: SQLite schema and persistence layer.
  - `snapshot.py`: Bulk data collection logic.
- `data/`: Storage for the SQLite database and internal strategies.
