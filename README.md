# Stock Bot Pro

A quantitative analysis and strategy development platform designed for market data visualization, feature engineering, and machine learning model integration.

## Core Capabilities

### Visualization and Interaction
* High-performance candlestick charting using PyQt6 and pyqtgraph.
* Integrated volume overlays and dynamic crosshair tracking.
* Synchronized multi-pane layouts for secondary technical indicators.
* Rich metadata tooltips for signal events and trade markers.

### Feature Engineering
* Dynamic technical indicator system (RSI, ATR, Moving Averages, Support/Resistance).
* Real-time parameter tuning with immediate visual feedback.
* Modular architecture for adding custom technical features.

### Strategy and Machine Learning
* Python-based strategy scripting for signal generation and custom scoring logic.
* In-application model training using custom ticker baskets or randomized data slices.
* Visual scoring underlays to map strategy outputs directly onto price charts.
* Persistent model management for comparing multiple trained instances per strategy.

### Data Management
* SQLite-backed local persistence for fast historical data access.
* Automated synchronization with Yahoo Finance for data retrieval and gap filling.
* Strategy serialization (.strat files) to preserve workspace configurations, features, and models.

## Technical Stack
* Interface: PyQt6
* Charting: pyqtgraph
* Data: Pandas, NumPy, SQLAlchemy (SQLite)
* Ingestion: yfinance
* Environment: Python 3.13+ managed via uv

## Installation

Ensure the `uv` package manager is installed on your system.

1. Clone the repository.
2. Initialize the environment:
   ```bash
   uv sync
   ```

## Usage

### GUI Application
The primary interface for charting and strategy development.
```bash
uv run python stock_bot.py
```

### CLI Operations
For backend data management and maintenance tasks.
```bash
# Synchronize data for a specific ticker
uv run python CLI.py --mode sync --ticker AAPL --interval 1d --period 10y

# Bulk synchronize the top 1000 market companies
uv run python CLI.py --mode bulk_sync --period 5y

# Clean duplicate database records
uv run python CLI.py --mode clean
```

## Project Structure
* `stock_bot.py`: Main entry point for the desktop application.
* `CLI.py`: Command-line interface for data operations.
* `src/`: Core logic including the GUI orchestration, engine, and feature loaders.
* `src/gui_components/`: Modular UI elements for plots and control panels.
* `src/features/`: Implementations of technical indicators.
* `src/signals/`: Base classes for rule-based and machine learning signal models.
* `strategies/`: User strategy scripts and saved configuration files.
* `data/`: Local storage for the SQLite market database.
