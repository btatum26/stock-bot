# Research Engine

A modular platform for financial strategy development, backtesting, and optimization.


## Deployment & Execution

### 1. Docker Deployment (Recommended)
The easiest way to run the full Research Engine stack (API, Worker, and Redis) is using Docker Compose.

```bash
# Start the entire stack in the background
docker compose up -d

# View logs for all services
docker compose logs -f

# Stop and remove containers
docker compose down
```

### 2. Command Line Interface (CLI)
Used for direct interaction with the engine for backtesting, training, and signal generation. Requires a local Python environment.

```bash
# Install dependencies
uv sync

# Launch the Graphical User Interface
uv run python main.py --gui

# Run a backtest
uv run python main.py BACKTEST --strategy momentum_surge --ticker AAPL --interval 1h

# Start an optimization/training job
uv run python main.py TRAIN --strategy momentum_surge --ticker TSLA --interval 1d

# Generate the latest signal only
uv run python main.py SIGNAL --strategy momentum_surge --ticker SPY
```

### 3. Graphical User Interface (GUI)
A Tkinter-based dashboard for strategy management, configuration, and remote job submission.

```bash
# Can be launched via the CLI
uv run python main.py --gui

# Or directly
uv run python src/gui_launcher.py
```
**Features:**
- Dynamic configuration of strategy hyperparameters via `manifest.json`.
- Strategy workspace synchronization (updates `context.py` and local metadata).
- Real-time job status tracking from the Compute Daemon.
- Strategy creation wizard.

---

## Testing

### Docker Testing
Run the test suite inside a controlled container environment:
First you must warm the container by running 
```bash
docker compose --profile test up -d pytest
```
once you have a container running, you can simply run 
```bash
docker exec -it research_tester uv run pytest tests/
```
The suite covers data fetching, backtesting logic, API endpoints, and the feature blast shield.
