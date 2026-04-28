"""
Centralized configuration for the GUI layer.
Edit these values to customize paths, timings, and layout constants.
"""

from engine.core.config import config as ENGINE_CONFIG

# --- Engine paths (resolved relative to the repo root) ---
WORKSPACE_DIR = ENGINE_CONFIG.STRATEGIES_FOLDER
DB_PATH = ENGINE_CONFIG.DB_PATH

# --- Debounce timings (milliseconds) ---
LOAD_DEBOUNCE_MS = 800    # delay after ticker input before chart fetch
Y_SCALE_DEBOUNCE_MS = 24  # Y-axis autoscale debounce in UnifiedPlot

# --- Period bar counts: maps period selector label → visible bar count ---
PERIOD_BARS = {
    "1mo":  21,
    "3mo":  63,
    "6mo":  126,
    "1y":   252,
    "2y":   504,
    "5y":   1260,
    "10y":  2520,
    "All":  None,
}

# --- Layout sizes (pixels) ---
SIDEBAR_MIN_WIDTH = 220
SUBPLOT_MIN_HEIGHT = 100
BACKTEST_PANEL_A_WIDTH = 350

# --- Fallback tickers when the local cache is empty ---
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL"]

# --- Signal chart default view window (bars) ---
SIGNAL_CHART_VIEW_BARS = 252

# --- Maximum bars loaded into the chart (tail slice) ---
MAX_CHART_BARS = 1500
