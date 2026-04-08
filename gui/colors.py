"""
Centralized color palette for the GUI layer.
All component files import from here instead of scattering hex strings.
"""

# --- Backgrounds ---
BG_MAIN    = "#1e1e1e"   # primary window / chart canvas
BG_DARK    = "#121212"   # dock titles, deep panels
BG_INPUT   = "#2d2d2d"   # inputs, dropdowns
BG_PANEL   = "#252525"   # table headers, panel sections
BG_TABLE   = "#1a1a1a"   # table widget background
BG_CONSOLE = "#0a0a0a"   # log console

# Row / button fills
BG_ROW_GREEN      = "#142814"   # positive trade row
BG_ROW_RED        = "#281414"   # negative trade row
BG_BTN_RUN        = "#1a4a1a"   # Run Batch button
BG_BTN_ADD_HPARAM = "#1a3a1a"   # Add Hyperparameter button
BG_BTN_CANCEL     = "#4a1a1a"   # Cancel button

# --- Text ---
TEXT_PRIMARY   = "#dddddd"
TEXT_SECONDARY = "#aaaaaa"
TEXT_MUTED     = "#888888"
TEXT_WHITE     = "#ffffff"

# --- Accents ---
ACCENT_BLUE   = "#007acc"   # default button / selection highlight
ACCENT_GREEN  = "#aaff00"   # selected tab / positive highlight
ACCENT_CYAN   = "#00bfff"   # volatility features
ACCENT_YELLOW = "#ffd700"   # trend features
ACCENT_PURPLE = "#da70d6"   # volume features
ACCENT_MINT   = "#00fa9a"   # palette slot 6

# --- Signal / trade markers ---
SIGNAL_LONG  = "#00ff88"   # long-entry triangle (chart preview)
SIGNAL_SHORT = "#ff4444"   # short-entry triangle
SIGNAL_EXIT  = "#aaaaaa"   # exit circle
TRADE_ENTRY  = "#44ff88"   # backtest entry marker
TRADE_EXIT   = "#ff5555"   # backtest exit marker

# --- Performance color coding (leaderboard) ---
PERF_GOOD = "#44ff88"
PERF_OK   = "#dddd44"
PERF_BAD  = "#ff5555"

# --- Chart chrome ---
CHART_BG        = "#1a1a1a"    # signal / drill-down chart background
CHART_CROSSHAIR = "#888888"
CHART_AXIS      = "#444444"
CHART_GRID      = "#444444"

# --- Feature rendering ---
# Cycled for multi-line features; index 0 is also the palette fallback
FEATURE_PALETTE = ["#aaff00", "#00bfff", "#ff6b6b", "#ffd700", "#da70d6", "#00fa9a"]

FEATURE_CATEGORY_COLORS = {
    "Oscillators (Momentum)": "#aaff00",
    "Volatility":             "#00bfff",
    "Trend":                  "#ffd700",
    "Volume":                 "#da70d6",
    "Price Levels":           "#ff6b6b",
}

LEVEL_COLORS = {
    "Overbought": "#ff4444",
    "Oversold":   "#44ff44",
}
