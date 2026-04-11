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



Strategy Plans:

Why Regime Models Can Work Better
Instead of asking "will price go up 10% in 15 days?" (very hard), you ask "what state is the market in right now?" (easier). Then you apply rules that historically work in each state.
The key insight: regimes are more persistent than returns. A trending market tends to stay trending. A choppy market tends to stay choppy. This persistence is what makes regime models tractable.

Regime Framework Options
Here are three approaches, ordered from simplest to most sophisticated:
Option 1: Volatility Regime (Simplest)
Markets alternate between low-vol (trending) and high-vol (choppy/reverting) states.
Regimes:
- LOW_VOL:  ATR percentile < 30 over trailing 252 days
- HIGH_VOL: ATR percentile > 70
- NORMAL:   everything else

Rules:
- LOW_VOL  → trend-follow (buy breakouts, ride momentum)
- HIGH_VOL → mean-revert (fade extremes, tight stops)
- NORMAL   → reduce size or stay flat
Why it works: Volatility is autocorrelated. Low-vol periods cluster together. Your ATR feature already captures this — you just need to use it as a regime indicator rather than a prediction input.
Option 2: Trend Regime (Your Consolidation/Breakout Idea)
This is what you mentioned. The market cycles through:
Regimes:
- TRENDING_UP:   price > 50 EMA > 200 EMA, ADX > 25
- TRENDING_DOWN: price < 50 EMA < 200 EMA, ADX > 25
- CONSOLIDATION: ADX < 20, price within N% of 50 EMA

Rules:
- TRENDING_UP   → buy pullbacks to 20 EMA, trail stops
- TRENDING_DOWN → stay flat (long-only) or short
- CONSOLIDATION → wait for breakout, or fade range edges
The problem you identified: This is simple, but simple isn't bad. The issue is that consolidation → breakout transitions are noisy. Many "breakouts" fail.
Solution: Don't predict the breakout. Wait for confirmation, then classify the new regime.
Option 3: Hidden Markov Model (Most Principled)
Let the data define the regimes rather than hand-coding rules.
pythonfrom hmmlearn import GaussianHMM

# Features for HMM: returns, volatility, maybe volume
X = df[['log_return', 'atr_pct', 'volume_zscore']].values

model = GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
model.fit(X)

df['regime'] = model.predict(X)
The HMM will discover regimes statistically — they often correspond to:

Regime 0: Low-vol uptrend
Regime 1: High-vol crash/recovery
Regime 2: Choppy sideways

Advantage: Data-driven, captures non-obvious patterns.
Disadvantage: Regimes can be unstable across time periods, and you still need rules for each regime.

My Recommendation: Hybrid Approach
Combine rule-based regime classification with ML-based entry timing.
Step 1: Define Regimes with Clear Rules
pythondef classify_regime(df: pd.DataFrame) -> pd.Series:
    """
    Returns regime labels:
    0 = RISK_OFF (downtrend or high volatility)
    1 = NEUTRAL  (no clear trend, normal vol)
    2 = RISK_ON  (uptrend, low/normal volatility)
    """
    # Trend condition
    uptrend = (df['close'] > df['ema_50']) & (df['ema_50'] > df['ema_200'])
    downtrend = (df['close'] < df['ema_50']) & (df['ema_50'] < df['ema_200'])
    
    # Volatility condition (ATR percentile over 1 year)
    atr_pct = df['atr'].rolling(252).apply(lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()))
    high_vol = atr_pct > 0.7
    low_vol = atr_pct < 0.3
    
    regime = pd.Series(1, index=df.index)  # Default: NEUTRAL
    regime[uptrend & ~high_vol] = 2        # RISK_ON
    regime[downtrend | high_vol] = 0       # RISK_OFF
    
    return regime
Step 2: Use ML Only Within RISK_ON Regime
Instead of predicting returns, predict: "Given we're in RISK_ON, is this a good entry point?"
pythondef build_labels(self, df, context, params):
    regime = classify_regime(df)
    
    # Only label RISK_ON periods
    # Target: does the next 10 days stay above entry price?
    future_min = df['low'].rolling(10).min().shift(-10)
    good_entry = (future_min > df['close'] * 0.97)  # Doesn't drop more than 3%
    
    y = pd.Series(np.nan, index=df.index)
    y[(regime == 2) & good_entry] = 1  # Good entry in uptrend
    y[(regime == 2) & ~good_entry] = 0  # Bad entry in uptrend
    # Leave RISK_OFF and NEUTRAL as NaN — don't train on them
    
    return y
Step 3: Trading Rules
pythondef generate_signals(self, df, context, params, artifacts):
    regime = classify_regime(df)
    entry_quality = predict_entry_quality(df, artifacts)  # ML model
    
    signals = np.zeros(len(df))
    position = 0.0
    
    for i in range(len(df)):
        if position == 0:
            # Only enter in RISK_ON regime with good entry signal
            if regime[i] == 2 and entry_quality[i] > 0.5:
                position = 1.0
        else:
            # Exit if regime changes OR entry quality collapses
            if regime[i] != 2 or entry_quality[i] < 0.3:
                position = 0.0
        
        signals[i] = position
    
    return pd.Series(signals, index=df.index)

Why This Works Better
Old ApproachNew ApproachPredict 15-day forward returns (hard)Classify current regime (easier)Model must learn everythingRules handle regime, ML handles timingTrades in all conditionsOnly trades in favorable regimesFeatures uncorrelated with targetFeatures directly define regime

Concrete Next Steps

Add ADX to your feature list — it's the standard trend strength indicator
Add ATR percentile — not raw ATR, but its rank over trailing 252 days
Implement classify_regime() as a rule-based function
Change your target — instead of "up/down/chop returns", use "good entry / bad entry within uptrend"
Filter training data — only train on RISK_ON periods

Would you like me to write out a complete model.py implementing this hybrid approach? I can adapt it to your existing system architecture.