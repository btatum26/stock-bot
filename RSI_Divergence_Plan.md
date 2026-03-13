# RSI Divergence Strategy: Development Plan

## Current Status (V1.3)
- **Signal Logic**: Bullish RSI Divergence.
  - Price: Lower Low (LL).
  - RSI: Higher Low (HL) while in Oversold territory (< 30).
  - Confirmation: Requires a confirmed pivot low (3-bar window).
- **Model**: XGBoost Regressor (predicting `(MaxGain/Time) - (alpha * Drawdown)`).
- **Core Parameters**:
  - `oversold_threshold`: 30.0
  - `lookback_window`: 60 bars
  - `pivot_window`: 3 bars (confirmation delay)
- **Feature Set**:
  - RSI & RSI Pivot History.
  - Price Z-Scores & Volatility-Scaled Returns (ATR-based).
  - ATR Percent (Relative Volatility) & ATR Scale.
  - Distances to Support/Resistance Levels (Nearest Level & Dynamic Extraction).
  - Distances to all available Moving Averages (Dynamic Extraction).
  - Local Bar Normalization (Prices/Indicators).

## Planned Improvements

### Feature Engineering
- [ ] **Volume Profile**: Distance from Value Area High/Low.
- [ ] **Trend Alignment**: Add Higher Timeframe (HTF) trend detection.
- [ ] **Volatility Regimes**: Categorize market as "Chop", "Trending", or "Expanding".

### Model Optimization
- [ ] **Hyperparameter Tuning**: Grid search for `max_depth` and `n_estimators`.
- [ ] **Target Variable Refinement**: Experiment with different `drawdown_penalty` values.
- [ ] **Feature Selection**: Use XGBoost feature importance to prune low-signal inputs.

### Signal Filtering
- [x] **Divergence Validation**: Ensure price made a lower low while RSI made a higher low.
- [ ] **Time-of-Day Filter**: Avoid signals during low-liquidity sessions.

### ML Exit Strategy: Optimal Sell Prediction
- [ ] **Target Labeling**: Implement the `SellScore` formula to identify optimal historical exits.
  - $$SellScore_k = \frac{Gain_k}{(k - t_0)^\gamma} - (\alpha \times MaxDrawdown_{t_0 \rightarrow k})$$
  - **Gamma ($\gamma$)**: Time decay tuner (e.g., 0.5 for square root scaling).
  - **Alpha ($\alpha$)**: Multiplier for maximum drawdown penalty.
- [ ] **Exit Model**: Train a separate XGBoost Regressor to predict `SellScore` based on active trade features.
- [ ] **Dynamic Exit Logic**: Exit when predicted `SellScore` reaches a peak or crosses a threshold.

## Training & Validation Goals
- [ ] Achieve > 10% average score in training metrics across multiple stocks.
- [ ] Validate on unseen "Out-of-Sample" data.
- [ ] Compare performance against a "Buy & Hold" baseline.

## Technical Debt / Notes
- [ ] Optimization: Vectorize more feature calculations to speed up GUI training.
- [ ] Bug Watch: Monitor `extract_sr_features` for edge cases with overlapping levels.

---
*Updated: March 11, 2026*
