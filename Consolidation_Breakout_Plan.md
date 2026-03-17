# Consolidation Breakout Strategy (Linear Regression)

## Objective
Identify periods of price consolidation using linear regression and signal a "Buy" when price breaks out above the upper boundary of the consolidation channel.

## Core Concept
1.  **Consolidation Detection**: Use a sliding window of $N$ bars to calculate a Linear Regression line ($y = mx + b$).
2.  **Channel Boundaries**:
    *   Find the **Maximum Deviation** above the regression line within the window ($MaxDiff = \max(Price_i - RegLine_i)$).
    *   Find the **Minimum Deviation** below the regression line within the window ($MinDiff = \min(Price_i - RegLine_i)$).
    *   The "Consolidation Channel" is defined by these max/min boundaries relative to the regression line.
3.  **Breakout Signal**:
    *   A **Buy Signal** is triggered when the current price closes above the `Upper Boundary` (Regression Line + MaxDiff).
    *   **Signal Strength**: The duration of the consolidation (window size $N$) determines the strength of the signal. Longer consolidation leads to higher conviction.
4.  **Risk Management**:
    *   **Stop Loss**: Set at a configurable percentage below the `Lower Boundary` (Regression Line + MinDiff) of the consolidation period.
    *   **Take Profit**: Can be based on a fixed Risk/Reward ratio or a trailing stop.

## Implementation Details

### Parameters (Signal Parameters)
*   `window_size` (int): Number of bars to look back for consolidation (e.g., 20, 50, 100).
*   `min_slope` (float): Maximum absolute slope allowed for the period to be considered "consolidating" (flatness filter).
*   `breakout_threshold` (float): Optional multiplier or fixed percentage above the upper boundary to confirm the breakout.
*   `stop_loss_pct` (float): Percentage below the consolidation low to place the stop loss.

### Strategy Logic (StrategySignal Class)
*   **`on_bar` / `generate_signals`**:
    *   Iterate through the data.
    *   For each bar, perform linear regression on the previous `window_size` bars.
    *   Calculate the residuals (Price - RegLine).
    *   Check if the current bar's `Close` > `RegLine_end` + `max(residuals)`.
    *   Ensure the slope of the regression line is within the `min_slope` tolerance (optional).

## Strength Calculation
The strategy can output a "Strength" score derived from:
*   $\text{Strength} = \text{window\_size} \times (1 / \text{ChannelWidth})$
*   This rewards longer periods of tighter consolidation.

## Verification Plan
1.  **Visual Validation**: Overlay the Linear Regression line and the calculated channel boundaries on the chart.
2.  **Backtest**: Run the strategy on historical data (e.g., AAPL 1h) to verify signal accuracy.
3.  **Sensitivity Analysis**: Test various `window_size` values to see how they impact profitability and signal frequency.
