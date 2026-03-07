# ML Strategy API & Model Management Plan

## Objective
Standardize how Machine Learning models are defined, trained, and managed across all strategies. This will turn `.strat` files into self-contained "AI Trading Capsules" that hold the logic, the feature configuration, and multiple trained model iterations.

---

## 1. Core Strategy API Standard
Every strategy script (`StrategySignal` class) should follow a standard interface:

### `train(self, df, feature_data, model_type, settings)`
*   **Purpose**: Logic to transform raw data into a trained model instance.
*   **Inputs**: 
    *   `df`: The OHLCV dataframe.
    *   `feature_data`: Computed feature series.
    *   `model_type`: String identifier (e.g., "RandomForest", "XGBoost").
    *   `settings`: Dict of hyperparameters (e.g., `n_estimators`, `max_depth`).
*   **Returns**: A trained model object (compatible with `pickle`).

### `generate_signals(self, df, feature_data)`
*   **Updated Logic**: Should now check for an "active" model instance within `self.available_models` or fallback to rule-based logic.

---

## 2. Updated `.strat` File Structure
The binary strategy file will be expanded to store:
*   **`metadata`**: Author, version, creation date.
*   **`feature_config`**: GUI-defined features (RSI, ATR, etc.).
*   **`script_content`**: The Python signal logic.
*   **`model_definitions`**: A list of *available* model architectures for this strategy and their default hyperparameters.
*   **`model_instances`**: A dictionary of trained runs:
    ```python
    {
        "uuid_or_name": {
            "weights": object,      # The actual trained model
            "timestamp": datetime,
            "comment": str,
            "settings": dict,       # hyperparameters used
            "training_scope": {     # data context
                "tickers": list,
                "interval": str,
                "date_range": tuple
            },
            "metrics": dict         # (NEW) Accuracy, F1-score, etc.
        }
    }
    ```

---

## 3. GUI Enhancements (Signals Tab)
The sidebar will be divided into three logical sections:

### A. Training Configuration
*   **Ticker(s) Selection**: Ability to train on a specific ticker or a basket of tickers. Can select n random tickers to train on.
*   **Time Range Picker**: Select the day to look back from and the number of bars to look back from. The lookback date can be picked randomly or set to n random lookback dates
*   **Validation Split**: A slider to set % of data kept for testing (e.g., 80% train / 20% test).
*   **Hyperparameter Editor**: A small form generated based on the selected `model_type`.

### B. Model Instance Browser
*   **Table View**: Columns for `Name/Comment`, `Date`, `Metric (e.g. Acc)`, and `Status`.
*   **Action Bar**: 
    *   `Set Active`: Mark a specific instance to be used for real-time signals.
    *   `Delete`: Remove instance from `.strat`.
    *   `Rename/Comment`: Edit metadata.

### C. Training Feedback
*   **Progress Bar**: Real-time feedback for long training runs.
*   **Log Console**: Displaying training logs, data preparation steps, and final metrics.

---

## 4. Proposed Additions (The "What Else?")

### 1. Automated Validation Metrics
Don't just train; evaluate. The `train` method should automatically output metrics like **Precision, Recall, and Sharps Ratio** (if a mini-backtest is run on the test set). This allows the user to compare model iterations scientifically.

### 2. Feature Importance Visualization
If using models like Random Forest, the GUI should show a chart of which features (RSI vs Volume vs ATR) the model is actually relying on. This helps prune useless features.

### 3. Training Threading (Performance)
Training can freeze the UI. All training calls must be wrapped in `QThread` or `QProcess` to keep the main chart interactive while the model "thinks."

### 4. Model Versioning & Portability
The ability to "Export" a single trained model instance from one strategy and "Import" it into another if they share similar feature sets.

### 5. "Safe Apply" Guardrails
A system to check if the current chart interval matches the training interval. (e.g., Warning: "This model was trained on 1h data, you are applying it to 1m data. Results may be unreliable.")
