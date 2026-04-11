import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from context import Context
from engine.core.controller import SignalModel


class RandomForestModel(SignalModel):
    """Long-only random forest regime classifier.

    Predicts one of three regimes over a lookforward window:
        0 = down  (forward return < down_threshold)
        1 = chop  (forward return between thresholds -> stay flat)
        2 = up    (forward return > up_threshold)

    Only goes long. The down and chop classes both map to flat; the
    model enters long when the up class is both dominant and confident.

    Multi-ticker training: ``build_labels`` is called once per ticker by
    the trainer (it only ever sees one asset at a time). The trainer
    concatenates features and labels across tickers, then calls
    ``fit_model`` once on the pooled matrix.
    """

    _FEATURE_COLS = [
        "Norm_MovingAverage_50_EMA",
        "Norm_MovingAverage_20_EMA",
        "Norm_MovingAverage_200_EMA",
        "RSI_14",
        "Norm_AverageTrueRange_14",
    ]

    def build_labels(
        self, df: pd.DataFrame, context: Context, params: dict
    ) -> pd.Series:
        """Computes the 3-class regime target on a single-ticker df.

        Returns a Series aligned to ``df.index`` with NaN for the last
        ``lookforward`` bars (no forward return available).
        """
        lookforward = int(params.get("lookforward", 5))
        up_threshold = float(params.get("up_threshold", 0.01))
        down_threshold = float(params.get("down_threshold", -0.01))

        future_close = df["close"].shift(-lookforward)
        forward_return = (future_close - df["close"]) / df["close"]

        y = pd.Series(np.nan, index=df.index, dtype="float64")
        y[forward_return > up_threshold] = 2.0
        y[forward_return < down_threshold] = 0.0
        y[(forward_return >= down_threshold) & (forward_return <= up_threshold)] = 1.0
        return y

    def fit_model(self, X, y, params: dict) -> dict:
        """Fits a random forest classifier on the pooled (X, y) matrix."""
        n_estimators = int(params.get("n_estimators", 200))
        max_depth = params.get("max_depth", 3)
        if max_depth is not None:
            max_depth = int(max_depth)
        min_samples_leaf = int(params.get("min_samples_leaf", 50))

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=42,
            n_jobs=-1,
            # Chop tends to dominate the sample; balance so up/down aren't ignored.
            class_weight="balanced",
        )
        clf.fit(X, y.astype(int))

        return {"model": clf}

    def generate_signals(
        self,
        df: pd.DataFrame,
        context: Context,
        params: dict,
        artifacts: dict,
    ) -> pd.Series:
        """Long-only signals from regime-class probabilities.

        Enters long when the up-class probability clears the entry
        threshold AND exceeds the down-class probability. Exits to
        flat when the up-class probability decays below the exit
        threshold. Never shorts.
        """
        clf = artifacts["model"]
        feature_cols = artifacts.get("feature_cols", self._FEATURE_COLS)

        X = df[feature_cols].copy()
        valid_mask = X.notna().all(axis=1)

        classes = list(clf.classes_)
        up_proba = pd.Series(0.0, index=df.index)
        down_proba = pd.Series(0.0, index=df.index)

        if valid_mask.any():
            proba = clf.predict_proba(X[valid_mask])
            if 2 in classes:
                up_proba.loc[valid_mask] = proba[:, classes.index(2)]
            if 0 in classes:
                down_proba.loc[valid_mask] = proba[:, classes.index(0)]

        # With 3 classes the max probability rarely exceeds ~0.5, so
        # thresholds are lower than the binary model's defaults.
        entry_threshold = float(params.get("entry_threshold", 0.45))
        exit_threshold = float(params.get("exit_threshold", 0.30))

        signals = np.zeros(len(df))
        position = 0.0

        up_vals = up_proba.to_numpy()
        dn_vals = down_proba.to_numpy()

        for i in range(len(df)):
            up = float(up_vals[i])
            dn = float(dn_vals[i])

            if position == 0.0:
                if up >= entry_threshold and up > dn:
                    position = 1.0
            elif position == 1.0:
                if up < exit_threshold:
                    position = 0.0

            signals[i] = position

        return pd.Series(signals, index=df.index)
