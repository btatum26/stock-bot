import pandas as pd
from context import Context
from engine.core.controller import SignalModel


class MacroRegimeRotation(SignalModel):

    def generate_signals(self, df: pd.DataFrame, context: Context, params: dict, artifacts: dict) -> pd.Series:
        # ── Component 1: Credit spread momentum ───────────────────────────────
        # HYSpread_ROC5 is the 5-day pct change of HY OAS (pre-computed by feature).
        # Compute a rolling z-score over the hyperparameter lookback window,
        # then invert: tightening (negative ROC) → bullish → score near 1.0.
        hy_roc5 = df[context.features.HYSPREAD_ROC5]
        lb = int(params['credit_spread_lookback'])
        hy_z = (hy_roc5 - hy_roc5.rolling(lb).mean()) / hy_roc5.rolling(lb).std()
        credit_score = (-hy_z).clip(-2, 2) / 4 + 0.5  # [-2,2] → [0,1]

        # ── Component 2: NFCI range position ──────────────────────────────────
        # Position of current NFCI within its trailing 26-week min/max range.
        # Inverted: lower NFCI (loose conditions) → score near 1.0.
        nfci = df[context.features.NFCI_LEVEL]
        nfci_window = int(params['nfci_lookback_weeks']) * 5  # weeks → trading days
        nfci_min = nfci.rolling(nfci_window).min()
        nfci_max = nfci.rolling(nfci_window).max()
        nfci_score = 1.0 - (nfci - nfci_min) / (nfci_max - nfci_min + 1e-8)
        nfci_score = nfci_score.clip(0, 1)

        # ── Component 3: VIX term structure ───────────────────────────────────
        # Linear interpolation: bull threshold (deep contango) → 1.0,
        # bear threshold (backwardation) → 0.0.
        ratio = df[context.features.VIXTERMSTRUCTURE]
        bull = float(params['vix_ratio_bull'])
        bear = float(params['vix_ratio_bear'])
        vts_score = (bear - ratio) / (bear - bull)
        vts_score = vts_score.clip(0, 1)

        # ── Composite: equal-weight average ───────────────────────────────────
        # Stays in [0, 1] (long-only): 0 = full cash, 1 = fully invested.
        composite = (credit_score + nfci_score + vts_score) / 3
        return composite.reindex(df.index).fillna(0.0)
