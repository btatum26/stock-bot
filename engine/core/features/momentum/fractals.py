from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature


@register_feature("Fractals")
class FractalFeatures(Feature):
    """
    Identifies swing-point fractals and classifies market structure.

    A fractal high at bar T is detected when high[T] is the maximum of the
    centered window [T-n, ..., T, ..., T+n].  Because this requires n bars
    of right-side data, the last n bars of every output will be NaN/False.

    The marker is stamped AT the swing bar (T), not the confirmation bar (T+n).
    This means the calculation uses n bars of look-ahead — suitable for
    visualization and backtesting replay but not for live execution.

    Custom data sources
    -------------------
    Pass ``high_col`` and/or ``low_col`` in params to run fractals over any
    column in the DataFrame — for example, an RSI series.  If only one is
    supplied the other still falls back to the price High/Low columns.

    Outputs
    -------
    Is_Fractal_High / Is_Fractal_Low
        Sparse boolean marker at each swing bar.

    Fractal_High_Price / Fractal_Low_Price
        Value of the source column at the swing bar.  NaN elsewhere.

    Last_Fractal_High / Last_Fractal_Low
        Forward-filled running resistance / support level.

    Prev_Fractal_High / Prev_Fractal_Low
        The fractal before the current one — used for HH/LH/HL/LL detection
        and divergence calculations.

    Struct_High
        Market-structure label for highs, forward-filled:
          +1 = Higher High (HH)   -1 = Lower High (LH)   0 = no fractal yet

    Struct_Low
        Market-structure label for lows, forward-filled:
          +1 = Higher Low (HL)   -1 = Lower Low (LL)   0 = no fractal yet

    Bars_Since_Last_High / Bars_Since_Last_Low
        Integer count of bars since the most recent confirmed fractal.
    """

    @property
    def name(self) -> str:
        return "Fractals"

    @property
    def description(self) -> str:
        return "Swing-point fractal identification and market structure classification."

    @property
    def category(self) -> str:
        return "Oscillators (Momentum)"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "fractal_n": 5,
            "source_col": "",   # "" = use price High/Low; any df column name otherwise
        }

    @property
    def source_param_keys(self) -> List[str]:
        """Keys that are dynamic column selectors — excluded from column-name generation."""
        return ["source_col"]

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            # --- Fractal markers (on price / source pane) ---
            OutputSchema(name="Is_Fractal_High",    output_type=OutputType.MARKER, pane=Pane.OVERLAY),
            OutputSchema(name="Is_Fractal_Low",     output_type=OutputType.MARKER, pane=Pane.OVERLAY),

            # --- Values at the swing point (sparse) ---
            OutputSchema(name="Fractal_High_Price", output_type=OutputType.LINE,   pane=Pane.OVERLAY),
            OutputSchema(name="Fractal_Low_Price",  output_type=OutputType.LINE,   pane=Pane.OVERLAY),

            # --- Running levels (forward-filled) ---
            OutputSchema(name="Last_Fractal_High",  output_type=OutputType.LINE,   pane=Pane.OVERLAY),
            OutputSchema(name="Last_Fractal_Low",   output_type=OutputType.LINE,   pane=Pane.OVERLAY),

            # --- Previous fractal (for divergence / structure comparison) ---
            OutputSchema(name="Prev_Fractal_High",  output_type=OutputType.LINE,   pane=Pane.OVERLAY),
            OutputSchema(name="Prev_Fractal_Low",   output_type=OutputType.LINE,   pane=Pane.OVERLAY),

            # --- Market structure labels ---
            OutputSchema(name="Struct_High",        output_type=OutputType.LINE,   pane=Pane.NEW),
            OutputSchema(name="Struct_Low",         output_type=OutputType.LINE,   pane=Pane.NEW),

            # --- Timing ---
            OutputSchema(name="Bars_Since_Last_High", output_type=OutputType.LINE, pane=Pane.NEW),
            OutputSchema(name="Bars_Since_Last_Low",  output_type=OutputType.LINE, pane=Pane.NEW),
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prev_values(sparse: pd.Series, confirmed_mask: pd.Series) -> pd.Series:
        """For each confirmation bar, return the value from the previous confirmation bar.

        Builds a series that is NaN everywhere except at confirmation bars,
        where it holds the PRIOR confirmed value.  Forward-filled afterward
        so strategies can compare current vs. previous at any bar.
        """
        conf_idx = confirmed_mask[confirmed_mask].index
        prev_vals = sparse.loc[conf_idx].shift(1)
        out = pd.Series(np.nan, index=sparse.index, dtype=float)
        out.loc[conf_idx] = prev_vals.values
        return out.ffill()

    @staticmethod
    def _bars_since(confirmed_mask: pd.Series) -> pd.Series:
        """Integer count of bars elapsed since the last True in confirmed_mask."""
        conf_pos = np.where(confirmed_mask)[0]
        bar_idx  = np.arange(len(confirmed_mask))
        last_pos = pd.Series(np.nan, index=confirmed_mask.index, dtype=float)
        last_pos.iloc[conf_pos] = conf_pos.astype(float)
        last_pos = last_pos.ffill()
        return (bar_idx - last_pos).where(last_pos.notna())

    @staticmethod
    def _struct_label(sparse_current: pd.Series,
                      sparse_prev: pd.Series,
                      confirmed_mask: pd.Series,
                      positive_if_higher: bool) -> pd.Series:
        """
        Returns +1 / -1 at each confirmation bar, forward-filled.

        positive_if_higher=True  → +1 when current > prev  (HH for highs, HL for lows)
        positive_if_higher=False → +1 when current < prev  (inverse)
        """
        label = pd.Series(0.0, index=sparse_current.index)
        conf_idx = confirmed_mask[confirmed_mask].index
        if len(conf_idx) == 0:
            return label

        cur  = sparse_current.loc[conf_idx]
        prev = sparse_prev.loc[conf_idx]

        # Use .values to get an unambiguous numpy bool array for Index indexing
        valid_mask = (cur.notna() & prev.notna()).values
        if not valid_mask.any():
            return label

        valid_idx  = conf_idx[valid_mask]
        cur_vals   = cur.values[valid_mask]
        prev_vals  = prev.values[valid_mask]

        if positive_if_higher:
            label.loc[valid_idx] = np.where(cur_vals > prev_vals, 1.0, -1.0)
        else:
            label.loc[valid_idx] = np.where(cur_vals < prev_vals, 1.0, -1.0)

        # Forward-fill so the current structure is always accessible
        label = label.replace(0.0, np.nan).ffill().fillna(0.0)
        return label

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        n = int(params.get("fractal_n", 5))
        window = 2 * n + 1

        # ---- Resolve source columns ----
        # source_col: single series used for both high and low fractal detection.
        # "" (widget default) means use price High / Low columns.
        source_col = params.get("source_col") or None

        if source_col is not None:
            src  = df[source_col]
            high = src
            low  = src
        else:
            high = df["High"] if "High" in df.columns else df["high"]
            low  = df["Low"]  if "Low"  in df.columns else df["low"]

        # ---- Fractal detection ----
        # center=True rolling finds the peak/valley AT bar T across [T-n … T+n].
        # Shifting by n places the signal at the CONFIRMATION bar T+n — the first
        # bar at which the fractal can be known without look-ahead.
        is_peak   = (high == high.rolling(window=window, center=True).max()).fillna(False)
        is_valley = (low  == low.rolling( window=window, center=True).min()).fillna(False)

        is_fractal_high = is_peak.shift(n).fillna(False)
        is_fractal_low  = is_valley.shift(n).fillna(False)

        # ---- Values at the swing point (price AT bar T, stamped at confirmation bar T+n) ----
        fractal_high_price = high.shift(n).where(is_fractal_high).astype(float)
        fractal_low_price  = low.shift(n).where(is_fractal_low).astype(float)

        # ---- Running levels ----
        last_fractal_high = fractal_high_price.ffill()
        last_fractal_low  = fractal_low_price.ffill()

        # ---- Previous fractals ----
        prev_fractal_high = self._prev_values(fractal_high_price, is_fractal_high)
        prev_fractal_low  = self._prev_values(fractal_low_price,  is_fractal_low)

        # ---- Market structure ----
        # Struct_High: +1 = HH (higher high), -1 = LH (lower high)
        struct_high = self._struct_label(
            fractal_high_price, prev_fractal_high, is_fractal_high,
            positive_if_higher=True,
        )
        # Struct_Low: +1 = HL (higher low), -1 = LL (lower low)
        struct_low = self._struct_label(
            fractal_low_price, prev_fractal_low, is_fractal_low,
            positive_if_higher=True,
        )

        # ---- Bars since last fractal ----
        bars_since_high = self._bars_since(is_fractal_high)
        bars_since_low  = self._bars_since(is_fractal_low)

        # ---- Assemble output ----
        # Strip source-selector keys so column names stay stable regardless of
        # which data source is selected (e.g. "Fractals_5_IS_FRACTAL_HIGH" always).
        p = {k: v for k, v in params.items() if k not in self.source_param_keys}
        G = self.generate_column_name

        data = {
            G("Fractals", p, "Is_Fractal_High"):       is_fractal_high,
            G("Fractals", p, "Is_Fractal_Low"):        is_fractal_low,
            G("Fractals", p, "Fractal_High_Price"):    fractal_high_price,
            G("Fractals", p, "Fractal_Low_Price"):     fractal_low_price,
            G("Fractals", p, "Last_Fractal_High"):     last_fractal_high,
            G("Fractals", p, "Last_Fractal_Low"):      last_fractal_low,
            G("Fractals", p, "Prev_Fractal_High"):     prev_fractal_high,
            G("Fractals", p, "Prev_Fractal_Low"):      prev_fractal_low,
            G("Fractals", p, "Struct_High"):           struct_high,
            G("Fractals", p, "Struct_Low"):            struct_low,
            G("Fractals", p, "Bars_Since_Last_High"):  bars_since_high,
            G("Fractals", p, "Bars_Since_Last_Low"):   bars_since_low,
        }

        return FeatureResult(data=data)
