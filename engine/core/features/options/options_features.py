"""Options-implied features using yfinance live options chains.

IMPORTANT: yfinance only provides current (live) options data — not historical snapshots.
These features return NaN for all historical rows and a live value only at the last row
when the price df ends within 5 calendar days of today.

Intended use: signal generation, not backtesting.
"""

import logging
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd
import yfinance as yf
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature

logger = logging.getLogger("model-engine.features.options")


def _atm_iv(chain: pd.DataFrame, spot: float) -> float:
    if chain.empty:
        return float("nan")
    idx = (chain["strike"] - spot).abs().idxmin()
    return float(chain.loc[idx, "impliedVolatility"])


def _otm_put_iv(puts: pd.DataFrame, spot: float, delta_proxy: float = 0.95) -> float:
    """Approximate 25-delta put IV as the put struck at spot * delta_proxy."""
    if puts.empty:
        return float("nan")
    target = spot * delta_proxy
    idx = (puts["strike"] - target).abs().idxmin()
    return float(puts.loc[idx, "impliedVolatility"])


def _nearest_expiry(expirations: tuple, target_days: int) -> str:
    today = datetime.now()
    return min(
        expirations,
        key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d") - today).days - target_days),
    )


@register_feature("OptionsFlow")
class OptionsFlow(Feature):
    """Options-implied signals: P/C ratio, IV skew, IV term structure, volume unusualness.

    Parameters
    ----------
    ticker : str
        The ticker symbol to fetch options data for (required).
    pcr_window : int
        Look-back days for P/C ratio change signal (default 5).
    unusual_window : int
        Rolling window in days for volume unusualness median (default 20).

    Output columns
    --------------
    pcr          : put/call volume ratio (short-term chain, ~30d expiry)
    pcr_chg5     : 5-day change in pcr
    iv_skew      : OTM put IV minus ATM IV (25-delta proxy)
    iv_ts        : short-term ATM IV divided by medium-term ATM IV (~30d / ~90d)
    vol_unusual  : total options volume divided by its 20-day rolling median
    """

    @property
    def name(self) -> str:
        return "Options Flow"

    @property
    def description(self) -> str:
        return (
            "Live options signals: P/C ratio, IV skew, IV term structure, volume unusualness. "
            "Returns NaN for historical rows — live signal generation only."
        )

    @property
    def category(self) -> str:
        return "Options"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "ticker":         "",
            "pcr_window":     5,
            "unusual_window": 20,
        }

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name="pcr",         output_type=OutputType.LINE, pane=Pane.NEW),
            OutputSchema(name="pcr_chg5",    output_type=OutputType.LINE, pane=Pane.NEW),
            OutputSchema(name="iv_skew",     output_type=OutputType.LINE, pane=Pane.NEW),
            OutputSchema(name="iv_ts",       output_type=OutputType.LINE, pane=Pane.NEW),
            OutputSchema(name="vol_unusual", output_type=OutputType.LINE, pane=Pane.NEW),
        ]

    def _is_live(self, df: pd.DataFrame) -> bool:
        end = df.index.max()
        if hasattr(end, "tz_localize") and end.tzinfo is not None:
            end = end.tz_localize(None)
        return (datetime.now() - end.to_pydatetime()).days <= 5

    def _snapshot(self, ticker_sym: str, spot: float) -> Dict[str, float]:
        t = yf.Ticker(ticker_sym)
        expirations = t.options
        if not expirations:
            return {}

        exp_short  = _nearest_expiry(expirations, 30)
        exp_medium = _nearest_expiry(expirations, 90)

        chain_s = t.option_chain(exp_short)
        chain_m = t.option_chain(exp_medium)

        calls_s, puts_s = chain_s.calls, chain_s.puts
        calls_m         = chain_m.calls

        put_vol  = float(puts_s["volume"].fillna(0).sum())
        call_vol = float(calls_s["volume"].fillna(0).sum())
        total_vol = put_vol + call_vol

        pcr      = put_vol / max(call_vol, 1.0)
        iv_skew  = _otm_put_iv(puts_s, spot) - _atm_iv(puts_s, spot)
        iv_ts    = _atm_iv(calls_s, spot) / max(_atm_iv(calls_m, spot), 1e-9)

        return {
            "pcr":       pcr,
            "iv_skew":   iv_skew,
            "iv_ts":     iv_ts,
            "total_vol": total_vol,
        }

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        P = "OptionsFlow"
        col_pcr     = self.generate_column_name(P, params, "pcr")
        col_chg5    = self.generate_column_name(P, params, "pcr_chg5")
        col_skew    = self.generate_column_name(P, params, "iv_skew")
        col_ts      = self.generate_column_name(P, params, "iv_ts")
        col_unusual = self.generate_column_name(P, params, "vol_unusual")

        nan = pd.Series(float("nan"), index=df.index)
        empty = {col_pcr: nan, col_chg5: nan, col_skew: nan, col_ts: nan, col_unusual: nan}

        ticker_sym = params.get("ticker", "")
        if not ticker_sym:
            logger.warning("OptionsFlow: 'ticker' param is required")
            return FeatureResult(data=empty)

        if not self._is_live(df):
            return FeatureResult(data=empty)

        close_col = "close" if "close" in df.columns else "Close"
        spot = float(df[close_col].iloc[-1])

        try:
            metrics = self._snapshot(ticker_sym, spot)
        except Exception as e:
            logger.warning(f"OptionsFlow snapshot failed for {ticker_sym}: {e}")
            return FeatureResult(data=empty)

        if not metrics:
            return FeatureResult(data=empty)

        last = df.index.max()

        pcr_s   = nan.copy(); pcr_s[last]   = metrics["pcr"]
        skew_s  = nan.copy(); skew_s[last]  = metrics["iv_skew"]
        ts_s    = nan.copy(); ts_s[last]    = metrics["iv_ts"]
        vol_s   = nan.copy(); vol_s[last]   = metrics["total_vol"]

        unusual_window = int(params.get("unusual_window", 20))
        vol_median = vol_s.rolling(unusual_window, min_periods=1).median()
        vol_unusual = vol_s / vol_median.replace(0, float("nan"))

        return FeatureResult(data={
            col_pcr:     pcr_s,
            col_chg5:    pcr_s.diff(5),
            col_skew:    skew_s,
            col_ts:      ts_s,
            col_unusual: vol_unusual,
        })
