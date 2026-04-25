import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict

from context import Context
from engine.core.controller import SignalModel


class OptionsImpliedCrossSectional(SignalModel):
    """
    Cross-sectional signal from options market microstructure.

    Ranks S&P 500 stocks by three options-derived components, z-scored
    across the universe, then goes long the highest-ranked stocks.

    Components (each z-scored cross-sectionally, equal weight):
      1. Put/call volume ratio      — high PCR = bearish positioning
      2. IV skew (OTM put - ATM)    — high skew = downside fear priced in
      3. Total options volume       — proxy for informed-trader attention

    Composite = -(pcr_z + skew_z + vol_z) / 3  (flipped: high = bullish)

    Data note: yfinance provides only a live snapshot of each options chain,
    not historical data. For historical backtests this strategy returns zero
    signals for all bars except the last (the live snapshot date). Meaningful
    performance evaluation requires a historical options data source
    (e.g. EODHD, CBOE data shop, or a self-collected forward-going dataset).
    """

    def train(self, data: Dict[str, pd.DataFrame], context: Context, params: dict) -> dict:
        return {}

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        context: Context,
        params: dict,
        artifacts: dict,
    ) -> Dict[str, pd.Series]:
        long_only = params.get('long_only', True)

        # --- Step 1: fetch a live options snapshot for every ticker ---
        pcr_raw: Dict[str, float] = {}
        skew_raw: Dict[str, float] = {}
        vol_raw: Dict[str, float] = {}

        for ticker, df in data.items():
            try:
                import yfinance as yf
                t = yf.Ticker(ticker)
                expirations = t.options
                if not expirations:
                    raise ValueError("no options expirations available")

                spot = float(df['close'].iloc[-1])

                # Nearest expiry to 30 days out (most liquid for IV skew)
                target_date = datetime.today() + timedelta(days=30)
                best_exp = min(
                    expirations,
                    key=lambda d: abs(
                        (datetime.strptime(d, '%Y-%m-%d') - target_date).days
                    ),
                )

                chain = t.option_chain(best_exp)
                calls = chain.calls.copy()
                puts = chain.puts.copy()

                calls['volume'] = calls['volume'].fillna(0)
                puts['volume'] = puts['volume'].fillna(0)

                # Component 1: put/call volume ratio
                call_vol = float(calls['volume'].sum())
                put_vol = float(puts['volume'].sum())
                pcr_raw[ticker] = put_vol / (call_vol + 1.0)

                # Component 2: IV skew — OTM put IV minus ATM put IV
                atm_puts = puts[
                    (puts['strike'] >= spot * 0.98) & (puts['strike'] <= spot * 1.02)
                ]
                otm_puts = puts[
                    (puts['strike'] >= spot * 0.85) & (puts['strike'] < spot * 0.95)
                ]
                atm_iv = atm_puts['impliedVolatility'].replace(0, np.nan).mean()
                otm_iv = otm_puts['impliedVolatility'].replace(0, np.nan).mean()

                if pd.notna(atm_iv) and pd.notna(otm_iv):
                    skew_raw[ticker] = float(otm_iv - atm_iv)
                else:
                    skew_raw[ticker] = np.nan

                # Component 3: total options volume (cross-sectional proxy for
                # unusual activity — no historical baseline available)
                vol_raw[ticker] = call_vol + put_vol

            except Exception:
                pcr_raw[ticker] = np.nan
                skew_raw[ticker] = np.nan
                vol_raw[ticker] = np.nan

        # --- Step 2: cross-sectional z-score each component ---
        tickers = list(data.keys())

        def cs_zscore(score_dict: Dict[str, float]) -> np.ndarray:
            vals = np.array([score_dict.get(t, np.nan) for t in tickers], dtype=float)
            valid = ~np.isnan(vals)
            result = np.zeros(len(tickers), dtype=float)
            if valid.sum() >= 2:
                result[valid] = (
                    (vals[valid] - vals[valid].mean()) / (vals[valid].std() + 1e-9)
                )
            return result

        pcr_z = cs_zscore(pcr_raw)
        skew_z = cs_zscore(skew_raw)
        vol_z = cs_zscore(vol_raw)

        # Equal-weight composite; flip sign: high bearish options → negative score
        composite = (pcr_z + skew_z + vol_z) / 3.0

        # --- Step 3: build output signals ---
        # Signal is stamped at the last bar only (live snapshot); all historical
        # bars are zero because we have no historical options data.
        results: Dict[str, pd.Series] = {}
        for i, ticker in enumerate(tickers):
            df = data[ticker]
            score = float(np.clip(-composite[i], -1.0, 1.0))
            if long_only and score < 0.0:
                score = 0.0

            # Optional: require price above 50-day SMA as an additional long filter
            sma_col = context.features.SMA_50
            if long_only and sma_col in df.columns:
                if float(df['close'].iloc[-1]) < float(df[sma_col].iloc[-1]):
                    score = 0.0

            sig = pd.Series(0.0, index=df.index)
            sig.iloc[-1] = score
            results[ticker] = sig

        return results
