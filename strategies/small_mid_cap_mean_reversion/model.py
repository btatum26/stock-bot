import pandas as pd
from context import Context
from engine.core.controller import SignalModel


class SmallMidCapMeanReversion(SignalModel):

    def generate_signals(self, df: pd.DataFrame, context: Context, params: dict, artifacts: dict,
                         regime_context=None) -> pd.Series:
        rsi2 = df[context.features.RSI_2]
        close = df['close'] if 'close' in df.columns else df['Close']
        sma200 = df[context.features.SMA_200]
        atr5 = df[context.features.AVERAGETRUERANGE_5]
        atr21 = df[context.features.AVERAGETRUERANGE_21]

        # Regime gate: require state 1 (normal ranging) probability > threshold
        # and no structural break detected by BOCPD (novelty <= 0.5 per bar)
        if regime_context is not None:
            ranging = regime_context.proba[1]
            regime_ok = (ranging > params['regime_prob_threshold']) & (regime_context.novelty <= 0.5)
        else:
            regime_ok = pd.Series(True, index=df.index)

        # Entry: deeply oversold + long-term uptrend + stable volatility + benign regime
        oversold = rsi2 < params['rsi_entry']
        uptrend = close > sma200
        vol_stable = (atr5 / atr21.replace(0, float('nan'))) < params['atr_ratio_max']
        entry_trigger = oversold & uptrend & vol_stable & regime_ok

        signal = pd.Series(0.0, index=df.index)
        in_position = False
        entry_price = 0.0
        bars_held = 0

        for i in range(len(df)):
            if in_position:
                bars_held += 1
                current_return = (close.iloc[i] - entry_price) / entry_price

                if (rsi2.iloc[i] > params['rsi_exit']
                        or bars_held >= params['max_hold_days']
                        or current_return < -params['stop_loss']):
                    in_position = False
                else:
                    signal.iloc[i] = 1.0
            else:
                if entry_trigger.iloc[i]:
                    in_position = True
                    entry_price = close.iloc[i]
                    bars_held = 0
                    signal.iloc[i] = 1.0

        return signal
