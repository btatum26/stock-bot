from typing import Dict, Any, List
import pandas as pd
from ..base import Feature, FeatureResult, OutputSchema, OutputType, Pane, register_feature

@register_feature("BollingerBands")
class BollingerBands(Feature):
    @property
    def name(self) -> str:
        return "Bollinger Bands"

    @property
    def description(self) -> str:
        return "Volatility bands based on Standard Deviation. Supports Bollinger Width and systematic normalization."

    @property
    def category(self) -> str:
        return "Volatility"

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "period": 20,
            "std_dev": 2.0,
            "normalize": "none"
        }

    @property
    def output_schema(self) -> List[OutputSchema]:
        return [
            OutputSchema(name="upper", output_type=OutputType.LINE, pane=Pane.OVERLAY),
            OutputSchema(name="mid",   output_type=OutputType.LINE, pane=Pane.OVERLAY),
            OutputSchema(name="lower", output_type=OutputType.LINE, pane=Pane.OVERLAY),
            OutputSchema(name="width", output_type=OutputType.LINE, pane=Pane.NEW),
            OutputSchema(name="fill",  output_type=OutputType.BAND, pane=Pane.OVERLAY,
                         band_pair=("upper", "lower")),
        ]

    def compute(self, df: pd.DataFrame, params: Dict[str, Any], cache: Any = None) -> FeatureResult:
        period = int(params.get("period", 20))
        std_dev = float(params.get("std_dev", 2.0))
        norm_method = params.get("normalize", "none")

        # Standardize OHLCV access
        close = df['Close'] if 'Close' in df.columns else df['close']

        # Calculate Bollinger Bands math using Cache for SMA
        if cache:
            mid_band = cache.get_series("SMA", {"period": period}, df)
        else:
            mid_band = close.rolling(window=period).mean()

        rolling_std = close.rolling(window=period).std()

        upper_band = mid_band + (rolling_std * std_dev)
        lower_band = mid_band - (rolling_std * std_dev)

        # Bollinger Width: (Upper - Lower) / Mid
        # Useful for identifying "squeezes"
        width = (upper_band - lower_band) / mid_band.replace(0, 1e-9)

        col_upper = self.generate_column_name("BollingerBands", params, "upper")
        col_mid = self.generate_column_name("BollingerBands", params, "mid")
        col_lower = self.generate_column_name("BollingerBands", params, "lower")
        col_width = self.generate_column_name("BollingerBands", params, "width")

        # Normalize data based on strategy request
        norm_upper = self.normalize(df, upper_band, norm_method)
        norm_mid = self.normalize(df, mid_band, norm_method)
        norm_lower = self.normalize(df, lower_band, norm_method)
        norm_width = self.normalize(df, width, norm_method)

        data_dict = {
            col_upper: norm_upper,
            col_mid: norm_mid,
            col_lower: norm_lower,
            col_width: norm_width
        }

        return FeatureResult(data=data_dict)
