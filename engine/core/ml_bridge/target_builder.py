import pandas as pd
import numpy as np
from typing import Tuple

class TargetBuilder:
    
    @staticmethod
    def create_target(df: pd.DataFrame, target_col: str = 'Close', lookforward: int = 1, mode: str = 'return') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Safely generates future targets for Machine Learning without lookahead bias.
        
        This utility shifts the target column backward in time to align future 
        price action with current-row features. It strictly drops the final 
        `lookforward` rows to prevent training on incomplete (NaN) targets.

        Args:
            df (pd.DataFrame): The feature-complete dataset.
            target_col (str, optional): The column to calculate targets against. 
                Defaults to 'Close'.
            lookforward (int, optional): The number of periods into the future 
                to forecast. Defaults to 1.
            mode (str, optional): The type of target to generate. 
                Options are 'return' (continuous fractional return) or 
                'binary' (1 for up, 0 for down). Defaults to 'return'.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: 
                - df (pd.DataFrame): The original dataframe truncated to remove NaN targets.
                - y (pd.Series): The generated target series, perfectly indexed to `df`.

        Raises:
            ValueError: If the `target_col` does not exist in the dataframe.
            ValueError: If an unsupported `mode` is provided.

        Example:
            >>> df_clean, y = TargetBuilder.create_target(df, 'Close', lookforward=5, mode='binary')
            >>> model.fit(df_clean[features], y)
        """
        if target_col not in df.columns:
            raise ValueError(f"Target column '{target_col}' not found in DataFrame.")

        # Calculate future state
        if mode == 'return':
            # (Future Price - Current Price) / Current Price
            target_series = (df[target_col].shift(-lookforward) - df[target_col]) / df[target_col]
        elif mode == 'binary':
            # 1 if Future Price > Current Price, else 0
            target_series = (df[target_col].shift(-lookforward) > df[target_col]).astype(int)
        else:
            raise ValueError(f"Unsupported target mode: {mode}. Use 'return' or 'binary'.")

        # Drop the "lookforward" tail where targets are inherently NaN
        df_truncated = df.iloc[:-lookforward].copy()
        y_truncated = target_series.iloc[:-lookforward].copy()

        return df_truncated, y_truncated
