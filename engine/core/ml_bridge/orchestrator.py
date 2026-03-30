import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, Any, Tuple, List
from ..logger import logger

class MLBridge:
    """
    Acts as the strict pre-processing boundary between Phase 2 features and Phase 3 models.
    
    This class ensures that Machine Learning feature matrices are correctly purged of 
    NaN values caused by rolling indicator windows, and enforces strictly stateful 
    dataset scaling to prevent lookahead bias during inference.
    """

    @staticmethod
    def prepare_training_matrix(
        df: pd.DataFrame, 
        feature_cols: List[str], 
        l_max: int
    ) -> Tuple[pd.DataFrame, Any]:
        """
        Prepares the historical feature matrix for model training.

        This method performs two critical operations:
        1. Purges the first `l_max` rows where rolling features are incomplete (NaN).
        2. Fits a global MinMaxScaler strictly on the defined feature columns, bounding 
           the data between [-1.0, 1.0].

        Args:
            df (pd.DataFrame): The raw historical dataset outputted by the FeatureOrchestrator.
            feature_cols (List[str]): A definitive list of column names that will be fed 
                into the model as features (X).
            l_max (int): The maximum lookback window discovered across all computed 
                features (e.g., if a 200-SMA is used, l_max is 200).

        Returns:
            Tuple[pd.DataFrame, Any]: 
                - df_clean (pd.DataFrame): The truncated and scaled dataset ready for `SignalModel.train()`.
                - scaler (Any): The fitted scikit-learn scaler. This MUST be saved via the ArtifactManager.

        Raises:
            ValueError: If `feature_cols` contains columns not present in the DataFrame.
            ValueError: If `l_max` is greater than or equal to the length of the DataFrame.

        Example:
            >>> df_train, scaler = MLBridge.prepare_training_matrix(df, ['RSI_14', 'MACD'], 200)
            >>> artifacts['system_scaler'] = scaler
        """
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns in DataFrame: {missing_cols}")
            
        if l_max >= len(df):
            raise ValueError(f"l_max ({l_max}) is larger than the dataset length ({len(df)}).")

        # 1. NaN Purge: Drop the top l_max rows safely
        df_clean = df.iloc[l_max:].copy()

        # 2. Stateful Scaling: Fit and transform
        scaler = MinMaxScaler(feature_range=(-1.0, 1.0))
        df_clean[feature_cols] = scaler.fit_transform(df_clean[feature_cols])

        logger.info(f"Training matrix prepared. Purged {l_max} rows. Scaler fitted.")
        return df_clean, scaler

    @staticmethod
    def prepare_inference_matrix(
        df: pd.DataFrame, 
        feature_cols: List[str], 
        l_max: int, 
        artifacts: Dict[str, Any],
        is_live: bool = False
    ) -> pd.DataFrame:
        """
        Prepares the feature matrix for backtesting or live execution.

        Crucially, this method NEVER fits a new scaler. It transforms the incoming 
        data using the exact scaler fitted during `ExecutionMode.TRAIN`. If no scaler 
        is found in the artifacts (e.g., a rule-based strategy), it skips scaling.

        Args:
            df (pd.DataFrame): The dataset outputted by the FeatureOrchestrator.
            feature_cols (List[str]): The exact list of feature columns used during training.
            l_max (int): The maximum lookback window required to compute current features.
            artifacts (Dict[str, Any]): The loaded artifact dictionary.
            is_live (bool, optional): If True, aggressively truncates the DataFrame to 
                only the single most recent row to optimize live execution latency. 
                Defaults to False (Backtest mode).

        Returns:
            pd.DataFrame: The pre-processed dataframe ready to be passed into the 
                user's `generate_signals()` method.

        Raises:
            KeyError: If a scaler is expected but not found in the artifacts.
        """
        # 1. Row Selection & Purging
        if is_live:
            # For live execution, we only need the absolute latest row (after ensuring features computed)
            df_clean = df.iloc[[-1]].copy()
        else:
            # For backtesting, we purge the top l_max rows just like in training
            df_clean = df.iloc[l_max:].copy()

        # 2. Extract and Apply Scaler
        scaler = artifacts.get("system_scaler")
        
        if scaler:
            # Strict TRANSFORM only. Never fit.
            df_clean[feature_cols] = scaler.transform(df_clean[feature_cols])
            logger.debug("Applied loaded scaler to inference matrix.")
        else:
            logger.debug("No system_scaler found in artifacts. Passing unscaled data.")

        return df_clean