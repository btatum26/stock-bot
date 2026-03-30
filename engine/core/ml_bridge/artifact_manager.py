import os
import joblib
from typing import Dict, Any
from ..logger import logger

class ArtifactManager:
    """
    Handles the persistent storage and retrieval of strategy-specific ML artifacts.
    
    This manager provides a 'magic' state-handling mechanism. It serializes the 
    dictionary returned by a user's `train()` method and seamlessly injects it 
    back into their `generate_signals()` method during backtesting or live execution.
    """

    FILENAME = "artifacts.joblib"

    @staticmethod
    def save_artifacts(strategy_dir: str, artifacts: Dict[str, Any]) -> None:
        """
        Serializes and saves a dictionary of ML artifacts to the strategy directory.

        This method should be called by the ApplicationController immediately 
        after a successful `ExecutionMode.TRAIN` run. If the user's `train` 
        method returns an empty dictionary (e.g., for rule-based strategies), 
        no file is created.

        Args:
            strategy_dir (str): The absolute or relative path to the specific 
                strategy's directory (e.g., 'src/strategies/MyXGBoostStrat').
            artifacts (Dict[str, Any]): A dictionary containing fitted models, 
                scalers, or custom state variables generated during training.

        Raises:
            IOError: If the directory does not exist or the system lacks write permissions.
            TypeError: If the artifacts dictionary contains objects that cannot be 
                pickled/serialized by joblib.

        Example:
            >>> my_models = {"xgb_main": fitted_xgb, "scaler": fitted_minmax}
            >>> ArtifactManager.save_artifacts("src/strategies/StratA", my_models)
        """
        if not artifacts:
            logger.info(f"No artifacts returned for {strategy_dir}. Skipping save.")
            return

        os.makedirs(strategy_dir, exist_ok=True)
        file_path = os.path.join(strategy_dir, ArtifactManager.FILENAME)
        
        try:
            joblib.dump(artifacts, file_path)
            logger.info(f"Successfully saved {len(artifacts)} artifacts to {file_path}")
        except Exception as e:
            logger.error(f"Failed to serialize artifacts to {file_path}: {e}")
            raise IOError(f"Artifact serialization failed: {e}")

    @staticmethod
    def load_artifacts(strategy_dir: str) -> Dict[str, Any]:
        """
        Retrieves and deserializes ML artifacts from the strategy directory.

        This method is called by the backtester or live execution engine prior to 
        invoking the user's `generate_signals()` method. If no artifact file exists 
        (common for purely technical/rule-based strategies), it gracefully returns 
        an empty dictionary.

        Args:
            strategy_dir (str): The path to the specific strategy's directory.

        Returns:
            Dict[str, Any]: The loaded dictionary of models and scalers. 
                Returns `{}` if the artifact file is not found.

        Raises:
            IOError: If the file exists but is corrupted or cannot be read.

        Example:
            >>> artifacts = ArtifactManager.load_artifacts("src/strategies/StratA")
            >>> model = artifacts.get("xgb_main")
        """
        file_path = os.path.join(strategy_dir, ArtifactManager.FILENAME)
        
        if not os.path.exists(file_path):
            logger.debug(f"No artifact file found at {file_path}. Returning empty dict.")
            return {}

        try:
            artifacts = joblib.load(file_path)
            logger.info(f"Successfully loaded artifacts from {file_path}")
            return artifacts
        except Exception as e:
            logger.error(f"Failed to load artifacts from {file_path}: {e}")
            raise IOError(f"Artifact deserialization failed: {e}")