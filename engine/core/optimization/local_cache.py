import os
import pandas as pd

from ..logger import logger

SHM_PATH = "/dev/shm/active_job.parquet"
if not os.path.exists("/dev/shm"):
    SHM_PATH = os.path.join(os.getcwd(), "transit", "active_job.parquet")
    os.makedirs(os.path.dirname(SHM_PATH), exist_ok=True)

def stage_data_to_shm(df: pd.DataFrame, symbol: str, timeframe: str):
    """
    Reads from SQLite ONCE and stages to RAM-disk to prevent 
    Joblib from copying massive DataFrames to every worker.
    """
    df.to_parquet(SHM_PATH, engine='pyarrow')
    logger.info(f"Data staged to memory at {SHM_PATH}")
    return SHM_PATH

def load_data_from_shm() -> pd.DataFrame:
    """Worker function to read data locally without IPC serialization overhead."""
    if not os.path.exists(SHM_PATH):
        raise FileNotFoundError(f"Shared memory cache missing at {SHM_PATH}")
    
    return pd.read_parquet(SHM_PATH, engine='pyarrow')

class LocalCache:
    """
    Compatibility wrapper for existing code, adapting to the new RAM-Disk protocol.
    """
    def __init__(self):
        pass

    def load_to_ram(self, dataset_ref: str, df: pd.DataFrame):
        stage_data_to_shm(df, dataset_ref, "")
        return dataset_ref
        
    def get_ref(self, dataset_ref: str):
        return dataset_ref
        
    def clear_cache(self, dataset_ref: str):
        if os.path.exists(SHM_PATH):
            os.remove(SHM_PATH)
