import logging
import os
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.INFO):
    """Configures a standardized logger for the project."""
    formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers if the logger is retrieved multiple times
    if not logger.handlers:
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler (Only if log_file is provided)
        if log_file:
            try:
                os.makedirs(os.path.dirname(log_file), exist_ok=True)
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
            except Exception as e:
                print(f"Warning: Could not initialize file logging for {name}: {e}")

    return logger

# Global default logger - Console Only
logger = setup_logger("model-engine")

# Specialized Daemon Logger - Console + logs/daemon.log
# Used for detailed tracking of jobs, scheduling, and error traces
daemon_logger = setup_logger(
    "model-engine.daemon", 
    log_file=os.path.join("logs", "daemon.log")
)
