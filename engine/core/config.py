import os
from dotenv import load_dotenv

# Repo root is two directories up from this file (engine/core/config.py)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ENGINE_ROOT = os.path.join(_REPO_ROOT, "engine")

load_dotenv(os.path.join(_ENGINE_ROOT, ".env"))

class Config:
    """Central configuration for the Research Engine."""

    REPO_ROOT = os.getenv("REPO_ROOT", _REPO_ROOT)
    ENGINE_ROOT = os.getenv("ENGINE_ROOT", _ENGINE_ROOT)

    # API Settings - Default to localhost for single-machine setups
    # Can be overridden via environment variables for WSL2 or remote setups
    API_HOST = os.getenv("ENGINE_API_HOST", "127.0.0.1")
    API_PORT = int(os.getenv("ENGINE_API_PORT", 8000))

    # Redis Message Broker Settings
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))

    FRED_API_KEY = os.getenv("FRED_API_KEY")

    STRATEGIES_FOLDER = os.getenv(
        "STRATEGIES_FOLDER",
        os.path.join(REPO_ROOT, "strategies")
    )
    DATA_DIR = os.getenv("DATA_DIR", os.path.join(REPO_ROOT, "data"))
    DB_PATH = os.getenv("DB_PATH", os.path.join(DATA_DIR, "stocks.db"))
    LOG_DIR = os.getenv("LOG_DIR", os.path.join(ENGINE_ROOT, "logs"))
    
    @property
    def api_url(self):
        """Returns the full base URL for the API."""
        return f"http://{self.API_HOST}:{self.API_PORT}"

    @property
    def redis_url(self):
        """Returns the fully qualified Redis connection string."""
        # We check for a raw REDIS_URL first (common in production environments like Heroku/Render)
        # If not found, we construct it from the individual host/port/db components.
        default_url = f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return os.getenv("REDIS_URL", default_url)

# Global config instance
config = Config()
