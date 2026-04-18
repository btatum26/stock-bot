import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Central configuration for the Research Engine."""
    
    # API Settings - Default to localhost for single-machine setups
    # Can be overridden via environment variables for WSL2 or remote setups
    API_HOST = os.getenv("ENGINE_API_HOST", "127.0.0.1")
    API_PORT = int(os.getenv("ENGINE_API_PORT", 8000))
    
    # Redis Message Broker Settings
    REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB = int(os.getenv("REDIS_DB", 0))
    
    FRED_API_KEY = os.getenv("FRED_API_KEY")
    
    STRATEGIES_FOLDER = os.getenv("STRATEGIES_FOLDER", "./strategies")
    
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