class EngineError(Exception):
    """Base exception for all model-engine errors."""
    pass

class DataError(EngineError):
    """Raised when data fetching or processing fails."""
    pass

class StrategyError(EngineError):
    """Raised when strategy loading or execution fails."""
    pass

class FeatureError(EngineError):
    """Raised when feature computation fails."""
    pass

class ValidationError(EngineError):
    """Raised when input validation fails."""
    pass
