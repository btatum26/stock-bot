# Tearsheet has moved to engine.core.backtester.
# This shim exists so existing imports continue to resolve without change.
from .backtester import Tearsheet

__all__ = ["Tearsheet"]
