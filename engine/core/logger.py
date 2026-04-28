import logging
import logging.handlers
import os

from .config import config

# Anchor log dir through engine config so Docker, CLI, and GUI all agree.
_LOG_DIR = os.path.normpath(config.LOG_DIR)

_FMT = "[%(asctime)s] [%(name)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"
_formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)


def _rotating(filename: str, level: int) -> logging.Handler:
    os.makedirs(_LOG_DIR, exist_ok=True)
    h = logging.handlers.RotatingFileHandler(
        os.path.join(_LOG_DIR, filename),
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    h.setLevel(level)
    h.setFormatter(_formatter)
    return h


def _console() -> logging.Handler:
    h = logging.StreamHandler()
    h.setLevel(logging.INFO)
    h.setFormatter(_formatter)
    return h


def _configure():
    root = logging.getLogger("model-engine")
    if root.handlers:
        return  # already configured (tests, re-import, etc.)
    root.setLevel(logging.DEBUG)
    root.propagate = False  # don't leak to the Python root logger
    root.addHandler(_console())
    root.addHandler(_rotating("engine.log", logging.DEBUG))
    root.addHandler(_rotating("errors.log", logging.WARNING))


_configure()

# Public loggers - import these instead of calling getLogger() directly.
# Child loggers inherit all handlers from model-engine automatically.
logger = logging.getLogger("model-engine")
daemon_logger = logging.getLogger("model-engine.daemon")
data_logger = logging.getLogger("model-engine.data")
