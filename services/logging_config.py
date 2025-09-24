"""
Improved structured logging setup for Authormaton core engine.

Features:
- JSON formatter with ISO-8601 UTC times and exception support
- Environment-configurable default level and optional "pretty" (indented) output
- Includes process id and hostname automatically
- Context-aware extras via contextvars (set_log_context/get_log_context/clear_log_context)
- Helper to add a RotatingFileHandler
- `get_logger()` convenience returning a RequestLoggerAdapter preloaded with context

Call `setup_logging()` at application startup.
"""

import logging
import json
import os
import socket
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, Optional, Tuple
import contextvars

# Context var used to hold per-request logging context
_log_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar("_log_context", default={})


class JsonFormatter(logging.Formatter):
    """JSON formatter that includes extras and safe serialization."""

    def __init__(self, *, pretty: bool = False, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.pretty = pretty
        self.hostname = socket.gethostname()
        self.pid = os.getpid()

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return dt.isoformat(timespec="milliseconds")

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        base: Dict[str, Any] = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "pid": self.pid,
            "host": self.hostname,
        }

        if record.exc_info:
            # Include textual exception info for debugging
            base["exc_text"] = self.formatException(record.exc_info)

        # Merge contextvar extras (per-request) first
        base.update(_log_context.get({}))

        # Merge any extras passed to logging calls
        for k, v in record.__dict__.items():
            if k in ("name", "msg", "args", "levelname", "levelno", "pathname", "filename",
                     "module", "exc_info", "stack_info", "lineno", "funcName", "created",
                     "msecs", "relativeCreated", "thread", "threadName", "processName", "process"):
                continue
            try:
                json.dumps({k: v})
                base[k] = v
            except Exception:
                base[k] = str(v)

        if self.pretty:
            return json.dumps(base, ensure_ascii=False, indent=2)
        return json.dumps(base, ensure_ascii=False, separators=(",", ":"))


def setup_logging(
    level: Optional[int] = None,
    *,
    pretty: Optional[bool] = None,
    force: bool = False,
) -> None:
    """Configure root logger.

    Args:
        level: optional logging level (int). If None, read from LOG_LEVEL env var or default to INFO.
        pretty: when True output indented JSON. If None, read LOG_PRETTY env var (1/true) or default False.
        force: if True, remove existing StreamHandler(s) and add the stdout JSON handler.
    """

    env_level = os.getenv("LOG_LEVEL")
    if level is None:
        if env_level:
            try:
                level = int(env_level)
            except Exception:
                level = getattr(logging, env_level.upper(), logging.INFO)
        else:
            level = logging.INFO

    if pretty is None:
        pretty_env = os.getenv("LOG_PRETTY", "0").lower()
        pretty = pretty_env in ("1", "true", "yes")

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Find any existing stdout StreamHandler(s)
    stdout_handlers = [h for h in root_logger.handlers
                       if isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout]

    if stdout_handlers and not force:
        # nothing to do
        return

    if force:
        # remove only StreamHandler(s) to preserve file/syslog handlers
        root_logger.handlers = [h for h in root_logger.handlers if not isinstance(h, logging.StreamHandler)]

    # Add stdout JSON handler if not already present
    stdout_present = any(isinstance(h, logging.StreamHandler) and getattr(h, "stream", None) is sys.stdout
                         for h in root_logger.handlers)
    if not stdout_present:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter(pretty=pretty))
        root_logger.addHandler(handler)


def add_rotating_file_handler(path: str, max_bytes: int = 10_000_000, backup_count: int = 3, *, level: Optional[int] = None,
                              pretty: bool = False) -> RotatingFileHandler:
    """Add a RotatingFileHandler to the root logger and return it."""
    handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
    handler.setFormatter(JsonFormatter(pretty=pretty))
    if level is None:
        level = logging.getLogger().level
    handler.setLevel(level)
    logging.getLogger().addHandler(handler)
    return handler


class RequestLoggerAdapter(logging.LoggerAdapter):
    """Attach contextual extras to logging records.

    Adapter merges adapter.extra, contextvar extras, and any kwargs['extra'] passed to log calls.
    """

    def process(self, msg: str, kwargs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        extra = kwargs.setdefault("extra", {})
        # merge adapter extras without overwriting
        for k, v in (self.extra or {}).items():
            extra.setdefault(k, v)
        # merge contextvar extras
        for k, v in _log_context.get({}).items():
            extra.setdefault(k, v)
        return msg, kwargs


def set_log_context(**kwargs: Any) -> None:
    ctx = dict(_log_context.get({}))
    ctx.update(kwargs)
    _log_context.set(ctx)


def get_log_context() -> Dict[str, Any]:
    return dict(_log_context.get({}))


def clear_log_context() -> None:
    _log_context.set({})


def get_logger(name: Optional[str] = None, **adapter_extra: Any) -> RequestLoggerAdapter:
    """Return a RequestLoggerAdapter for `name` (or module logger) preloaded with adapter_extra."""
    logger = logging.getLogger(name)
    return RequestLoggerAdapter(logger, adapter_extra)


# Backwards compatibility: simple setup call
if __name__ == "__main__":
    setup_logging()
