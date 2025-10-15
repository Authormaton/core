

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

    def format(self, record: logging.LogRecord) -> str:
        base: Dict[str, Any] = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "pid": self.pid,
            "host": self.hostname,
        }
        if record.exc_info:
            base["exc_text"] = self.formatException(record.exc_info)
        base.update(_log_context.get())
        excluded = {
            "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
            "module", "exc_info", "stack_info", "lineno", "funcName", "created",
            "msecs", "relativeCreated", "thread", "threadName", "processName", "process"
        }
        for k, v in record.__dict__.items():
            if k in excluded:
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
        # Create a fresh dict for 'extra' to avoid mutating caller-supplied kwargs
        new_extra = dict(kwargs.get("extra", {}))
        new_extra.update({k: v for k, v in self.extra.items() if k not in new_extra})
        new_extra.update({k: v for k, v in _log_context.get().items() if k not in new_extra})
        kwargs["extra"] = new_extra
        return msg, kwargs


def set_log_context(**kwargs: Any) -> None:
    ctx = dict(_log_context.get())
    ctx.update(kwargs)
    _log_context.set(ctx)


def get_log_context() -> Dict[str, Any]:
    return dict(_log_context.get())


def clear_log_context() -> None:
    _log_context.set({})


def get_logger(name: Optional[str] = None, **adapter_extra: Any) -> RequestLoggerAdapter:
    """Return a RequestLoggerAdapter for `name` (or module logger) preloaded with adapter_extra."""
    logger = logging.getLogger(name)
    return RequestLoggerAdapter(logger, adapter_extra)


# Call setup_logging when the module is imported to ensure logging is configured
setup_logging()
