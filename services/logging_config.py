"""
Structured logging setup for Authormaton core engine.
"""

import logging
import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple


class JsonFormatter(logging.Formatter):
    """Formatter that outputs JSON lines with a minimal structured schema.

    It safely serializes record attributes and includes any extra attributes
    supplied via the ``extra`` parameter on logging calls.
    """

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        # Always use UTC ISO-8601 with milliseconds
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
        return dt.isoformat(timespec="milliseconds")

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        log_record: Dict[str, Any] = {
            "level": record.levelname,
            "time": self.formatTime(record, self.datefmt),
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include standard exception info if present
        if record.exc_info:
            # Use the built-in formatter to render exception text but don't
            # include it in the machine-readable payload. Keep a short flag.
            log_record["exc_text"] = self.formatException(record.exc_info)

        # Attach any extra attributes provided via ``extra={...}``
        for key, value in record.__dict__.items():
            if key in ("name", "msg", "args", "levelname", "levelno", "pathname", "filename",
                       "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
                       "created", "msecs", "relativeCreated", "thread", "threadName", "processName",
                       "process"):
                continue
            # Only include JSON-serializable extras; fall back to str()
            try:
                json.dumps({key: value})
                log_record[key] = value
            except Exception:
                log_record[key] = str(value)

        # Ensure deterministic output order for readability in logs
        try:
            return json.dumps(log_record, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            # Fallback to a safe string representation
            return json.dumps({"level": record.levelname, "message": record.getMessage()})


def setup_logging(level: int = logging.INFO, *, force: bool = False) -> None:
    """Configure root logger to emit JSON logs to stdout.

    Args:
        level: Logging level for the root logger.
        force: If True, replace existing handlers. Default False (adds a handler only
            if no stream handler is present).

    Notes:
        Call this once at application startup (for example from `main.py`).
    """

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Avoid adding duplicate stream handlers unless forced
    has_stream = any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers)
    if has_stream and not force:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    if force:
        root_logger.handlers = [handler]
    else:
        root_logger.addHandler(handler)


class RequestLoggerAdapter(logging.LoggerAdapter):
    """Convenience adapter to attach contextual info (like request_id) to logs.

    Usage:
        logger = RequestLoggerAdapter(logging.getLogger(__name__), {"request_id": rid})
        logger.info("started")
    """

    def process(self, msg: str, kwargs: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        extra = kwargs.setdefault("extra", {})
        # Merge adapter's context without overwriting existing keys
        for k, v in (self.extra or {}).items():
            extra.setdefault(k, v)
        return msg, kwargs


# Example: Call `setup_logging()` at app startup (e.g., in main.py or api/__init__.py)

