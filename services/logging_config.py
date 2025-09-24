"""
Structured logging setup for Authormaton core engine.
"""

import logging
import json
import sys

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            "level": record.levelname,
            "time": self.formatTime(record, self.datefmt),
            "name": record.name,
            "message": record.getMessage(),
        }
        # Optionally add extra context if present
        if hasattr(record, "request_id"):
            log_record["request_id"] = record.request_id
        if hasattr(record, "user"):
            log_record["user"] = record.user
        if hasattr(record, "document_id"):
            log_record["document_id"] = record.document_id
        return json.dumps(log_record)


def setup_logging(level=logging.INFO):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers = [handler]

# Call setup_logging() at app startup (e.g., in main.py or api/__init__.py)
