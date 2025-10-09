"""Structured logging helpers for JSON-formatted application logs."""
from __future__ import annotations

import json
import logging
import logging.config
import os
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional

from utils.logging_utils import install_sensitive_filter


TRACE_ID_VAR: ContextVar[str] = ContextVar("ai_trader_trace_id", default="-")


class TraceIdFilter(logging.Filter):
    """Inject the active trace ID from the contextvar into log records."""

    def filter(self, record: logging.LogRecord) -> bool:  # pragma: no cover - logging API
        record.trace_id = TRACE_ID_VAR.get("-") or "-"
        return True


class JsonLogFormatter(logging.Formatter):
    """Serialise log records as structured JSON payloads."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - logging API
        timestamp = datetime.fromtimestamp(record.created, timezone.utc).isoformat()
        base: Dict[str, Any] = {
            "timestamp": timestamp,
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": getattr(record, "trace_id", TRACE_ID_VAR.get("-")) or "-",
            "module": record.module,
            "line": record.lineno,
            "process": record.process,
            "thread": record.threadName,
        }

        if record.exc_info:
            base["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            base["stack"] = self.formatStack(record.stack_info)

        extras: Dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "trace_id",
            }:
                continue
            try:
                json.dumps(value)
                extras[key] = value
            except TypeError:
                extras[key] = repr(value)

        if extras:
            base["context"] = extras

        return json.dumps(base, ensure_ascii=False)


def configure_structured_logging(*, level: Optional[str] = None) -> None:
    """Configure the root logger with JSON formatting and trace propagation."""

    desired_level = str(level or os.getenv("LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, desired_level, logging.INFO)
    log_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "trace": {
                "()": "utils.structured_logging.TraceIdFilter",
            }
        },
        "formatters": {
            "json": {
                "()": "utils.structured_logging.JsonLogFormatter",
            }
        },
        "handlers": {
            "default": {
                "class": "logging.StreamHandler",
                "level": numeric_level,
                "filters": ["trace"],
                "formatter": "json",
            }
        },
        "root": {
            "level": numeric_level,
            "handlers": ["default"],
        },
    }

    logging.config.dictConfig(log_config)
    install_sensitive_filter(logging.getLogger())


def get_logger(name: str, *, mask_fields: Iterable[str] = ()) -> logging.Logger:
    """Return a logger that is configured for JSON output and masking."""

    logger = logging.getLogger(name)
    install_sensitive_filter(logger, fields=mask_fields)
    return logger


def set_trace_id(trace_id: str):  # pragma: no cover - thin wrapper
    return TRACE_ID_VAR.set(trace_id)


def reset_trace_id(token) -> None:  # pragma: no cover - thin wrapper
    TRACE_ID_VAR.reset(token)


def current_trace_id() -> str:
    return TRACE_ID_VAR.get("-") or "-"
