from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable


class JsonFormatter(logging.Formatter):
    """Format log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        data = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", ""),
        }
        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(data)


def _configure(logger: logging.Logger, json_mode: bool) -> None:
    level = logging.INFO
    logger.setLevel(level)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    formatter = (
        JsonFormatter()
        if json_mode
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False


class LoggerRegistry:
    """Registry that lazily configures and returns loggers."""

    def __init__(self, json: bool = False) -> None:
        self._json = json
        self._loggers: dict[str, logging.Logger] = {}

    def get_logger(
        self, name: str, *pos: object, request_id: str | None = None
    ) -> logging.LoggerAdapter:
        if pos:
            raise TypeError(
                "request_id must be passed as a keyword argument: "
                "get_logger(name, request_id=...)"
            )
        logger = self._loggers.get(name)
        if logger is None:
            logger = logging.getLogger(name)
            _configure(logger, self._json)
            self._loggers[name] = logger
        rid = request_id or str(uuid.uuid4())
        return logging.LoggerAdapter(logger, {"request_id": rid})


LoggerFactory = Callable[[str, str | None], logging.LoggerAdapter]

__all__: list[str] = ["LoggerRegistry", "LoggerFactory"]
