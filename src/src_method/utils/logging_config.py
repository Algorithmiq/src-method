"""Structured logging configuration."""

from __future__ import annotations

import logging
import os
import sys

import structlog


def setup_logging(default_level: str = "INFO") -> None:
    """Configure logging for the entire application.

    Args:
        default_level: Default logging level.

    The logging level can be overridden by setting the `LOG_LEVEL_SRC`
    environment. For example:

    ```bash
    LOG_LEVEL_SRC=DEBUG python -m <your_module>
    pytest -o log_cli_level=DEBUG tests/
    ```
    """
    # Configure Python's standard logging
    level_name = os.environ.get("LOG_LEVEL_SRC", default_level)
    level = logging.getLevelName(level_name.upper())
    logging.basicConfig(
        level=level,
        format="%(message)s",
        stream=sys.stdout,
    )

    # Remove colors for slurm logs but not for pytest
    use_colors = "SLURM_JOB_ID" not in os.environ

    # Configure structlog to use the standard logger
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            structlog.dev.ConsoleRenderer(colors=use_colors),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
