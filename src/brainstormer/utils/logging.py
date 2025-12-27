"""Logging configuration for Brainstormer."""

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure logging with rich output."""
    handlers: list[logging.Handler] = [
        RichHandler(
            console=Console(stderr=True),
            show_time=True,
            show_path=False,
            rich_tracebacks=True,
        )
    ]

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )

    # Reduce noise from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("langchain").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)
