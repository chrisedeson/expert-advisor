"""
Logging configuration for Expert Advisor
Uses loguru for structured, colorized logging
"""

import sys
from pathlib import Path
from loguru import logger

from .config import get_config


def setup_logger():
    """Configure loguru logger with file and console outputs"""
    config = get_config()

    # Remove default logger
    logger.remove()

    # Console output with colors
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=config.log_level,
        colorize=True,
    )

    # File output
    log_file = config.logs_dir / "expert_advisor.log"
    logger.add(
        log_file,
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=config.log_level,
    )

    logger.info("Logger initialized")
    logger.info(f"Log level: {config.log_level}")
    logger.info(f"Log file: {log_file}")

    return logger


# Initialize logger on module import
setup_logger()
