import logging
from typing import Dict, Any
from .config_loader import ConfigManager


def setup_logging() -> logging.Logger:
    """
    Configure the logging system based on settings in the configuration file.

    Returns:
        A configured logger instance
    """
    # Get logging configuration
    log_config = ConfigManager.get_config("logging")

    # Set up basic configuration
    logging.basicConfig(
        level=getattr(logging, log_config.get("level", "INFO")),
        format=log_config.get("format", "%(asctime)s [%(levelname)s] %(name)s: %(message)s"),
        datefmt=log_config.get("date_format", "%Y-%m-%d %H:%M:%S")
    )

    # Get logger for the application
    logger = logging.getLogger("pdf_processor")
    return logger