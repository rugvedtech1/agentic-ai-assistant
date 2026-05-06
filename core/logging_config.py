import logging
import os
from datetime import datetime

def setup_logging(log_level: str = "INFO", log_file: str = "logs/app.log"):
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Log format - every line shows time, level, and message
    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        datefmt=date_format,
        handlers=[
            # Handler 1: Write logs to file
            logging.FileHandler(log_file),
            # Handler 2: Also show logs in terminal
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return logger

# Single logger instance for the whole app
logger = setup_logging()