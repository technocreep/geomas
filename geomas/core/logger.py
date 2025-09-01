import logging
import logging.handlers
import os
from datetime import datetime
from geomas.core.utils import PROJECT_PATH

_LOG_FILE_NAME = None


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[37m",
        logging.INFO: "\033[36m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;41m",
    }
    RESET = "\033[0m"

    def format(self, record):
        log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_fmt)
        formatted = formatter.format(record)
        color = self.COLORS.get(record.levelno, self.RESET)
        return f"{color}{formatted}{self.RESET}"


def get_logger(
    logger_name: str = "GEOMAS",
    log_dir="geomas_logs",
    level=logging.DEBUG,
    rotation_bytes: int = 5_242_880,
    backup_count: int = 3,
) -> logging.Logger:
    global _LOG_FILE_NAME
    log_dir = PROJECT_PATH + "/" + log_dir
    os.makedirs(log_dir, exist_ok=True)

    if _LOG_FILE_NAME is None:
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
        _LOG_FILE_NAME = os.path.join(log_dir, f"geomas_launch_{current_time}.log")

    logger = logging.getLogger(logger_name)
    logger.propagate = False
    logger.setLevel(level)

    if not logger.handlers:
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        file_handler = logging.handlers.RotatingFileHandler(
            filename=_LOG_FILE_NAME,
            maxBytes=rotation_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setFormatter(file_formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColorFormatter())

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger