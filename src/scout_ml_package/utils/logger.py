# File: scout_ml_package/utils/logger.py
import logging
from logging.handlers import RotatingFileHandler
import os
import sys


class StreamToLogger(object):
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


class NoTracebackFilter(logging.Filter):
    def filter(self, record):
        record.exc_text = None
        return True


def setup_logger(log_file="app.log"):
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs"
    )
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger("scout_ml")
    logger.setLevel(logging.DEBUG)

    file_handler = RotatingFileHandler(
        log_path, maxBytes=1024 * 1024, backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Add NoTracebackFilter to remove tracebacks from error logs
    no_traceback_filter = NoTracebackFilter()
    logger.addFilter(no_traceback_filter)
    file_handler.addFilter(no_traceback_filter)
    console_handler.addFilter(no_traceback_filter)

    sys.stdout = StreamToLogger(logger, logging.INFO)
    sys.stderr = StreamToLogger(logger, logging.ERROR)

    return logger


# Create and export the logger
logger = setup_logger()
