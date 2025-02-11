# File: scout_ml_package/utils/logger.py
import logging

def configure_logger(logger_name, log_file_path, log_level=logging.DEBUG, console_level=logging.INFO):
    """
    Configures a logger with specified name and log file path.

    Parameters:
    - logger_name: Name of the logger.
    - log_file_path: Path to the log file.
    - log_level: Logging level for the file handler (default: DEBUG).
    - console_level: Logging level for the console handler (default: INFO).

    Returns:
    - A configured logger object.
    """
    # Create a logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # Create a formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create a file handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    # Create a stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(console_level)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

# Usage
#logger = configure_logger('my_logger', 'logs/my_app.log')

