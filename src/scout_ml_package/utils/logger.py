# File: scout_ml_package/utils/logger.py
def configure_logger(logger_name, log_dir_path, log_file_name='app.log', log_level=logging.DEBUG, console_level=logging.INFO):
    """
    Configures a logger with specified name and log file path.

    Parameters:
    - logger_name: Name of the logger.
    - log_dir_path: Directory path for the log file.
    - log_file_name: Name of the log file (default: 'app.log').
    - log_level: Logging level for the file handler (default: DEBUG).
    - console_level: Logging level for the console handler (default: INFO).

    Returns:
    - A configured logger object.
    """
    import os

    # Ensure the log directory exists
    os.makedirs(log_dir_path, exist_ok=True)

    # Construct the full log file path
    log_file_path = os.path.join(log_dir_path, log_file_name)

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
# logger = configure_logger('demo_logger', '/data/model-data/logs', 'demo.log')

# `log_dir_path`: Specifies the directory where the log file should be saved.
# `log_file_name`: Specifies the name of the log file. If not provided, it defaults to `'app.log'`.
# 	
