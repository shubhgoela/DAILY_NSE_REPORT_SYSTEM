import logging
import os

def setup_logging(logger_name = 'my_logger',
                  info_file='info.log', 
                  warning_file='warning.log', 
                  error_file='error.log', 
                  environment='PRODUCTION'):
    """
    Sets up the logging configuration with separate files for info, warning, and error messages.
    """

    logger = logging.getLogger(logger_name)

    if environment == 'DEBUG':
        log_level = logging.DEBUG  
    else:
        log_level = logging.WARNING

    logger.setLevel(log_level)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)


    info_file_path = os.path.join(log_dir, info_file)
    warning_file_path = os.path.join(log_dir, warning_file)
    error_file_path = os.path.join(log_dir, error_file)


    info_handler = logging.FileHandler(info_file_path)
    info_handler.setLevel(logging.INFO) 
    # info_handler.addFilter(lambda record: record.levelno == logging.INFO)  # Filter for INFO messages only

    # Create a warning handler (captures WARNING messages only)
    warning_handler = logging.FileHandler(warning_file_path)
    warning_handler.setLevel(logging.WARNING)  # Set level to DEBUG to capture all messages
    # warning_handler.addFilter(lambda record: record.levelno == logging.WARNING)  # Filter for WARNING messages only

    # Create an error handler (captures ERROR messages only)
    error_handler = logging.FileHandler(error_file_path)
    error_handler.setLevel(logging.ERROR)  # Set level to DEBUG to capture all messages
    # error_handler.addFilter(lambda record: record.levelno >= logging.ERROR)  # Filter for ERROR and CRITICAL messages

    # Set up the formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    info_handler.setFormatter(formatter)
    warning_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(info_handler)
    logger.addHandler(warning_handler)
    logger.addHandler(error_handler)

    logger.info("Logging is set up for environment: %s", environment)
    # logger.info(f"Active handlers: {logger.handlers}")  # Log the active handlers

    return logger  # Return the logger for use in other modules