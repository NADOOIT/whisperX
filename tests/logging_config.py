import logging
import os
import sys
from datetime import datetime

def setup_logging(test_name):
    """Set up logging for tests with both file and console output."""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)

    # Create a unique log file for this test run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{test_name}_{timestamp}.log')

    # Configure logging
    logger = logging.getLogger('whisperx_test')
    logger.setLevel(logging.DEBUG)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def log_test_result(logger, test_name, result, error=None):
    """Log test results with appropriate level and details."""
    if error:
        logger.error(f"Test '{test_name}' failed: {error}")
        if hasattr(error, '__dict__'):
            logger.debug(f"Error details: {error.__dict__}")
    else:
        logger.info(f"Test '{test_name}' passed")

def log_test_values(logger, **kwargs):
    """Log test input values and intermediate results."""
    logger.debug("Test values:")
    for key, value in kwargs.items():
        logger.debug(f"{key}: {value}")
