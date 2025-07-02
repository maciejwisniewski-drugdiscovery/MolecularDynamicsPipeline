import logging
import os
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str = "plinder_dynamics", 
                log_level: int = logging.INFO,
                log_dir: str = "logs",
                console_output: bool = True) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name (str): Name of the logger
        log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG)
        log_dir (str): Directory to store log files
        console_output (bool): Whether to output logs to console
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Create logs directory if it doesn't exist
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler - create a new log file for each run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir_path / f"{name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler (optional)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

# Convenience functions for different log levels
def get_logger(name: str = "plinder_dynamics") -> logging.Logger:
    """Get an existing logger or create a new one with default settings."""
    logger = logging.getLogger(name)
    if not logger.handlers:  # Only setup if logger hasn't been configured
        logger = setup_logger(name)
    return logger

def log_error(logger: logging.Logger, message: str, exc_info: bool = True):
    """Log an error message with optional exception info."""
    logger.error(message, exc_info=exc_info)

def log_warning(logger: logging.Logger, message: str):
    """Log a warning message."""
    logger.warning(message)

def log_info(logger: logging.Logger, message: str):
    """Log an info message."""
    logger.info(message)

def log_debug(logger: logging.Logger, message: str):
    """Log a debug message."""
    logger.debug(message) 