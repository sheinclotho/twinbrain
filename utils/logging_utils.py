"""
Unified logging system for TwinBrain.
Provides consistent logging across all modules with both console and file output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Colored log formatter for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    output_dir: Optional[Path] = None,
    level: int = logging.INFO,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_filename: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration for TwinBrain.
    
    Args:
        output_dir: Directory for log files (if None, no file logging)
        level: Root logger level
        console_level: Console handler level
        file_level: File handler level
        log_filename: Custom log filename (default: twinbrain_YYYYMMDD_HHMMSS.log)
        
    Returns:
        Configured root logger
    """
    # Create formatters
    console_format = '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    file_format = '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    console_formatter = ColoredFormatter(console_format, datefmt=date_format)
    file_formatter = logging.Formatter(file_format, datefmt=date_format)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if output_dir provided)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if log_filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_filename = f'twinbrain_{timestamp}.log'
        
        log_file = output_dir / log_filename
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        root_logger.info(f"Logging to file: {log_file}")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class LogContext:
    """Context manager for logging stages with timing."""
    
    def __init__(self, stage_name: str, logger: Optional[logging.Logger] = None):
        self.stage_name = stage_name
        self.logger = logger or logging.getLogger()
        self.start_time = None
    
    def __enter__(self):
        self.logger.info("=" * 80)
        self.logger.info(f"Starting stage: {self.stage_name}")
        self.logger.info("=" * 80)
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if exc_type is not None:
            self.logger.error(
                f"Stage '{self.stage_name}' failed after {elapsed:.2f}s: {exc_val}",
                exc_info=True
            )
        else:
            self.logger.info(f"Completed stage: {self.stage_name} ({elapsed:.2f}s)")
        
        return False  # Don't suppress exceptions


# Convenience functions
def log_stage(stage_name: str, logger: Optional[logging.Logger] = None):
    """
    Create a context manager for logging a workflow stage.
    
    Usage:
        with log_stage("Data Loading"):
            load_data()
    """
    return LogContext(stage_name, logger)
