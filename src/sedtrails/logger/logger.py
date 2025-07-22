"""
SedTRAILS Logger
================
A global logger for the SedTRAILS Particle Tracer System.
"""

import logging
import os
import datetime

class LoggerManager:
    def __init__(self):
        self.logger = None
        self.log_dir = "logs"
    
    def setup_logger(self):
        """Set up the logger with timestamped file."""
        if self.logger is not None:
            return self.logger
            
        # Create logs directory if not exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Generate log filename based on timestamp
        log_filename = os.path.join(self.log_dir, f"simulation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        # Create logger directly (avoid basicConfig)
        self.logger = logging.getLogger("simulation_logger")
        
        # Only configure if not already configured
        if not self.logger.handlers:
            # Set logging level
            self.logger.setLevel(logging.DEBUG)
            
            # Create formatter
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            
            # Create file handler
            file_handler = logging.FileHandler(log_filename)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            
            # Add handlers to logger
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
            
            # Prevent propagation to root logger
            self.logger.propagate = False
        
        return self.logger

# Module instance
_logger_manager = LoggerManager()

def log_simulation_state(state: dict, level=logging.INFO) -> None:
    """
    Logs the current state of the simulation.
    
    :param state: Dictionary containing relevant simulation data.
    :param level: Logging level (e.g., logging.INFO, logging.DEBUG)
    """
    logger = _logger_manager.setup_logger()    
    message = ", ".join(f"{key}={value}" for key, value in state.items())
    logger.log(level, f"Simulation State: {message}")

def log_exception(e: Exception) -> None:
    """Logs exceptions with stack trace."""
    logger = _logger_manager.setup_logger()
    logger.error("Simulation encountered an error!", exc_info=True)
