"""
SedTRAILS Logger
================
A global logger for the SedTRAILS Particle Tracer System.
"""

import logging
import os
import datetime

# Create logs directory if not exists
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Generate log filename based on timestamp
log_filename = os.path.join(LOG_DIR, f"simulation_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Configure logger
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Optional: Print logs to console
    ]
)

logger = logging.getLogger("simulation_logger")

def log_simulation_state(state: dict, level=logging.INFO) -> None:
    """
    Logs the current state of the simulation.
    
    :param state: Dictionary containing relevant simulation data.
    :param level: Logging level (e.g., logging.INFO, logging.DEBUG)
    """
    message = ", ".join(f"{key}={value}" for key, value in state.items())
    logger.log(level, f"Simulation State: {message}")

def log_exception(e: Exception) -> None:
    """Logs exceptions with stack trace."""
    logger.error("Simulation encountered an error!", exc_info=True)
