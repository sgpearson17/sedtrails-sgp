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
    Logs the current state of the simulation with human-readable sentences.
    """
    logger = _logger_manager.setup_logger()
    
    status = state.get('status', state.get('state', 'unknown'))
    
    # Create full sentence messages
    if status == 'particles_created' or status == 'particles_initialized':
        count = state.get('count', 1)
        position = state.get('start_position', state.get('position', 'unknown'))
        message = f"Created {count} particle(s) starting at position {position}"
        
    elif status == 'compilation_complete':
        time_sec = state.get('time_sec', 'unknown')
        message = f"Particle calculator compiled successfully in {time_sec} seconds"
        
    elif status == 'warmup_complete':
        time_sec = state.get('time_sec', 'unknown')
        message = f"JIT compiler warmed up in {time_sec} seconds"
        
    elif status == 'simulation_progress':
        progress = state.get('progress_pct', 0)
        step = state.get('step', 0)
        position = state.get('position', 'unknown')
        message = f"Simulation {progress}% complete (step {step}) - particle at {position}"
        
    elif status == 'simulation_complete' or status == 'simulation_completed':
        steps = state.get('total_steps', 'unknown')
        time_sec = state.get('total_time_sec', 'unknown')
        final_pos = state.get('final_position', 'unknown')
        message = f"Simulation completed! {steps} steps in {time_sec}s, final position: {final_pos}"
        
    elif status == 'visualization_saved':
        filename = state.get('file', 'trajectory plot')
        message = f"Trajectory visualization saved as {filename}"
        
    else:
        # Fallback for unmapped states
        message = f"{status.replace('_', ' ').title()}"
        details = [f"{k}: {v}" for k, v in state.items() if k not in ['status', 'state']]
        if details:
            message += f" ({', '.join(details)})"
    
    logger.log(level, message)

def log_exception(e: Exception) -> None:
    """Logs exceptions with stack trace."""
    logger = _logger_manager.setup_logger()
    logger.error("Simulation encountered an error!", exc_info=True)
