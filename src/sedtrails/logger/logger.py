"""
SedTRAILS Logger
================
A global logger for the SedTRAILS Particle Tracer System.
"""

import logging
import os

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
        
        # Simple log file name
        log_filename = os.path.join(self.log_dir, "log.txt")
        
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

            # Log the file location immediately
            absolute_path = os.path.abspath(log_filename)
            self.logger.info("=== LOGGING INITIALIZED ===")
            self.logger.info(f"Log file: {os.path.basename(absolute_path)}")
            self.logger.info(f"Location: {os.path.dirname(absolute_path)}")

        return self.logger

# Module instance
logger_manager = LoggerManager()

def _format_seeding_strategy(strategy):
    """Helper function to format seeding strategy for readable logging."""
    if not strategy or strategy == 'unknown':
        return 'none'
    
    if isinstance(strategy, str):
        return strategy
    
    if isinstance(strategy, dict):
        formatted_parts = []
        
        for strategy_type, config in strategy.items():
            if strategy_type == 'point':
                points = config.get('points', [])
                if isinstance(points, list):
                    point_count = len(points)
                    if point_count == 1:
                        formatted_parts.append(f"single point at {points[0]}")
                    else:
                        formatted_parts.append(f"{point_count} points")
                else:
                    formatted_parts.append("point seeding")
            
            elif strategy_type == 'line':
                formatted_parts.append("line seeding")
            
            elif strategy_type == 'grid':
                formatted_parts.append("grid seeding")
            
            elif strategy_type == 'random':
                count = config.get('count', 'unknown')
                formatted_parts.append(f"random ({count} particles)")
            
            else:
                formatted_parts.append(f"{strategy_type} seeding")
        
        return ', '.join(formatted_parts)
    
    return str(strategy)

def log_simulation_state(state: dict, level=logging.INFO) -> None:
    """
    Logs the current state of the simulation with human-readable sentences.
    Long messages are split into multiple lines for readability.
    """
    logger = logger_manager.setup_logger()
    
    status = state.get('status', state.get('state', 'unknown'))
    
    # Create messages, split when necessary
    if status == 'simulation_started':
        command = state.get('command', 'unknown')
        config = state.get('config_file', 'unknown')
        python_ver = state.get('python_version', 'unknown')
        working_dir = state.get('working_directory', 'unknown')
        
        # Split into multiple log lines
        logger.log(level, "=== SIMULATION START ===")
        logger.log(level, f"Command: {command}")
        logger.log(level, f"Python: {python_ver}")
        logger.log(level, f"Config: {config}")
        logger.log(level, f"Working directory: {working_dir}")
        
    elif status == 'config_loading':
        config_path = state.get('config_file_path', 'unknown')
        logger.log(level, f"Loading configuration: {config_path}")
        
    elif status == 'simulation_parameters':
        duration = state.get('duration', 'unknown')
        timestep = state.get('timestep', 'unknown')
        start_time = state.get('start_time', 'unknown')
        strategy = state.get('seeding_strategy', 'unknown')
        output_dir = state.get('output_dir', 'unknown')
        
        # Format seeding strategy for readability
        formatted_strategy = _format_seeding_strategy(strategy)
        
        # Split parameters into multiple lines
        logger.log(level, "--- Simulation Parameters ---")
        logger.log(level, f"Duration: {duration}")
        logger.log(level, f"Time step: {timestep}")
        logger.log(level, f"Start time: {start_time}")
        logger.log(level, f"Seeding: {formatted_strategy}")
        logger.log(level, f"Output directory: {output_dir}")
        
    elif status == 'particles_created' or status == 'particles_initialized':
        count = state.get('count', 1)
        position = state.get('start_position', state.get('position', 'unknown'))
        seeding_strategy = state.get('seeding_strategy', {})
        
        logger.log(level, f"Created {count} particle(s)")
        logger.log(level, f"Starting position: {position}")
        
        # Format seeding strategy if provided
        if seeding_strategy and seeding_strategy != {}:
            formatted_strategy = _format_seeding_strategy(seeding_strategy)
            logger.log(level, f"Seeding method: {formatted_strategy}")
            
    elif status == 'data_conversion_completed':
        timesteps = state.get('num_timesteps', 'unknown')
        field_name = state.get('flow_field_name', 'unknown')
        duration = state.get('simulation_duration_seconds', 'unknown')
        timestep = state.get('simulation_timestep_seconds', 'unknown')
        
        logger.log(level, "--- Data Conversion Complete ---")
        logger.log(level, f"Timesteps: {timesteps}")
        logger.log(level, f"Flow field: {field_name}")
        logger.log(level, f"Duration: {duration}s")
        logger.log(level, f"Time step: {timestep}s")
        
    elif status == 'numba_compilation_started':
        grid_x = state.get('grid_size_x', 'unknown')
        grid_y = state.get('grid_size_y', 'unknown')
        logger.log(level, f"Compiling calculator for {grid_x}x{grid_y} grid")
        
    elif status == 'compilation_complete':
        time_sec = state.get('time_sec', 'unknown')
        logger.log(level, f"Calculator compiled in {time_sec} seconds")
        
    elif status == 'warmup_complete':
        time_sec = state.get('time_sec', 'unknown')
        logger.log(level, f"JIT warmed up in {time_sec} seconds")
        
    elif status == 'simulation_progress':
        progress = state.get('progress_pct', 0)
        step = state.get('step', 0)
        position = state.get('position', 'unknown')
        logger.log(level, f"Progress: {progress}% (step {step}) at {position}")
        
    elif status == 'simulation_complete' or status == 'simulation_completed':
        steps = state.get('total_steps', 'unknown')
        time_sec = state.get('total_time_sec', 'unknown')
        final_pos = state.get('final_position', 'unknown')
        
        logger.log(level, "=== SIMULATION COMPLETE ===")
        logger.log(level, f"Total steps: {steps}")
        logger.log(level, f"Runtime: {time_sec}s")
        logger.log(level, f"Final position: {final_pos}")
        
    elif status == 'visualization_saved':
        filename = state.get('file', 'trajectory plot')
        output_path = state.get('output_plot_path', 'unknown')
        
        logger.log(level, f"Visualization saved: {filename}")
        if output_path != 'unknown':
            logger.log(level, f"Location: {output_path}")
            
    elif status == 'creating_visualization':
        points = state.get('trajectory_points', 'unknown')
        logger.log(level, f"Creating visualization with {points} trajectory points")

    elif status == 'simulation_failed':
        error_type = state.get('error_type', 'unknown')
        error_message = state.get('error_message', 'unknown')
        
        logger.log(level, "=== SIMULATION FAILED ===")
        logger.log(level, f"Error type: {error_type}")
        logger.log(level, f"Error message: {error_message}")
        logger.log(level, "Check log above for full stack trace")
        
    else:
        # Fallback for unmapped states
        clean_status = status.replace('_', ' ').title()
        logger.log(level, f"Status: {clean_status}")
        
        # Log each parameter on separate line if many parameters
        params = {k: v for k, v in state.items() if k not in ['status', 'state']}
        if len(params) > 3:  # If more than 3 params, split them
            for k, v in params.items():
                logger.log(level, f"  {k}: {v}")
        elif params:  # If 3 or fewer, keep on one line
            param_str = ', '.join(f"{k}: {v}" for k, v in params.items())
            logger.log(level, f"  {param_str}")

def log_exception(e: Exception, context: str = None) -> None:
    """
    Logs exceptions with stack trace and context information.
    
    Parameters
    ----------
    e : Exception
        The exception that occurred
    context : str, optional
        Additional context about where/when the exception occurred
    """
    logger = logger_manager.setup_logger()
    
    # Log exception with context
    if context:
        logger.error(f"=== ERROR: {context} ===")
    else:
        logger.error("=== SIMULATION ERROR ===")
    
    logger.error(f"Exception type: {type(e).__name__}")
    logger.error(f"Exception message: {str(e)}")
    logger.error("Stack trace:", exc_info=True)
    logger.error("=" * 50)
