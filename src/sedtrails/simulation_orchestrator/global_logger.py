"""
A wrapper for sedtrailg logger.
This might be proven useful or redundant in the future.
A possible useful case might be if a different logging configurations are needed
for different simulation or analytical workflows.
"""

# Logging functions relevant for Simulation runs
from sedtrails.logger.logger import setup_logging, log_simulation_state

__all__ = ['setup_logging', 'log_simulation_state']
