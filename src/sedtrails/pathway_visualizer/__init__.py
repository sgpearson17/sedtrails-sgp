"""A directory for the Pathway Visualizer"""

from .simulation_dashboard import SimulationDashboard
from .trajectories import plot_trajectories, read_netcdf, save_trajectories_animation_with_bathymetry

__all__ = ['SimulationDashboard', 'plot_trajectories', 'read_netcdf', 'save_trajectories_animation_with_bathymetry']
