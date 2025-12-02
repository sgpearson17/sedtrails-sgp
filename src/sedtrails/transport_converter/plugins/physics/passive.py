"""
A plugin for passive tracer physics in Lagrangian transport calculations
"""

from sedtrails.transport_converter.plugins import BasePhysicsPlugin
from sedtrails.transport_converter import SedtrailsData


class PhysicsPlugin(BasePhysicsPlugin):  # all classes should be called the PhysicsPlugin
    """
    Plugin for passive tracer physics.
    Passive tracers do not interact with the flow or other particles and simply follow the flow field.
    """

    def __init__(self, config, tracer_config):
        super().__init__()
        self.config = config

    def add_physics(self, sedtrails_data: SedtrailsData, grain_properties: dict[str, float], transport_probability_method: str) -> None:
        """
        Add physics using a simple passive tracer approach.
        """
        print('Using passive tracer approach to compute transport velocities and add to SedTRAILS data...')
    
        # Define particle velocities as the depth-averaged flow velocities
        # TODO: allow specification of different flow fields via config
        particle_velocity_x = sedtrails_data.depth_avg_flow_velocity['x']
        particle_velocity_y = sedtrails_data.depth_avg_flow_velocity['y']
        particle_velocity_magnitude = sedtrails_data.depth_avg_flow_velocity['magnitude']
            
        # add particle velocities (vector fields) to sedtrails data
        sedtrails_data.add_physics_field(
            'particle_velocity', 
            {'x': particle_velocity_x, 'y': particle_velocity_y, 'magnitude': particle_velocity_magnitude},
        )

    # additional methods can be added here
