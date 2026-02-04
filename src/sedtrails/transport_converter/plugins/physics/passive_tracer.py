"""A plugin for passive tracer calculations."""

from sedtrails.transport_converter import SedtrailsData
from sedtrails.transport_converter.plugins import BasePhysicsPlugin


class PhysicsPlugin(BasePhysicsPlugin):  # all clases should be called the PhysicsPlugin
    """
    Plugin for passive tracer calculations.
    """

    def __init__(self, config, tracer_config):
        super().__init__()
        self.config = config

    def add_physics(self, sedtrails_data: SedtrailsData, *args, **kwargs) -> None:
        """
        Add flow fields for passive tracer
        """
        flow_velocity_x = sedtrails_data.depth_avg_flow_velocity['x']
        flow_velocity_y = sedtrails_data.depth_avg_flow_velocity['y']
        flow_velocity_magnitude = sedtrails_data.depth_avg_flow_velocity['magnitude']

        print('Adding physics fields to SedtrailsData...')

        # Flow velocities (vector fields)
        sedtrails_data.add_physics_field(
            'depth_avg_flow_velocity',
            {'x': flow_velocity_x, 'y': flow_velocity_y, 'magnitude': flow_velocity_magnitude},
        )
