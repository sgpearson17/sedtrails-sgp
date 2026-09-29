"""A plugin for mangrove propagule calculations."""

import logging

import numpy as np

from sedtrails.transport_converter import SedtrailsData
from sedtrails.transport_converter.plugins import BasePhysicsPlugin

logger = logging.getLogger(__name__)


class PhysicsPlugin(BasePhysicsPlugin):
    """Plugin preparing physics fields for mangrove propagule tracking."""

    def __init__(self, config, tracer_config):
        super().__init__()
        self.config = config
        self.tracer_config = tracer_config or {}

    def add_physics(self, sedtrails_data: SedtrailsData, *args, **kwargs) -> None:
        """Expose passive advection fields plus optional mangrove support fields."""
        flow_velocity_x = sedtrails_data.depth_avg_flow_velocity['x']
        flow_velocity_y = sedtrails_data.depth_avg_flow_velocity['y']
        flow_velocity_magnitude = sedtrails_data.depth_avg_flow_velocity['magnitude']

        logger.info('Adding mangrove physics fields to SedtrailsData')

        sedtrails_data.add_physics_field(
            'depth_avg_flow_velocity',
            {'x': flow_velocity_x, 'y': flow_velocity_y, 'magnitude': flow_velocity_magnitude},
        )

        mangrove_config = self.tracer_config.get('mangrove', self.tracer_config)
        settlement_config = mangrove_config.get('settlement', {}) if isinstance(mangrove_config, dict) else {}
        if not settlement_config.get('enabled', False):
            return

        if settlement_config.get('depth_mode', 'instantaneous') != 'max_over_simulation':
            return

        if hasattr(sedtrails_data, 'max_water_depth') and getattr(sedtrails_data, 'max_water_depth') is not None:
            max_water_depth = np.asarray(getattr(sedtrails_data, 'max_water_depth'))
        else:
            max_water_depth = np.max(np.asarray(sedtrails_data.water_depth), axis=0)

        sedtrails_data.add_physics_field('max_water_depth', max_water_depth)
