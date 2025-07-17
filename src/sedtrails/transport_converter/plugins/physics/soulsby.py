"""A plugin for Soulsby sediment transport physics calculations."""

from sedtrails.transport_converter.plugins import BasePhysicsPlugin


class PhysicsPlugin(BasePhysicsPlugin):  # all clases should be called the PhysicsPlugin
    """
    Plugin for Soulsby et al. (2011) sediment transport physics calculations.
    This plugin implements the physics calculations as described in Soulsby et al. (2011).
    """

    def __init__(
        self,
    ):  # this is the minumum required for the plugin to work. Additional parameters can be added as needed.
        super().__init__()

    def add_physics(self) -> None:  # Add additional parameters as needed
        """
        Add physics using Soulsby et al. (2011) approach.
        """

        # Placeholder for Soulsby calculations
        raise NotImplementedError(
            # Notes by Bart:
            'Soulsby et al. (2011) method not yet implemented. '
            'This would have a completely different workflow:\n'
            '1. Focus on individual particle tracking velocities\n'
            '2. Different approach to settling and resuspension\n'
            '3. Particle-specific rather than layer-based calculations\n'
            'See: Soulsby, R. L., et al. (2011). Lagrangian model for simulating '
            'the dispersal of sand-sized particles in coastal waters.'
        )
