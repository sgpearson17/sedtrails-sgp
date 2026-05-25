"""
Particle State Checker
======================
Applies status and burial methods to determine a particle status 
(e.g., active, dead, alive, stuck, etc) and burial state
"""

from enum import Enum
from sedtrails.particle_tracer.particle import Particle


class Status(Enum):
    """
    Enumeration for particle status.
    """
    ACTIVE = "active"
    DEAD = "dead"
    ALIVE = "alive"
    STUCK = "stuck"

    # add more statuses as needed


class StateChecker:
    """
    Class to check the state of particles in a simulation.
    """

    def __init__(self, **config):
        """
        Initialize the StateChecker with configuration parameters.

        Parameters
        ----------
        config : dict
            Configuration parameters for the state checker provided by
            the configuration interface.
        """
        # Accept variable key/value configuration settings.
        self.config = config


    def check_state(self, particle: Particle, flow_field) -> str:
        """
        Check the state of a particle based on its properties.

        Parameters
        ----------
        particle : Particle
            The particle to check.
        flow_field : object 
            The flow field data to use for checking the state.

        Returns
        -------
        Status
            The state of the particle (e.g., ACTIVE, ALIVE, etc).
        """

        # the position of a particle is given by the particle object as
        # x, y coordinates. 

        # Implement state checking logic here:  do somthing with the particle and the flow field

        # TODO: return the state of the particle 
        # as Status.ACTIVE, Status.DEAD, etc.
        pass

