"""
Classes for representing internal data structures of the particle tracer.
"""

from dataclasses import dataclass
from abc import ABC
from numpy import ndarray
from typing import List, Optional




@dataclass
class Time:
    """
    Class representing a time step in the simulation.

    Attributes
    ----------
    time : float
        The time of the time step in seconds.
    """

    time: float
    

@ dataclass
class Position:
    """
    Class representing the position of a Particle on a particular time.

    Attributes
    ----------
    x : float
        The x-coordinate of the particle in meters.
    y : float
        The y-coordinate of the particle in meters.
    time : Time
        The time of the position in seconds.

    """

    x: float
    y: float
    time: Time



@dataclass
class ParticleTrace:
    """"
    Represents a collection of particle positions and values over time.
    """

    positions: List[Position] = []

    def current_position(self):
        """
        Returns the current position of the particle.
        """
        pass


@dataclass
class Particle(ABC):
    """
    Base class for particles in the sediment tracer model.

    Attributes
    ----------
    name : str
        The name of the particle.
    """

    name: str
    trace: Optional[ParticleTrace] = None





@dataclass
class Mud(Particle):
    """
    Class representing mud particles.

    Attributes
    ----------
    particle_velocity : float
        The velocity of the mud particles.
    """

    particle_velocity: float


@dataclass
class Sand(Particle):
    """
    Class representing sand particles.

    Attributes
    ----------
    particle_velocity : float
        The velocity of the sand particles.
    """

    particle_velocity: float


@dataclass
class Passive(Particle):
    """
    Class representing passive particles.

    Attributes
    ----------
    particle_velocity : float
        The velocity of the passive particles.
    """

    particle_velocity: float


@dataclass
class InterpolatedValue:
    """
    Class for storing interpolated values of a Particle.

    Attributes
    ----------
    bed_level : float
        The bed level of the particle in meters.
    averaged_velocity : float
        The averaged flow velocity of the particle in m/s.
    no_fractions : int
        The number of fractions of the particle.
    bed_load : ndarray
        The bed load sediment transport of the particle in kg/m/s.
    sediment_transport : float
        The suspended sediment transport of the particle in kg/m/s.
    depth : float
        The water depth of the particle in meters.
    mean_shear_stress : float
        The mean bed shear stress of the particle in Pa.
    max_shear_stress : float
        The maximum bed shear stress of the particle in Pa.
    sediment_concentration : float
        The suspended sediment concentration of the particle in kg/m^3.
    wave_velocity : ndarray
        The non-linear wave velocity of the particle in m/s. 
    """

    # TODO: move some of these attributes to the Physics class
    bed_level: float
    averaged_velocity: float
    no_fractions: int
    bed_load: ndarray
    sediment_transport: float
    depth: float
    mean_shear_stress: float
    max_shear_stress: float
    sediment_concentration: float
    wave_velocity: ndarray


class Physics:
    """
    Class for storing physics converted values for Particles.

    Attributes
    ----------
    """
    # TODO: define which attributes are needed for the Physics class
    pass



if __name__ == "__main__":
    mud = Mud(name="Mud-1", particle_velocity=0.1)

    