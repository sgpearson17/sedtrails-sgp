"""
Classes for representing internal data structures of the particle tracer.
"""

from dataclasses import dataclass, field
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
        The time of the time step in seconds since the reference date in the datafile.
    """

    time: float

    def __post_init__(self):
        if not isinstance(self.time, float):
            raise TypeError(f"Expected 'time' to be a float, got {type(self.time).__name__}")
    

@dataclass
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
        The time of the position in seconds since the reference date in the datafile.

    """

    x: float
    y: float
    time: Time


    def __post_init__(self):
        if not isinstance(self.x, float):
            raise TypeError(f"Expected 'x' to be a float, got {type(self.x).__name__}")
        if not isinstance(self.y, float):
            raise TypeError(f"Expected 'y' to be a float, got {type(self.y).__name__}")
        if not isinstance(self.time, Time):
            raise TypeError(f"Expected 'time' to be an instance of Time, got {type(self.time).__name__}")


@dataclass
class ParticleTrace:
    """"
    Represents a collection of particle positions and values over time.
    """

    positions: List[Position] = field(default_factory=list)

    def __post_init__(self):
        if not all(isinstance(pos, Position) for pos in self.positions if len(self.positions) > 0):
            raise TypeError("All positions must be instances of the Position class.")

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
    trace: Optional[ParticleTrace] = field(default=None, init=False)

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError(f"Expected 'name' to be a string, got {type(self.name).__name__}")
        if self.trace is not None and not isinstance(self.trace, ParticleTrace):
            raise TypeError(f"Expected 'trace' to be an instance of ParticleTrace, got {type(self.trace).__name__}")


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
    #TODO: define the physical properties of the passive particles
    physical_properties: dict = field(default_factory=dict)
    
    def __post_init__(self):
        # TODO: validate data types once the physical properties are defined
        pass


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
    #TODO: define the physical properties of the passive particles
    physical_properties: dict = field(default_factory=dict)

    def __post_init__(self):
    # TODO: validate data types once the physical properties are defined
        pass


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
    #TODO: define the physical properties of the passive particles
    physical_properties: dict = field(default_factory=dict)

    def __post_init__(self):
        # TODO: validate data types once the physical properties are defined
        pass


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
    suspended_load : float
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
    suspended_load: float
    depth: float
    mean_shear_stress: float
    max_shear_stress: float
    sediment_concentration: float
    wave_velocity: ndarray

    def __post_init__(self):
        # TODO: validate data types once the physical properties are defined
        pass


class Physics:
    """
    Class for storing physics converted values for Particles.

    Attributes
    ----------
    """
    # TODO: define which attributes are needed for the Physics class
    pass

    def __post_init__(self):
        # TODO: validate data types once the physical properties are defined
        pass


if __name__ == "__main__":
    p = Particle(name="Test Particle")
    # s = Sand(name="Test Sand Particle", particle_velocity=0.5)
    # print(s)

    
