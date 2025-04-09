"""
Classes for representing internal data structures of the particle tracer.
"""

from dataclasses import dataclass, field
from abc import ABC
from numpy import ndarray
from typing import List, Optional
from datetime import datetime, timedelta


@dataclass
class Time:
    """
    Class representing a time step in the simulation.

    Attributes
    ----------
    time_step : int
        The current time step in a simulation. A non-negative integer.

    Methods
    -------
    """

    time_step: int = field(default=0)

    def __post_init__(self):
        if not isinstance(self.time_step, int):
            raise TypeError(f"Expected 'time' to be a float, got {type(self.time_step).__name__}")
        if self.time_step < 0:
            raise ValueError(f"Expected 'time' to be a non-negative integer, got {self.time_step}")
    
    def update(self, time_step: int):
        """
        Updates the time step of the simulation.
        Parameters
        ----------
        time_step : int
            The new time step to be set.
        """
        if not isinstance(time_step, int):
            raise TypeError(f"Expected 'time' to be a float, got {type(time_step).__name__}")
        if time_step < 0:
            raise ValueError(f"Expected 'time' to be a non-negative integer, got {time_step}")
        self.time_step = time_step

    def datetime(self, reference_date: datetime, step_size: float) -> datetime:
        """
        Returns the datetime object representing the current time step.
        Parameters
        ----------
        reference_date : datetime
            The reference date from which the time step is calculated. This is the starting date time of the simulation.
        step_size : float
            The size of the time step in seconds.
        Returns
        -------
        datetime
            The datetime object representing the time step.
        """
        
        return reference_date + timedelta(seconds=self.time_step * step_size)
        
        

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
    """

    x: float
    y: float

    def __post_init__(self):
        if not isinstance(self.x, float):
            raise TypeError(f"Expected 'x' to be a float, got {type(self.x).__name__}")
        if not isinstance(self.y, float):
            raise TypeError(f"Expected 'y' to be a float, got {type(self.y).__name__}")



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
    Base class for particles during a simulation time step.

    Attributes
    ----------
    name : str
        The name of the particle.
    x : float
        The x-coordinate of the particle in meters.
    y : float
        The y-coordinate of the particle in meters.
    """

    name: str
    _x: float  # initial position
    _y: float   # initial position

    def __post_init__(self):
        if not isinstance(self.name, str):
            raise TypeError(f"Expected 'name' to be a string, got {type(self.name).__name__}")
        
    @property
    def x(self) -> None:
        return self._x
    
    @x.setter
    def x(self, value: int | float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected 'x' to be an integer or float, got {type(value).__name__}")
        self._x = value

    @property
    def y(self) -> None:
        return self._y
    
    @y.setter
    def y(self, value: int | float) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected 'y' to be an integer or float, got {type(value).__name__}")
        self._y = value



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

    t = Time(time_step=0)
    t.update(5)
    print(t.datetime(datetime(2023, 10, 1), 1.0))
    print(t.datetime(datetime(2023, 10, 1), 30))
