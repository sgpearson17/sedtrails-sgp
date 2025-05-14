"""
Classes for representing internal data structures of the particle tracer.
"""

from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from numpy import ndarray
from typing import Optional
import numpy as np


@dataclass
class Time:
    """
    Class representing a time step in the simulation.

    Attributes
    ----------
    reference_date : numpy.datetime64
        The reference date for calculating time.
    offset_seconds : numpy.timedelta64
        The offset in seconds from the reference date.

    Methods
    -------
    get_current_time()
        Returns the current time as a numpy.datetime64 object.
    update(delta_seconds: numpy.timedelta64)
        Updates the current time by adding a delta in seconds.
    """

    reference_date: np.datetime64 = field(default=np.datetime64('1970-01-01T00:00:00', 's'))
    offset_seconds: np.timedelta64 = field(default=np.timedelta64(0, 's'))

    def __post_init__(self):
        if not isinstance(self.reference_date, np.datetime64):
            raise TypeError(f"Expected 'reference_date' to be a numpy.datetime64, got {type(self.reference_date).__name__}")
        if self.reference_date.dtype != 'datetime64[s]':
            raise ValueError("reference_date must have a resolution of seconds (datetime64[s]).")
        if not isinstance(self.offset_seconds, np.timedelta64):
            raise TypeError(f"Expected 'offset_seconds' to be a numpy.timedelta64, got {type(self.offset_seconds).__name__}")
        if self.offset_seconds.dtype != 'timedelta64[s]':
            raise ValueError("offset_seconds must have a resolution of seconds (timedelta64[s]).")

    def get_current_time(self) -> np.datetime64:
        """
        Returns the current time as a numpy.datetime64 object.

        Returns
        -------
        numpy.datetime64
            The current time in the simulation.
        """
        return self.reference_date + self.offset_seconds

    def update(self, delta_seconds: np.timedelta64):
        """
        Updates the current time by adding a delta in seconds.

        Parameters
        ----------
        delta_seconds : numpy.timedelta64
            The time delta to add to the current time.
        """
        if not isinstance(delta_seconds, np.timedelta64):
            raise TypeError(f"Expected 'delta_seconds' to be a numpy.timedelta64, got {type(delta_seconds).__name__}")
        if delta_seconds.dtype != 'timedelta64[s]':
            raise ValueError("delta_seconds must have a resolution of seconds (timedelta64[s]).")
        self.offset_seconds += delta_seconds


@dataclass
class Particle(ABC):
    """
    Base class for particles during a simulation time step.

    Attributes
    ----------

    id : int
        The unique identifier of the particle.
    x : float
        The x-coordinate of the particle in meters.
    y : float
        The y-coordinate of the particle in meters.
    is_mobile : bool
        Whether the particle can move in the current simulation step or not. Default is True.
    name : str
        A name for the particle. Optional.
    """

    id: int
    _x: float  # initial position
    _y: float  # initial position
    _is_mobile: bool = field(default=True)  # whether the particle is mobile or not
    name: Optional[str] = field(default='')  # name of the particle

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

    @property
    def is_mobile(self) -> None:
        return self._is_mobile

    @is_mobile.setter
    def is_mobile(self, value: bool) -> None:
        if not isinstance(value, bool):
            raise TypeError(f"Expected 'is_mobile' to be a boolean, got {type(value).__name__}")
        self._is_mobile = value

    @is_mobile.getter
    def is_mobile(self) -> bool:
        return self._is_mobile

    @abstractmethod
    def particle_velocity(self) -> float:
        """
        A method to compute the particle's velocity.
        This method should be implemented by each particle type.

        Returns
        -------
        float
            The velocity of the particle in meters per second.
        """
        pass


@dataclass
class Mud(Particle):
    """
    Class representing mud particles.

    Attributes
    ----------
    particle_velocity : float
        The velocity of the mud particles.
    """

    # TODO: define the physical properties of the passive particles
    physical_properties: dict = field(default_factory=dict)

    def __post_init__(self):
        # TODO: validate data types once the physical properties are defined
        pass

    def particle_velocity(self) -> float:
        """
        A method to compute the velocity of a mud particle.

        Returns
        -------
        float
            The velocity of the particle in meters per second.
        """
        pass  # TODO: implement the velocity calculation for mud particles


@dataclass
class Sand(Particle):
    """
    Class representing sand particles.

    Attributes
    ----------
    particle_velocity : float
        The velocity of the sand particles.
    """

    # TODO: define the physical properties of the passive particles
    physical_properties: dict = field(default_factory=dict)

    def __post_init__(self):
        # TODO: validate data types once the physical properties are defined
        pass

    def particle_velocity(self) -> float:
        """
        A method to compute the velocity of a sand particle.

        Returns
        -------
        float
            The velocity of the particle in meters per second.
        """
        pass  # TODO: implement the velocity calculation for sand particles


@dataclass
class Passive(Particle):
    """
    Class representing passive particles.

    Attributes
    ----------
    particle_velocity : float
        The velocity of the passive particles.
    """

    # TODO: define the physical properties of the passive particles
    physical_properties: dict = field(default_factory=dict)

    def __post_init__(self):
        # TODO: validate data types once the physical properties are defined
        pass

    def particle_velocity(self) -> float:
        """
        A method to compute the velocity of a passive particle.

        Returns
        -------
        float
            The velocity of the particle in meters per second.
        """
        pass  # TODO: implement the velocity calculation for passive particles


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
    suspended_sed_transport : float
        The suspended sediment transport of the particle in kg/m/s.
    depth : float
        The water depth of the particle in meters.
    mean_shear_stress : float
        The mean bed shear stress of the particle in Pa.
    max_shear_stress : float
        The maximum bed shear stress of the particle in Pa.
    sed_concentration : float
        The suspended sediment concentration of the particle in kg/m^3.
    wave_velocity : ndarray
        The non-linear wave velocity of the particle in m/s.
    """

    x: float
    y: float
    bed_level: float
    bed_load: ndarray
    flow_velocity: float
    sed_concentration: float
    water_level: float
    water_depth: float
    averaged_flow_velocity: float
    suspended_sed_transport: float
    wave_velocity: ndarray
    mean_shear_stress: float
    max_shear_stress: float


@dataclass
class Physics:
    """
    Class for storing physics converted values of a Particle.

    Attributes
    ----------
    """

    # TODO: define which attributes are needed for the Physics class
    pass

    def __post_init__(self):
        # TODO: validate data types once the physical properties are defined
        pass


if __name__ == '__main__':
    s = Sand(id=1, _x=0.0, _y=0.0, name='Sand Particle')

    print(s)
