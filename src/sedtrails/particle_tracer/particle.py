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
    reference_date : str
        The reference date as string in format 'YYYY-MM-DD'
    offset_seconds : int
        The offset in seconds from the reference date.

    Methods
    -------
    get_current_time()
        Returns the current time as a numpy.datetime64 object.
    update(delta_seconds: numpy.timedelta64)
        Updates the current time by adding a delta in seconds.
    """

    # reference_date should be a str, the class will do the transformation into a numpy.datetime64 object
    reference_date: str = field(default='1970-01-01')
    offset_seconds: int = 0
    _reference_date_np: np.datetime64 = field(init=False)

    def _convert_to_datetime64(self, date_str: str) -> np.datetime64:
        """Convert reference_date in str format to numpy.datetime64"""
        return np.datetime64(f"{date_str}T00:00:00", 's')
    
    def __post_init__(self):
        # Accept reference_date in str format TODO: should we allow for non str formats?
        try:
            self._reference_date_np = self._convert_to_datetime64(self.reference_date)
        except ValueError:
            raise ValueError("reference_date must be in format 'YYYY-MM-DD'")
        if not isinstance(self.offset_seconds, int):
                raise TypeError(f"Expected 'offset_seconds' to be an int, got {type(self.offset_seconds).__name__}")

    def get_current_time(self) -> np.datetime64:
        """
        Returns the current time as a numpy.datetime64 object.

        Returns
        -------
        numpy.datetime64
            The current time in the simulation.
        """
        return self._reference_date_np + self.offset_as_timedelta64()

    def get_seconds_since_reference(self) -> int:
        """
        Returns the number of seconds since the reference date/time.
        """
        # Will complete once I can clarify the different between reference_date and simulation start
        pass   

    def offset_as_timedelta64(self) -> np.timedelta64:
        """
        Returns the offset_seconds as a numpy.timedelta64 object.
        """
        return np.timedelta64(self.offset_seconds, 's')

    def update(self, delta_seconds: np.timedelta64):
        """
        Advances the simulation time by a given number of seconds.

        The reference_date defines simulation time zero (Ts=0). The simulation
        can start at any time after that, with offset_seconds representing the
        number of seconds since reference_date. This method increments the
        simulation time by delta_seconds.

        Parameters
        ----------
        delta_seconds : int
            The number of seconds to advance the simulation time.
        """
        if not isinstance(delta_seconds, int):
            raise TypeError(f"Expected 'delta_seconds' to be an int, got {type(delta_seconds).__name__}")
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
