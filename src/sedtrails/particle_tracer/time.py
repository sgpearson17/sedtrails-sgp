"""
Time class used internally to represent simulation time.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class Time:
    """
    Class representing a time step in the simulation.

    Attributes
    ----------
    reference_date : str
        The reference date as string in format 'YYYY-MM-DD'
    start_time : int
        The simulation start time in seconds (from the reference date).
    end_time : int
        The simulation end time in seconds (from the reference date).

    Methods
    -------
    get_current_time()
        Returns the current time as a numpy.datetime64 object.
    update(delta_seconds: numpy.timedelta64)
        Updates the current time by adding a delta in seconds.
    """

    # reference_date should be a str, the class will do the transformation into a numpy.datetime64 object
    reference_date: str = field(default='1970-01-01')
    start_time: int = 0
    end_time: int = 0
    _reference_date_np: np.datetime64 = field(init=False)

    def _convert_to_datetime64(self, date_str: str) -> np.datetime64:
        """Convert reference_date in str format to numpy.datetime64"""
        return np.datetime64(f"{date_str}T00:00:00", 's')
    
    def __post_init__(self):
        # Accept reference_date in str format TODO: should we allow for non str formats?
        try:
            self._reference_date_np = self._convert_to_datetime64(self.reference_date)
        except ValueError as e:
            raise ValueError("reference_date must be in format 'YYYY-MM-DD'") from e
        if not isinstance(self.start_time, int):
                raise TypeError(f"Expected 'start_time' to be an int, got {type(self.start_time).__name__}")

    def get_current_time(self) -> np.datetime64:
        """
        Returns the current time as a numpy.datetime64 object.

        Returns
        -------
        numpy.datetime64
            The current time in the simulation.
        """
        return self._reference_date_np + self.time_as_timedelta64()

    def get_seconds_since_reference(self) -> int:
        """
        Returns the number of seconds since the reference date/time.
        """
        # Will complete once I can clarify the different between reference_date and simulation start
        pass   

    def time_as_timedelta64(self) -> np.timedelta64:
        """
        Returns the start_time as a numpy.timedelta64 object.
        """
        return np.timedelta64(self.start_time, 's')

    def update(self, delta_seconds: np.timedelta64):
        """
        Advances the simulation time by a given number of seconds.

        The reference_date defines simulation time zero (Ts=0). The simulation
        can start at any time after that, with start_time representing the
        number of seconds since reference_date. This method increments the
        simulation time by delta_seconds.

        Parameters
        ----------
        delta_seconds : int
            The number of seconds to advance the simulation time.
        """
        if not isinstance(delta_seconds, int):
            raise TypeError(f"Expected 'delta_seconds' to be an int, got {type(delta_seconds).__name__}")
        self.start_time += delta_seconds

