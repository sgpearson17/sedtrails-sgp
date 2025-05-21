"""
Time class used internally to represent simulation time.
"""

from dataclasses import dataclass, field
import numpy as np
import re
from sedtrails.exceptions.exceptions import ReferenceDateFormatError

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
        """
        Convert reference_date in str format to numpy.datetime64
        enforcing 'YYYY-MM-DD hh:mm:ss'.
        """
        if not re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", date_str):
            raise ReferenceDateFormatError(
                f"reference_date '{date_str}' does not match required format 'YYYY-MM-DD hh:mm:ss'"
            )
        return np.datetime64(date_str, 's')
    
    def __post_init__(self):
        # Accept reference_date in str format, enforcing YYYY-MM-DD hh:mm:ss format
        try:
            self._reference_date_np = self._convert_to_datetime64(self.reference_date)
        except ReferenceDateFormatError as e:
            raise ReferenceDateFormatError(str(e)) from e
        if not isinstance(self.start_time, int):
                raise TypeError(f"Expected 'start_time' to be an int, got {type(self.start_time).__name__}")

    def get_current_time(self, delta_seconds: int = 0) -> np.datetime64:
        """
        Returns the current time in simulation as a numpy.datetime64 object,
        optionally including an additional delta in seconds.

        Parameters
        ----------
        delta_seconds : int, optional
            Additional seconds to add to the current simulation time (default is 0).

        Returns
        -------
        numpy.datetime64
            The current time in the simulation.
        """
        total_seconds = self.start_time + delta_seconds
        return self._reference_date_np + np.timedelta64(total_seconds, 's')

    def get_seconds_since_reference(self, delta_seconds: int = 0) -> int:
        """
        Returns the current time in simulation as integer in units of seconds.

        Parameters
        ----------
        delta_seconds : int, optional
            Additional seconds to add to the current simulation time (default is 0).

        Returns
        -------
        total_seconds : int
            The total seconds in the simulation since the reference date.
        """
        total_seconds = self.start_time + delta_seconds
        return total_seconds

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

