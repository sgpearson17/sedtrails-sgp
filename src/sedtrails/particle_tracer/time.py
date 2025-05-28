"""
Time class used internally to represent simulation time.
"""

from dataclasses import dataclass, field
import numpy as np
import re
from sedtrails.exceptions.exceptions import DateFormatError

@dataclass
class Time:
    """
    Class representing a time step in the simulation.

    Attributes
    ----------
    reference_date : str
        The reference date as string in format 'YYYY-MM-DD hh:mm:ss'
    start_time : str
        The simulation start time as string in format 'YYYY-MM-DD hh:mm:ss'.
    duration : int
        The simulation duration in seconds.
    time_step : int or float
        The simulation time step in seconds.

    Methods
    -------
    get_current_time()
        Returns the current time as a numpy.datetime64 object.
    update(delta_seconds: numpy.timedelta64)
        Updates the current time by adding a delta in seconds.
    """

    # reference_date and start_time should be a str, 
    # the class will do the transformation into a numpy.datetime64 object
    reference_date: str = field(default='1970-01-01 00:00:00')
    start_time: str = field(default='1970-01-01 00:00:00')
    duration: int = 0
    time_step: float = 1.0
    _reference_date_np: np.datetime64 = field(init=False)

    @property
    def end_time(self) -> int:
        """
        Returns the simulation end time in seconds from the reference date.
        """
        return self.start_time + self.duration        

    def _convert_to_datetime64(self, date_str: str) -> np.datetime64:
        """
        Convert date in str format to numpy.datetime64 enforcing 'YYYY-MM-DD hh:mm:ss'.
        """
        if not re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", date_str):
            raise ReferenceDateFormatError(
                f"date string '{date_str}' does not match required format 'YYYY-MM-DD hh:mm:ss'"
            )
        return np.datetime64(date_str, 's')
    
    def __post_init__(self):
        # Accept reference_date and start_time in str format, enforcing YYYY-MM-DD hh:mm:ss format
        try:
            self._reference_date_np = self._convert_to_datetime64(self.reference_date)
            self._start_time_np = self._convert_to_datetime64(self.start_time)
        except DateFormatError as e:
            raise DateFormatError(str(e)) from e
        if not isinstance(self.time_step, (int, float)):
            raise TypeError(f"Expected 'time_step' to be int or float, got {type(self.time_step).__name__}")

    def get_current_time(self, step: int = 0) -> np.datetime64:
        """
        Returns the current time in simulation as a numpy.datetime64 object,
        given a simulation step.

        Parameters
        ----------
        step : int
            The simulation step number (default is 0).

        Returns
        -------
        numpy.datetime64
            The current time in the simulation.
        """
        total_seconds = int(self.start_time + step * self.time_step)
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

