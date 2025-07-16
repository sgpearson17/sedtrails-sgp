"""
Time class used internally to represent simulation time.
"""

from dataclasses import dataclass, field
import numpy as np
import re
from sedtrails.exceptions.exceptions import DateFormatError, DurationFormatError


class Duration:
    """
    Class representing a duration in the simulation.

    Attributes
    ----------
    duration : str
        The duration as a string in format '3D 2H1M3S'.

    Methods
    -------
    to_seconds()
        Converts the duration string to total seconds.
    """

    def __init__(self, duration: str):
        self._duration = duration

    @property
    def duration(self) -> str:
        """
        Returns the duration string.
        """
        return self._duration

    def to_seconds(self) -> int:
        """
        Parse a duration string ('3D 2H1M3S') into the total number of seconds.

        The duration string may include days (D), hours (H), minutes (M), and seconds (S)
        in any combination and order, separated by optional spaces. Missing units are treated as zero.

        Parameters
        ----------
        duration_str : str
            Duration string to parse ('3D 2H1M3S', '45S', '1H 30M').

        Returns
        -------
        int
            Total duration in seconds.

        Raises
        ------
        DurationFormatError
            If the input string does not match the expected format.
        """
        pattern = r'(?:(\d+)D)?\s*(?:(\d+)H)?\s*(?:(\d+)M)?\s*(?:(\d+)S)?'
        try:
            match = re.fullmatch(pattern, self._duration.strip())
            if not match:
                raise DurationFormatError(f"Invalid duration format: '{self.duration}' (expected e.g. '3D 2H1M3S')")
        except Exception as e:
            raise DurationFormatError(f"Invalid duration format: '{self._duration}' (expected e.g. '3D 2H1M3S')") from e

        days, hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
        return days * 86400 + hours * 3600 + minutes * 60 + seconds

    def to_deltatime64(self) -> np.timedelta64:
        """
        Converts the duration string to a numpy.timedelta64 object.

        Returns
        -------
        numpy.timedelta64
            The duration as a Numpy timedelta64 object.
        """
        return np.timedelta64(self.to_seconds(), 's')


@dataclass
class Timer:
    start_time: str  # Accept string input like '1970-01-01 00:00:00'
    time_step: str  # Accept string input like '3D 2H1M3S'
    _current: np.datetime64 = field(init=False)
    _start_time_np: np.datetime64 = field(init=False)
    _time_step_np: np.timedelta64 = field(init=False)

    def __post_init__(self):
        """Initialize current time from start_time string and time_step from duration string."""
        self._start_time_np = self._convert_to_datetime64(self.start_time)
        self._time_step_np = self._convert_duration_to_timedelta64(self.time_step)
        self._current = self._start_time_np

    def _convert_to_datetime64(self, date_str: str) -> np.datetime64:
        """
        Convert date in str format to numpy.datetime64 enforcing 'YYYY-MM-DD hh:mm:ss'.
        """
        if not re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', date_str):
            raise DateFormatError(f"date string '{date_str}' does not match required format 'YYYY-MM-DD hh:mm:ss'")
        return np.datetime64(date_str, 's')

    def _convert_duration_to_timedelta64(self, duration_str: str) -> np.timedelta64:
        """
        Convert duration string to numpy.timedelta64 using Duration class.
        """
        duration = Duration(duration_str)
        return duration.to_deltatime64()

    @property
    def current(self) -> np.datetime64:
        """
        Returns the current time as a numpy.datetime64 object.
        """
        return self._current

    @current.setter
    def current(self, value: np.datetime64) -> None:
        """
        Sets the current time as a numpy.datetime64 object.

        Parameters
        ----------
        value : np.datetime64
            The new current time.
        """
        if not isinstance(value, np.datetime64):
            raise TypeError(f"Expected 'current' to be a numpy.datetime64, got {type(value).__name__}")
        self._current = value

    @property
    def next(self) -> np.datetime64:
        """
        Returns the next time as current + time_step.
        """
        return self._current + self._time_step_np

    def advance(self) -> None:
        """
        Advance the current time by one time step.
        """
        self._current = self.next


@dataclass
class Time:
    """
    Class representing time parameters in the simulation.

    Attributes
    ----------
    reference_date : str
        The reference date as string in format 'YYYY-MM-DD hh:mm:ss'
    start_time : str
        The simulation start time as string in format 'YYYY-MM-DD hh:mm:ss'.
    duration : str
        The simulation duration as a string.
    """

    # reference_date and start_time should be a str,
    # the class will do the transformation into a numpy.datetime64 object
    reference_date: str = field(default='1970-01-01 00:00:00')  # TODO: I thinks this is not necessary here.
    start: str = field(default='1970-01-01 00:00:00')
    duration: Duration = field(default_factory=lambda: Duration('3D 2H1M3S'))

    @property
    def end(self) -> np.datetime64:
        """
        Returns the simulation end-time as a numpy.datetime64 object.
        """
        start_datetime = self._convert_to_datetime64(self.start)
        duration_timedelta = self.duration.to_deltatime64()
        return start_datetime + duration_timedelta

    def _convert_to_datetime64(self, date_str: str) -> np.datetime64:
        """
        Convert date in str format to numpy.datetime64 enforcing 'YYYY-MM-DD hh:mm:ss'.
        """
        if not re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', date_str):
            raise DateFormatError(f"date string '{date_str}' does not match required format 'YYYY-MM-DD hh:mm:ss'")
        return np.datetime64(date_str, 's')
