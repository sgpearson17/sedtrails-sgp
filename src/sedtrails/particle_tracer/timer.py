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


@dataclass
class Time:
    """
    Class representing time in the simulation.

    Attributes
    ----------
    reference_date : str
        The reference date as string in format 'YYYY-MM-DD hh:mm:ss'
    start_time : str
        The simulation start time as string in format 'YYYY-MM-DD hh:mm:ss'.
    duration : str
        The simulation duration as a string.
    time_step : str
        The simulation time step as a string in format '3D 2H1M3S'.

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
    duration: str = field(default='3D 2H1M3S')
    time_step: str = field(default='30S')
    _reference_date_np: np.datetime64 = field(init=False)
    _start_time_np: np.datetime64 = field(init=False)
    _duration_seconds: int = field(init=False)
    _duration_timedelta: np.timedelta64 = field(init=False)
    _time_step_seconds: int = field(init=False)
    _time_step_timedelta: np.timedelta64 = field(init=False)

    def __post_init__(self):
        self._reference_date_np = self._convert_to_datetime64(self.reference_date)
        self._start_time_np = self._convert_to_datetime64(self.start_time)
        self._duration_seconds = self._parse_duration_to_seconds(self.duration)
        self._duration_timedelta = np.timedelta64(self._duration_seconds, 's')
        self._time_step_seconds = self._parse_duration_to_seconds(self.time_step)
        self._time_step_timedelta = np.timedelta64(self._time_step_seconds, 's')

    def _convert_to_datetime64(self, date_str: str) -> np.datetime64:
        """
        Convert date in str format to numpy.datetime64 enforcing 'YYYY-MM-DD hh:mm:ss'.
        """
        if not re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', date_str):
            raise DateFormatError(f"date string '{date_str}' does not match required format 'YYYY-MM-DD hh:mm:ss'")
        return np.datetime64(date_str, 's')

    @staticmethod
    def _parse_duration_to_seconds(duration_str: str) -> int:
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
            match = re.fullmatch(pattern, duration_str.strip())
            if not match:
                raise DurationFormatError(f"Invalid duration format: '{duration_str}' (expected e.g. '3D 2H1M3S')")
        except Exception as e:
            raise DurationFormatError(f"Invalid duration format: '{duration_str}' (expected e.g. '3D 2H1M3S')") from e

        days, hours, minutes, seconds = (int(g) if g else 0 for g in match.groups())
        return days * 86400 + hours * 3600 + minutes * 60 + seconds

    @property
    def end_time(self) -> np.datetime64:
        """
        Returns the simulation end time as a numpy.datetime64 object.
        """
        return self._start_time_np + self._duration_timedelta

    def get_current_time(self, step: int = 0) -> np.datetime64:
        """
        Returns the current time in the simulation as a numpy.datetime64 object,
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
        return self._start_time_np + step * self._time_step_timedelta
