"""
Time related classes used internally to represent simulation time.
"""

from dataclasses import dataclass, field
import numpy as np
import re
import isodate

from sedtrails.exceptions.exceptions import DateFormatError, DurationFormatError, ZeroDuration


def convert_duration_string_to_seconds(duration_str: str) -> int:
    """
    Parse an ISO 8601 duration string into the total number of seconds.

    Parameters
    ----------
    duration_str : str
        Duration string in ISO 8601 format (e.g., 'P1Y6M15DT12H25M30S', 'PT30S', 'P1Y').

    Returns
    -------
    int
        Total duration in seconds.

    Raises
    ------
    DurationFormatError
        If the input string does not match ISO 8601 duration format.
    """
    try:
        duration = isodate.parse_duration(duration_str)
        # Convert to total seconds
        return int(duration.total_seconds())
    except (isodate.ISO8601Error, ValueError) as e:
        raise DurationFormatError(
            f"Invalid ISO 8601 duration format: '{duration_str}' "
            "(expected e.g. 'P1Y6M15DT12H25M30S', 'PT30S', 'P1Y')"
        ) from e


def convert_datetime_string_to_datetime64(datetime_str: str) -> np.datetime64:
    """
    Convert date in str format to numpy.datetime64 enforcing 'YYYY-MM-DD hh:mm:ss'.

    Parameters
    ----------
    datetime_str : str
        Date string in format 'YYYY-MM-DD hh:mm:ss'.

    Returns
    -------
    np.datetime64
        The datetime as a numpy.datetime64 object with second precision.

    Raises
    ------
    DateFormatError
        If the input string does not match the required format.
    """
    if not re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', datetime_str):
        raise DateFormatError(f"date string '{datetime_str}' does not match required format 'YYYY-MM-DD hh:mm:ss'")
    return np.datetime64(datetime_str, 's')


class Duration:
    """
    Class representing a duration in the simulation using ISO 8601 format.

    Attributes
    ----------
    _duration : str
        The duration as a string in ISO 8601 format.

    Properties
    ----------
    string : str
        Returns the duration string.
    seconds : int
        Returns the duration in seconds.
    deltatime : np.timedelta64
        Returns the duration as a numpy.timedelta64 object.
    """

    def __init__(self, duration: str):
        """
        Initialize a Duration object.

        Parameters
        ----------
        duration : str
            Duration string in ISO 8601 format (e.g., 'P1Y6M15DT12H25M30S', 'PT30S', 'P1Y').
        """
        self._duration = duration

    @property
    def string(self) -> str:
        """
        Returns the duration string.

        Returns
        -------
        str
            The original duration string in ISO 8601 format.
        """
        return self._duration

    @property
    def seconds(self) -> int:
        """
        Returns the duration in seconds.

        Returns
        -------
        int
            Total duration converted to seconds.

        Raises
        ------
        DurationFormatError
            If the duration string format is invalid.
        """
        return convert_duration_string_to_seconds(self._duration)

    @property
    def deltatime(self) -> np.timedelta64:
        """
        Converts the duration string to a numpy.timedelta64 object.

        Returns
        -------
        numpy.timedelta64
            The duration as a Numpy timedelta64 object.
        """
        return np.timedelta64(self.seconds, 's')


@dataclass
class Time:
    """
    Class representing time parameters in the simulation.

    Attributes
    ----------
    _start : str
        The simulation start time as string in format 'YYYY-MM-DD hh:mm:ss'.
    time_step : Duration
        The simulation time step as a Duration object. Defaults to Duration('PT1H').
    duration : Duration
        The simulation duration as a Duration object. Defaults to Duration('P3DT2H1M3S').
    reference_date : str
        The reference date as string in format 'YYYY-MM-DD hh:mm:ss'.
        Defaults to UTC epoch '1970-01-01 00:00:00'.
    _start_time_np : np.datetime64
        Internal numpy datetime64 representation of the start time.
    """

    # reference_date and start_time should be a str,
    # the class will do the transformation into a numpy.datetime64 object
    _start: str
    time_step: Duration = field(default_factory=lambda: Duration('PT1H'), init=True)
    duration: Duration = field(default_factory=lambda: Duration('P3DT2H1M3S'), init=True)
    reference_date: str = field(default='1970-01-01 00:00:00')
    _start_time_np: np.datetime64 = field(init=False)

    def __post_init__(self):
        """
        Validates time_step and duration aren't zero and initializes internal datetime representation.

        Raises
        ------
        ZeroDuration
            If time_step or duration is zero length.
        DateFormatError
            If the start time string format is invalid.
        """
        if self.time_step.seconds == 0:
            raise ZeroDuration('time_step cannot be of length zero')
        if self.duration.seconds == 0:
            raise ZeroDuration('duration cannot be of length zero')

        # Convert start time to numpy.datetime64
        self._start_time_np = convert_datetime_string_to_datetime64(self._start)

    @property
    def start(self) -> int:
        """Returns the simulation start time as an integer representing seconds since the reference date."""
        start_datetime = convert_datetime_string_to_datetime64(self._start)
        reference_datetime = convert_datetime_string_to_datetime64(self.reference_date)
        # Calculate the difference in seconds
        delta_seconds = (start_datetime - reference_datetime).astype('timedelta64[s]').astype(int)
        return delta_seconds

    @start.setter
    def start(self, value: str) -> None:
        """
        Sets the simulation start time from a string in format 'YYYY-MM-DD hh:mm:ss'.

        Parameters
        ----------
        value : str
            The new start time as a string.

        Raises
        ------
        DateFormatError
            If the date string does not match the required format.
        """
        if not re.match(r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$', value):
            raise DateFormatError(f"date string '{value}' does not match required format 'YYYY-MM-DD hh:mm:ss'")
        self._start = value

    @property
    def end(self) -> int:
        """
        Returns the simulation end-time in seconds since reference date.

        Returns
        -------
        int
            The simulation end time as seconds since the reference date.
        """
        return self.start + self.duration.seconds


@dataclass
class Timer:
    """
    Class representing a timer for simulation time management.

    Manages the current simulation time and provides methods to advance
    through the simulation timeline.

    Attributes
    ----------
    simulation_time : Time
        The Time object containing simulation time parameters.
    _current : int | float
        Current time in seconds since reference date (internal field).
    _start_time_np : np.datetime64
        Internal numpy datetime64 representation (unused, kept for compatibility).
    _time_step_np : np.timedelta64
        Internal numpy timedelta64 representation (unused, kept for compatibility).

    Properties
    ----------
    current : int | float
        Returns/sets the current time in seconds since reference time.
    next : int | float
        Returns the next time step as current + time_step.
    steps : int
        Returns the total number of time steps in the simulation.

    Methods
    -------
    advance()
        Advance the current time by one time step.
    """

    simulation_time: Time
    _current: int | float = field(init=False)  # in seconds
    _start_time_np: np.datetime64 = field(init=False)
    _time_step_np: np.timedelta64 = field(init=False)
    stop: bool = False

    def __post_init__(self):
        """
        Initialize current time from the simulation start time.

        Sets the current time to the simulation start time in seconds
        since the reference date.
        """
        self._current = self.simulation_time.start

    @property
    def current(self) -> int | float:
        """
        Returns the current time in seconds since reference time.
        """
        return self._current

    @current.setter
    def current(self, value: int | float) -> None:
        """
        Sets a value for the current time.

        Parameters
        ----------
        value : int | float
            The new current time in seconds since the reference date.

        Raises
        ------
        TypeError
            If value is not an integer or float.
        """
        if not isinstance(value, (int, float)):
            raise TypeError(f"Expected 'current' to be integer or float, got {type(value).__name__}")
        self._current = value

    @property
    def next(self) -> int | float:
        """
        Returns the next time as current + time_step in seconds.

        Returns
        -------
        int | float
            The next time step value in seconds since reference date.
        """
        return self._current + self.simulation_time.time_step.seconds

    @property
    def steps(self) -> int:
        """
        Returns the number of time steps in the simulation.

        Calculates the total number of time steps by dividing the simulation
        duration by the time step size, rounded down to the nearest integer.

        Returns
        -------
        int
            The total number of time steps in the simulation.
        """
        import math

        duration_seconds = self.simulation_time.duration.seconds
        time_step_seconds = self.simulation_time.time_step.seconds
        return math.floor(duration_seconds / time_step_seconds)

    def advance(self) -> None:
        """
        Advance the current time by one time step.

        Moves the current time forward by one time step if it doesn't
        exceed the simulation end time.

        Raises
        ------
        ValueError
            If advancing would exceed the simulation end time.
        """

        if self.stop:
            raise RuntimeWarning('Timer has stopped. Cannot advance time.')

        if self.next <= self.simulation_time.end:
            self._current = self.next
        else:
            self.stop = True