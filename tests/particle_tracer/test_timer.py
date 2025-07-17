"""
Unit tests for the Timer class in the time.py module of the sedtrails package.
"""

import pytest
import numpy as np
from sedtrails.particle_tracer.timer import Duration, Timer, Time
from sedtrails.exceptions.exceptions import DateFormatError, DurationFormatError


class TestDuration:
    """Test cases for the Duration class."""

    def test_duration_init(self):
        """Test Duration initialization."""
        duration = Duration('3D 2H1M3S')
        assert duration.duration == '3D 2H1M3S'

    def test_to_seconds_full_format(self):
        """Test conversion of full duration format to seconds."""
        duration = Duration('3D 2H1M3S')
        expected = 3 * 86400 + 2 * 3600 + 1 * 60 + 3  # 266463 seconds
        assert duration.to_seconds() == expected

    def test_to_seconds_partial_formats(self):
        """Test conversion of partial duration formats to seconds."""
        test_cases = [
            ('1D', 86400),
            ('2H', 7200),
            ('30M', 1800),
            ('45S', 45),
            ('1H 30M', 5400),
            ('2D 5H', 2 * 86400 + 5 * 3600),
            ('1M 30S', 90),
        ]

        for duration_str, expected_seconds in test_cases:
            duration = Duration(duration_str)
            assert duration.to_seconds() == expected_seconds

    def test_to_seconds_with_spaces(self):
        """Test duration parsing with various spacing."""
        test_cases = [
            ('1D 2H 3M 4S', 1 * 86400 + 2 * 3600 + 3 * 60 + 4),
            ('1D2H3M4S', 1 * 86400 + 2 * 3600 + 3 * 60 + 4),
            ('  1D  2H  3M  4S  ', 1 * 86400 + 2 * 3600 + 3 * 60 + 4),
        ]

        for duration_str, expected_seconds in test_cases:
            duration = Duration(duration_str)
            assert duration.to_seconds() == expected_seconds

    def test_to_seconds_zero_duration(self):
        """Test conversion of zero/empty duration."""
        duration = Duration('')
        # Empty string should return 0 seconds as all groups are optional
        assert duration.to_seconds() == 0

    def test_to_seconds_invalid_format(self):
        """Test invalid duration formats raise DurationFormatError."""
        invalid_durations = [
            'invalid',
            '1X',
            '1D 2X',
            '1.5D',
            '-1D',
        ]

        for invalid_duration in invalid_durations:
            duration = Duration(invalid_duration)
            with pytest.raises(DurationFormatError):
                duration.to_seconds()

    def test_to_seconds_valid_edge_cases(self):
        """Test edge cases that should be valid."""
        # These should work fine
        valid_cases = [
            ('1D 25H', 1 * 86400 + 25 * 3600),  # 25 hours is valid
            ('0D', 0),
            ('0H', 0),
            ('0M', 0),
            ('0S', 0),
        ]

        for duration_str, expected_seconds in valid_cases:
            duration = Duration(duration_str)
            assert duration.to_seconds() == expected_seconds

    def test_to_deltatime64(self):
        """Test conversion to numpy timedelta64."""
        duration = Duration('1D 2H3M4S')
        timedelta = duration.to_deltatime64()
        expected_seconds = 1 * 86400 + 2 * 3600 + 3 * 60 + 4
        expected_timedelta = np.timedelta64(expected_seconds, 's')
        assert timedelta == expected_timedelta


class TestTimer:
    """Test cases for the Timer class."""

    def test_timer_init(self):
        """Test Timer initialization."""
        timer = Timer(start_time='2023-01-01 12:00:00', time_step='1H')
        assert timer.start_time == '2023-01-01 12:00:00'
        assert timer.time_step == '1H'
        assert timer.current == np.datetime64('2023-01-01 12:00:00', 's')

    def test_timer_invalid_date_format(self):
        """Test Timer with invalid date format raises DateFormatError."""
        invalid_dates = [
            '2023-1-1 12:00:00',  # Single digit month/day
            '23-01-01 12:00:00',  # Two digit year
            '2023-01-01 12:00',  # Missing seconds
            '2023/01/01 12:00:00',  # Wrong separator
            'invalid date',
        ]

        for invalid_date in invalid_dates:
            with pytest.raises(DateFormatError):
                Timer(start_time=invalid_date, time_step='1H')

    def test_timer_invalid_duration_format(self):
        """Test Timer with invalid duration format raises DurationFormatError."""
        with pytest.raises(DurationFormatError):
            Timer(start_time='2023-01-01 12:00:00', time_step='invalid')

    def test_current_property_getter(self):
        """Test current property getter."""
        timer = Timer(start_time='2023-01-01 12:00:00', time_step='1H')
        expected = np.datetime64('2023-01-01 12:00:00', 's')
        assert timer.current == expected

    def test_current_property_setter(self):
        """Test current property setter."""
        timer = Timer(start_time='2023-01-01 12:00:00', time_step='1H')
        new_time = np.datetime64('2023-01-02 15:30:00', 's')
        timer.current = new_time
        assert timer.current == new_time

    def test_current_property_setter_invalid_type(self):
        """Test current property setter with invalid type raises TypeError."""
        timer = Timer(start_time='2023-01-01 12:00:00', time_step='1H')
        with pytest.raises(TypeError):
            timer.current = '2023-01-02 15:30:00'

    def test_next_property(self):
        """Test next property returns current + time_step."""
        timer = Timer(start_time='2023-01-01 12:00:00', time_step='1H')
        expected = np.datetime64('2023-01-01 13:00:00', 's')
        assert timer.next == expected

    def test_next_property_complex_time_step(self):
        """Test next property with complex time step."""
        timer = Timer(start_time='2023-01-01 12:00:00', time_step='1D 2H30M')
        expected = np.datetime64('2023-01-02 14:30:00', 's')
        assert timer.next == expected

    def test_advance(self):
        """Test advance method updates current time."""
        timer = Timer(start_time='2023-01-01 12:00:00', time_step='1H')
        original_current = timer.current
        original_next = timer.next

        timer.advance()

        assert timer.current == original_next
        assert timer.current != original_current

    def test_multiple_advances(self):
        """Test multiple advances work correctly."""
        timer = Timer(start_time='2023-01-01 12:00:00', time_step='30M')

        # First advance
        timer.advance()
        assert timer.current == np.datetime64('2023-01-01 12:30:00', 's')

        # Second advance
        timer.advance()
        assert timer.current == np.datetime64('2023-01-01 13:00:00', 's')

        # Third advance
        timer.advance()
        assert timer.current == np.datetime64('2023-01-01 13:30:00', 's')


class TestTime:
    """Test cases for the Time class."""

    def test_time_default_init(self):
        """Test Time initialization with default values."""
        time = Time()
        assert time.reference_date == '1970-01-01 00:00:00'
        assert time.start == '1970-01-01 00:00:00'
        assert isinstance(time.duration, Duration)
        assert time.duration.duration == '3D 2H1M3S'

    def test_time_custom_init(self):
        """Test Time initialization with custom values."""
        custom_duration = Duration('5D 10H')
        time = Time(reference_date='2023-01-01 00:00:00', start='2023-01-15 12:00:00', duration=custom_duration)
        assert time.reference_date == '2023-01-01 00:00:00'
        assert time.start == '2023-01-15 12:00:00'
        assert time.duration == custom_duration

    def test_end_property(self):
        """Test end property calculation."""
        duration = Duration('2D 12H')
        time = Time(start='2023-01-01 12:00:00', duration=duration)
        expected_end = np.datetime64('2023-01-04 00:00:00', 's')
        assert time.end == expected_end

    def test_end_property_complex_duration(self):
        """Test end property with complex duration."""
        duration = Duration('1D 2H30M45S')
        time = Time(start='2023-06-15 09:15:30', duration=duration)
        expected_end = np.datetime64('2023-06-16 11:46:15', 's')
        assert time.end == expected_end

    def test_time_invalid_start_date(self):
        """Test Time with invalid start date format."""
        with pytest.raises(DateFormatError):
            time = Time(start='invalid-date')
            _ = time.end  # This should trigger the validation

    def test_time_invalid_reference_date(self):
        """Test Time with invalid reference date format."""
        time = Time(reference_date='invalid-date')
        # Note: reference_date validation only happens if it's used somewhere
        # Currently it's not used in the end property, so no error is raised
        assert time.reference_date == 'invalid-date'

    def test_duration_integration(self):
        """Test integration between Time and Duration classes."""
        duration = Duration('7D')
        time = Time(start='2023-12-25 00:00:00', duration=duration)
        expected_end = np.datetime64('2024-01-01 00:00:00', 's')
        assert time.end == expected_end


class TestIntegration:
    """Integration tests for Timer and Time classes working together."""

    def test_timer_with_time_parameters(self):
        """Test Timer using parameters from Time class."""
        time = Time(start='2023-01-01 00:00:00', duration=Duration('2D'))

        timer = Timer(start_time=time.start, time_step='6H')

        assert timer.current == np.datetime64('2023-01-01 00:00:00', 's')

        # Advance through the simulation
        timer.advance()  # 06:00:00
        timer.advance()  # 12:00:00
        timer.advance()  # 18:00:00
        timer.advance()  # 2023-01-02 00:00:00

        assert timer.current == np.datetime64('2023-01-02 00:00:00', 's')

        # Check if we're still within the simulation duration
        assert timer.current < time.end

    def test_timer_simulation_boundary(self):
        """Test timer advancing to simulation end boundary."""
        time = Time(start='2023-01-01 00:00:00', duration=Duration('1D'))

        timer = Timer(start_time=time.start, time_step='12H')

        # Should be able to advance twice to reach the end
        timer.advance()  # 12:00:00
        assert timer.current < time.end

        timer.advance()  # 2023-01-02 00:00:00
        assert timer.current == time.end
