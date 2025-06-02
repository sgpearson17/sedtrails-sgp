"""
Unit tests for the Time class in the time.py module of the sedtrails package.
"""

import numpy as np
from sedtrails.particle_tracer.time import Time


class TestTime:
    """
    Test suite for the Time class.
    """

    def test_initial_current_time(self):
        """
        Test that the initial current time matches the reference date.
        """
        reference_date = "2025-05-20 00:00:00"
        start_time = "2025-05-20 00:00:00"
        time_step = "30S"
        duration = "1H"
        time_instance = Time(reference_date=reference_date, start_time=start_time, time_step=time_step, duration=duration)
        expected = np.datetime64("2025-05-20T00:00:00", 's')
        actual = time_instance.get_current_time()
        assert actual == expected, f'Initial current time: expected={expected}, actual={actual}'

    def test_get_current_time_with_steps(self):
        """
        Test getting the current time at a specific simulation step.
        """
        reference_date = "2025-05-20 00:00:00"
        start_time = "2025-05-20 00:00:00"
        time_step = "30S"
        duration = "1H"
        time_instance = Time(reference_date=reference_date, start_time=start_time, time_step=time_step, duration=duration)
        # At step 2, should be 60 seconds after start
        expected = np.datetime64('2025-05-20T00:01:00', 's')
        actual = time_instance.get_current_time(step=2)
        assert actual == expected, f'Current time at step 2: expected={expected}, actual={actual}'

    def test_get_current_time_with_start_time(self):
        """
        Test getting the current time with a nonzero start_time.
        """
        reference_date = "2025-05-20 00:00:00"
        start_time = "2025-05-20 00:01:30"
        time_step = "30S"
        duration = "1H"
        time_instance = Time(reference_date=reference_date, start_time=start_time, time_step=time_step, duration=duration)
        # At step 1, should be 90 + 30 = 120 seconds after reference
        expected = np.datetime64('2025-05-20T00:02:00', 's')
        actual = time_instance.get_current_time(step=1)
        assert actual == expected, f"Current time with start_time and step: expected={expected}, actual={actual}"

    def test_end_time(self):
        """
        Test the calculation of the simulation end time.
        """
        reference_date = "2025-05-20 00:00:00"
        start_time = "2025-05-20 00:00:00"
        time_step = "30S"
        duration = "1H"
        time_instance = Time(reference_date=reference_date, start_time=start_time, time_step=time_step, duration=duration)
        expected = np.datetime64("2025-05-20T01:00:00", 's')
        actual = time_instance.end_time
        assert actual == expected, f"End time: expected={expected}, actual={actual}"
