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
        start_time = 0
        time_step = 30
        time_instance = Time(reference_date=reference_date, start_time=start_time, time_step=time_step)
        expected = np.datetime64("2025-05-20T00:00:00", 's')
        actual = time_instance.get_current_time()
        assert actual == expected, f"Initial current time: expected={expected}, actual={actual}"

    def test_get_current_time_with_steps(self):
        """
        Test getting the current time at a specific simulation step.
        """
        reference_date = "2025-05-20 00:00:00"
        start_time = 0
        time_step = 30
        time_instance = Time(reference_date=reference_date, start_time=start_time, time_step=time_step)
        # At step 2, should be 60 seconds after start
        expected = np.datetime64("2025-05-20T00:01:00", 's')
        actual = time_instance.get_current_time(step=2)
        assert actual == expected, f"Current time at step 2: expected={expected}, actual={actual}"

    def test_get_current_time_with_start_time(self):
        """
        Test getting the current time with a nonzero start_time.
        """
        reference_date = "2025-05-20 00:00:00"
        start_time = 90
        time_step = 30
        time_instance = Time(reference_date=reference_date, start_time=start_time, time_step=time_step)
        # At step 1, should be 90 + 30 = 120 seconds after reference
        expected = np.datetime64("2025-05-20T00:02:00", 's')
        actual = time_instance.get_current_time(step=1)
        assert actual == expected, f"Current time with start_time and step: expected={expected}, actual={actual}"


    def test_get_seconds_since_reference(self):
        """
        Test retrieving the number of seconds since the reference date.
        """
        reference_date = "2025-05-20 00:00:00"
        start_time = 90
        time_step = 30
        time_instance = Time(reference_date=reference_date, start_time=start_time, time_step=time_step)
        # At step 2, should be 90 + 2*30 = 150 seconds
        actual = time_instance.get_seconds_since_reference(delta_seconds=2*time_step)
        expected = 150
        assert actual == expected, f"Seconds since reference: expected={expected}, actual={actual}"

    def test_update_add_seconds(self):
        """
        Test updating the current time by adding seconds.
        """
        reference_date = "2025-05-20 00:00:00"
        start_time = 0
        time_instance = Time(reference_date=reference_date, start_time=start_time)
        delta_seconds = 120
        time_instance.update(delta_seconds)
        expected = np.datetime64("2025-05-20T00:02:00", 's')
        actual = time_instance.get_current_time()
        assert actual == expected, f"After adding 120s: expected={expected}, actual={actual}"
        
    def test_update_subtract_seconds(self):
        """
        Test updating the current time by subtracting seconds.
        """
        reference_date = "2025-05-20 00:00:00"
        start_time = 120
        time_instance = Time(reference_date=reference_date, start_time=start_time)
        delta_seconds = -60
        time_instance.update(delta_seconds)
        expected = np.datetime64("2025-05-20T00:01:00", 's')
        actual = time_instance.get_current_time()
        assert actual == expected, f"After subtracting 60s: expected={expected}, actual={actual}"

    def test_get_seconds_since_reference(self):
        """
        Test retrieving the number of seconds since the reference date in a 
        human readable format
        """
        reference_date = "2025-05-20 00:00:00"
        start_time = 90
        time_instance = Time(reference_date=reference_date, start_time=start_time)
        actual = time_instance.get_seconds_since_reference()
        expected = 90
        assert actual == expected, f"Current time string: expected={expected}, actual={actual}"