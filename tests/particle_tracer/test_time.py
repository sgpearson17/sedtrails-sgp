"""
Unit tests for the Time class in the time.py module of the sedtrails package.
"""

import pytest
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
        reference_date = "2025-05-20"
        start_time = 0
        time_instance = Time(reference_date=reference_date, start_time=start_time)
        expected = np.datetime64("2025-05-20 00:00:00", 's')
        actual = time_instance.get_current_time()
        assert actual == expected, f"Initial current time: expected={expected}, actual={actual}"

    def test_update_add_seconds(self):
        """
        Test updating the current time by adding seconds.
        """
        reference_date = "2025-05-20"
        start_time = 0
        time_instance = Time(reference_date=reference_date, start_time=start_time)
        delta_seconds = 120
        time_instance.update(delta_seconds)
        expected = np.datetime64("2025-05-20 00:02:00", 's')
        actual = time_instance.get_current_time()
        assert actual == expected, f"After adding 120s: expected={expected}, actual={actual}"
        
    def test_update_subtract_seconds(self):
        """
        Test updating the current time by subtracting seconds.
        """
        reference_date = "2025-05-20"
        start_time = 120
        time_instance = Time(reference_date=reference_date, start_time=start_time)
        delta_seconds = -60
        time_instance.update(delta_seconds)
        expected = np.datetime64("2025-05-20 00:01:00", 's')
        actual = time_instance.get_current_time()
        assert actual == expected, f"After subtracting 60s: expected={expected}, actual={actual}"

    def test_get_seconds_since_reference(self):
        """
        Test retrieving the number of seconds since the reference date.
        TODO: will finish when the refence_date is clarified
        """
        pass