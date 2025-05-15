"""
Unit tests for data classes in the particle.py module of the sedtrails package.
"""

import pytest
import numpy as np
from sedtrails.particle_tracer.particle import Mud, Sand, Passive, Particle, Time


class TestParticle:
    """
    Test suite for the Particle class and its children.
    """

    def test_mud_initialization(self):
        """
        Test the initialization of the Mud class.
        """

        mud = Mud(id=1, name='Mud Particle', _x=1, _y=2)
        assert isinstance(mud, Mud)

    def test_sand_initialization(self):
        """
        Test the initialization of the Sand class.
        """
        sand = Sand(id=1, name='Sand Particle', _x=1, _y=2)
        assert isinstance(sand, Sand)

    def test_passive_initialization(self):
        """
        Test the initialization of the Passive class.
        """
        passive = Passive(id=1, name='Passive Particle', _x=1, _y=2)
        assert isinstance(passive, Passive)

    def test_particle_name_type_error(self):
        """
        Test the type error for invalid name in Particle class.
        """

        with pytest.raises(TypeError):
            Particle(name=123)

    def test_particle_trace_type_error(self):
        """
        Test the type error for invalid trace in Particle class.
        """
        with pytest.raises(TypeError):
            Particle(name='Particle', trace=123)

    def test_name_property_is_optional(self):
        """
        Test that the name property is optional in the Particle class, and
        the default value is an empty string.
        """
        particle = Sand(id=1, _x=1, _y=2)
        assert particle.name == ''

        particle = Mud(id=1, _x=1, _y=2)
        assert particle.name == ''

        particle = Passive(id=1, _x=1, _y=2)
        assert particle.name == ''


class TestMud:
    """
    Test suite for the Mud class.
    """

    # TODO: implement tests for Mud class


class TestSand:
    """
    Test suite for the Sand class.
    """

    # TODO: implement tests for Mud class


class TestPassive:
    """
    Test suite for the Passive class.
    """

    # TODO: implement tests for Mud class


class TestInterpolatedValue:
    """
    Test suite for the InterpolatedValue class.
    """


class TestPhysics:
    """
    Test suite for the Physics class.
    """

class TestTime:
    """
    Test suite for the Time class.
    """

    def test_initial_current_time(self):
        """
        Test that the initial current time matches the reference date.
        """
        reference_date = "2023-01-01"
        offset_seconds = np.timedelta64(0, 's')
        time_instance = Time(reference_date=reference_date, offset_seconds=offset_seconds)
        expected = np.datetime64("2023-01-01T00:00:00", 's')
        assert actual == expected, 
            f"Initial current time: expected={expected}, actual={actual}"

    def test_update_add_seconds(self):
        """
        Test updating the current time by adding seconds.
        """
        reference_date = "2023-01-01"
        offset_seconds = np.timedelta64(0, 's')
        time_instance = Time(reference_date=reference_date, offset_seconds=offset_seconds)
        delta_seconds = np.timedelta64(120, 's')
        time_instance.update(delta_seconds)
        expected = np.datetime64("2023-01-01T00:02:00", 's')
        assert time_instance.get_current_time() == expected,
            f"Updated current time should be {expected}, but got {time_instance.get_current_time()}."

    def test_update_subtract_seconds(self):
        """
        Test updating the current time by subtracting seconds.
        """
        reference_date = "2023-01-01"
        offset_seconds = np.timedelta64(120, 's')
        time_instance = Time(reference_date=reference_date, offset_seconds=offset_seconds)
        delta_seconds = np.timedelta64(-60, 's')
        time_instance.update(delta_seconds)
        expected = np.datetime64("2023-01-01T00:01:00", 's')
        assert time_instance.get_current_time() == expected,
            f"Updated current time after subtraction should be {expected}, but got {time_instance.get_current_time()}."

    def test_get_seconds_since_reference(self):
        """
        Test retrieving the number of seconds since the reference date.
        """
        reference_date = "2023-01-01"
        offset_seconds = np.timedelta64(90, 's')
        time_instance = Time(reference_date=reference_date, offset_seconds=offset_seconds)
        actual = time_instance.get_seconds_since_reference()
        expected = 90
        assert actual == expected, 
            f"Seconds since reference: expected={expected}, actual={actual}"
