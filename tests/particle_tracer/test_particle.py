"""
Unit tests for data classes in the particle.py module of the sedtrails package.
"""
import pytest
from sedtrails.particle_tracer.particle import Mud,Sand, Passive, \
      Particle


class TestParticle:
    """
    Test suite for the Particle class and its children.
    """

    def test_mud_initialization(self):
        """
        Test the initialization of the Mud class.
        """
        
        mud = Mud(name="Mud Particle", _x=1, _y=2, particle_velocity=0.5)
        assert isinstance(mud, Mud)

    def test_sand_initialization(self):
        """
        Test the initialization of the Sand class.
        """
        sand = Sand(name="Sand Particle", _x=1, _y=2, particle_velocity=0.7)
        assert isinstance(sand, Sand)

    def test_passive_initialization(self):  
        """
        Test the initialization of the Passive class.
        """
        passive = Passive(name="Passive Particle", _x=1, _y=2, particle_velocity=0.9)
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
            Particle(name="Particle", trace=123)


class TestMud:
    """
    Test suite for the Mud class.
    """
    #TODO: implement tests for Mud class

class TestSand:
    """
    Test suite for the Sand class.
    """

    #TODO: implement tests for Mud class


class TestPassive:
    """
    Test suite for the Passive class.
    """

    #TODO: implement tests for Mud class


class TestInterpolatedValue:
    """
    Test suite for the InterpolatedValue class.
    """

class TestPhysics:
    """
    Test suite for the Physics class.
    """
