"""
Unit tests for the Particle class.
"""

from sedtrails.particle_tracer.particle import Mud, Sand, Passive


def test_mud_initialization():
    """
    Test the initialization of the Mud class.
    """
    
    mud = Mud(name="Mud Particle", particle_velocity=0.5)
    assert isinstance(mud, Mud)


def test_sand_initialization():
    """
    Test the initialization of the Sand class.
    """
    sand = Sand(name="Sand Particle", particle_velocity=0.7)
    assert isinstance(sand, Sand)


def test_passive_initialization():  
    """
    Test the initialization of the Passive class.
    """
    passive = Passive(name="Passive Particle", particle_velocity=0.9)
    assert isinstance(passive, Passive)
  