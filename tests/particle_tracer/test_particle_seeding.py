"""
Unit tests for data classes in the particle.py module of the sedtrails package.
"""

import pytest
from sedtrails.particle_tracer.particle_seeding import XYSeeding, RandomSeeding
from sedtrails.particle_tracer.particle import Particle


@pytest.fixture
def xy_seeder():
    """
    Fixture for creating an instance of XYSeeding.
    """
    return XYSeeding()


@pytest.fixture
def random_seeder():
    """
    Fixture for creating an instance of RandomSeeding.
    """
    return RandomSeeding()


class TestXYSeeding:
    def test_seed_single_position(self, xy_seeder):
        """Test seeding a single particle at a specific position."""
        particles = xy_seeder.seed(initial_positions=(1.0, 2.0), release_times=1, count=1)
        assert len(particles) == 1
        for particle in particles:
            assert isinstance(particle, Particle)
            assert particle.x == 1.0
            assert particle.y == 2.0

    def test_seed_multiple_positions(self, xy_seeder):
        """Test seeding multiple particles at different positions."""
        particles = xy_seeder.seed(initial_positions=[(1.0, 2.0), (3.0, 4.0)], release_times=[1, 2], count=2)
        assert len(particles) == 2
        for i, particle in enumerate(particles):
            assert isinstance(particle, Particle)
            assert particle.x == [1.0, 3.0][i]
            assert particle.y == [2.0, 4.0][i]

    def test_seed_invalid_count(self, xy_seeder):
        """Test seeding with an invalid count."""

        with pytest.raises(ValueError):
            xy_seeder.seed(initial_positions=(1.0, 2.0), release_times=1, count=3)

    def test_seed_invalid_initial_positions(self):
        seeder = XYSeeding(initial_positions=(1.0, 2.0))
        with pytest.raises(TypeError):
            seeder.seed(count='invalid')


class TestRandomSeeding:
    def test_seed_single_positions(self, random_seeder):
        """Test seeding random particles."""
        """Test seeding a single particle at a specific position."""
        particles = random_seeder.seed(x_range=(0, 10), y_range=(0, 100), release_times=1, count=1)
        assert len(particles) == 1
        for particle in particles:
            assert isinstance(particle, Particle)

    def test_seed_invalid_count(self, random_seeder):
        """Test seeding with an invalid count."""
        with pytest.raises(TypeError):
            random_seeder.seed(x_range=(0, 10), y_range=(0, 100), release_times=1, count='uno')

    def test_seed_multiple_positions(self, random_seeder):
        """Test seeding multiple particles at different positions."""
        particles = random_seeder.seed(x_range=(0, 10), y_range=(0, 100), release_times=[1, 2], count=2)
        assert len(particles) == 2
        for particle in particles:
            assert isinstance(particle, Particle)
