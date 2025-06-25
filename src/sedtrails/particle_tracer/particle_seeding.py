"""
Particle Seeding Tool
=====================
Computes initial particle positions (x,y) and  particle's release times (t)
using various particle release strategies.
Seeding strategies for positions include:
- At location: Release particles at a specific location (x,y).
- Regular grid: Release particles in a regular grid pattern based
    on the specified grid size (distance between particles), and the simulation
    domain size. A mask can be applied to restrict the area of seeding.
- Linear: release particle along a line between two points (x1,y1) and (x2,y2), and
    spaced by a specified distance.
"""

from abc import ABC, abstractmethod
from sedtrails.particle_tracer.particle import Particle, Passive
from typing import List, Tuple
import random


class SeedingStrategy(ABC):
    """
    Abstract base class for seeding strategies.
    """

    def __init__(self, **config):
        self.config = config

    @abstractmethod
    def seed(self, initial_positions: List[tuple] | tuple, release_times: List | str, count: int = 1) -> list[Particle]:
        """
        Seed particles at the specified location.

        Parametersc
        ----------
        initial_positions : list[tuple] | tuple
            List of tuples containing the (x,y) coordinates of the particles.
            If a single tuple is provided, it will be used for all particles.
        count : int
            Number of particles to seed. Default is 1.

        Returns
        -------
        list[Particle]
            List of seeded particles.
        """
        pass

    def _validate_initial_positions(self, initial_positions: List[tuple] | tuple, count: int) -> bool:
        """
        Validate the initial positions and count of particles.

        Parameters
        ----------
        initial_positions : list[tuple] | tuple
            List of tuples containing the (x,y) coordinates of the particles.
            If a single tuple is provided, it will be used for all particles.
        count : int
            Number of particles to seed. Default is 1.
        release_times : list | int
            List of release times for each particle. Times must refer to simulation time.
            If a single integer is provided, it will be used for all particles.

        Returns
        -------
        bool
            True if the initial positions are valid, False otherwise.

        Raises
        ------
        TypeError
            If initial_positions is not a list or tuple.
        ValueError
            If the number of initial positions does not match the count.
        """

        # TODO: we should add a validation check to ensure that the initial positions all lie within the model domain
        # (i.e. to make sure we are placing particles somewhere where they will actually move)

        validity = True
        if not isinstance(initial_positions, (list, tuple)):
            validity = False
            raise TypeError(
                f"Expected 'initial_positions' to be a list or tuple, got {type(initial_positions).__name__}"
            )
        if isinstance(initial_positions, list) and len(initial_positions) != count:
            validity = False
            raise ValueError(
                f'Number of initial positions ({len(initial_positions)}) does not match the count ({count}).'
            )
        if isinstance(initial_positions, tuple) and count != 1:
            validity = False
            raise ValueError(f'Number of initial positions ({initial_positions}) does not match the count ({count}).')
        return validity

    def _validate_release_times(self, release_times: List | int, count: int):
        """
        Validate the release times and count of particles.

        Parameters
        ----------
        release_times : list | int
            List of release times for each particle. Times must refer to simulation time.
            If a single integer is provided, it will be used for all particles.
        count : int
            Number of particles to seed.

        Raises
        ------
        TypeError
            If release_times is not a list or int.
        ValueError
            If the number of release times does not match the count.
        """
        validity = True
        if isinstance(release_times, int):
            release_times = [release_times] * count
        elif not isinstance(release_times, list):
            validity = False
            raise TypeError(f"Expected 'release_times' to be a list or int, got {type(release_times).__name__}")
        if len(release_times) != count:
            validity = False
            raise ValueError(f'Number of release times ({len(release_times)}) does not match the count ({count}).')
        return validity  # List to store the particles


class XYSeeding(SeedingStrategy):
    """
    Seeding strategy to release particles at a specific location (x,y).
    """

    def seed(self, initial_positions: List[tuple] | tuple, release_times: List | str, count: int = 1) -> list[Particle]:
        """
        Seed particles at the specified location.

        Parameters
        ----------
        initial_positions : list[tuple] | tuple
            List of tuples containing the (x,y) coordinates of the particles.
            If a single tuple is provided, it will be used for all particles.
        count : int
            Number of particles to seed. Default is 1.
        release_times : list | int
            List of release times for each particle. Times must refer to simulation time.
            If a single integer is provided, it will be used for all particles.

        Returns
        -------
        list[Particle]
            List of seeded particles.
        """

        # Validate the initial positions and realise times
        self._validate_initial_positions(initial_positions, count)
        self._validate_release_times(release_times, count)

        particles = []  # List to store the particles
        if isinstance(initial_positions, tuple):  # If a single tuple is provided, use it for all particles
            initial_positions = [initial_positions] * count

        if isinstance(release_times, int):  # If a single integer is provided, use it for all particles
            release_times = [release_times] * count
        for i in range(count):
            # Create a particle object with the initial position and release time

            particle = Passive(
                id=i, _x=initial_positions[i][0], _y=initial_positions[i][1], _release_time=release_times[i]
            )
            particles.append(particle)

        return particles


class RandomSeeding(SeedingStrategy):
    """
    Seeding strategy to release particles at at random locations (x,y).
    """

    def seed(
        self, x_range: Tuple[float, float], y_range: Tuple[float, float], release_times: List[str] | str, count: int = 1
    ) -> list[Particle]:
        """
        Seed particles at the specified location.

        Parameters
        ----------
        count : int
            Number of particles to seed. Default is 1.
        release_times : list | int
            List of release times for each particle. Times must refer to simulation time.
            If a single integer is provided, it will be used for all particles.

        Returns
        -------
        list[Particle]
            List of seeded particles.
        """

        # Validate the initial positions and realise times
        if isinstance(release_times, list):
            if len(release_times) != count:
                raise ValueError(f'Number of release times ({len(release_times)}) does not match the count ({count}).')

        def random_position(x_range: float, y_range: float) -> Tuple[float, float]:
            """
            Generate a random position within the specified ranges.
            """
            x = random.uniform(x_range[0], x_range[1])
            y = random.uniform(y_range[0], y_range[1])
            return x, y

        particles = []  # List to store the particles

        if isinstance(
            release_times, int
        ):  # generate a list of release times when the same time should be used for all particles
            release_times = [release_times] * count
        for i in range(count):
            initial_position = random_position(x_range, y_range)
            # TODO: find if it is needed to round the position to a specific number of decimal places
            particle = Passive(id=i, _x=initial_position[0], _y=initial_position[1], _release_time=release_times[i])
            particles.append(particle)

        return particles


class GridSeeding(SeedingStrategy):
    """
    Seeding strategy to release particles at a regular grid pattern.
    """

    def seed(self, spatial_domain, count: int = 1, release_times: List | int = 1, mask=None) -> list[Particle]:
        # TODO: agree ont he data type for spatial_domain and mask

        """
        Seed particles in a regular grid pattern based on the specified grid size (distance between particles),
        and the simulation domain size. A mask can be applied to restrict the area of seeding.

        Parameters
        ----------
        spatial_domain :
            The simulation domain size (width, height).
        count : int
            Number of particles to seed. Default is 1.
        release_times : list | int
            List of release times for each particle. Times must refer to simulation time.
            If a single integer is provided, it will be used for all particles. Default is 1.
        mask :
            Optional mask to restrict the area of seeding. Default is None.
            If None, particles will be seeded in the entire domain.

        Returns
        -------
        list[Particle]
            List of seeded particles.
        """

        particles = []  # List to store the particles

        # TODO: Add logic to create a list of particles objects using the Particle class and the 'count', and
        # assign the initial position (x,y) to each particle, and the release time (t) to each particle.

        return particles


if __name__ == '__main__':
    pos = [(1, 2), (3, 4), (5, 6)]
    release_times = [0, 1, 2]
    count = 3
    xy_seeding = XYSeeding()
    particles = xy_seeding.seed(initial_positions=pos, release_times=release_times, count=count)
    particles_single = xy_seeding.seed(initial_positions=(1, 2), release_times=0, count=1)
    print(particles)

    # Example usage of the RandomSeeding class
    random_seeding = RandomSeeding()
    x_range = (0, 10)
    y_range = (0, 100)
    release_times = [0, 1, 2]
    count = 3
    particles = random_seeding.seed(x_range=x_range, y_range=y_range, release_times=release_times, count=count)
    print(particles)
