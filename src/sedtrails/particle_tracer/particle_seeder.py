"""
Particle Seeding Tool
=====================
Manage the creation of particles, their positions (x,y) and distribution.
using various release strategies.
Seeding strategies for positions include:
- At location: Release particles at a specific locations (x,y).
- Regular grid: Release particles in a regular grid pattern based
    on distances between particles in x and y directions, and the
    simulation domain size. A mask can be applied to restrict the area of seeding.
- Transect: release particle along line segments  defined by two points(x1,y1) and (x2,y2).
- Random: Release particles at random locations (x,y) within an area
    constrained by a bounding box (xmin, xmax, ymin, ymax).
"""

from abc import ABC, abstractmethod
from sedtrails.particle_tracer.particle import Particle
from sedtrails.exceptions import MissingConfigurationParameter
from typing import List, Tuple, Optional, Dict
import random
from dataclasses import dataclass, field
from sedtrails.configuration_interface.find import find_value


@dataclass
class SeedingConfig:
    """
    A class to represent the seeding parameters of a population of particle.
    A population is a group of particles that share the same type and seeding strategy.
    """

    population_config: Dict
    particle_type: str = field(init=False)
    release_start: str = field(init=False)  # particle for a given population are released at this time
    quantity: Optional[int] = field(init=False)  # number of particles to release per release location

    def __post_init__(self):
        _strategy = find_value(self.population_config, 'population.seeding.strategy', {}).keys()
        # self.population_config.get('population', {}).get('seeding', {}).get('strategy', {}).keys()
        if not _strategy:
            raise MissingConfigurationParameter('"strategy" is not defined as seeding parameter.')
        self.strategy = next(iter(_strategy))
        _quantity = find_value(self.population_config, 'population.seeding.quantity', {})
        if not _quantity:
            raise MissingConfigurationParameter('"quantity" is not defined as seeding parameter.')
        self.quantity = _quantity

    def _extract_release_times(self) -> None:
        """
        Extract release time from the configuration.
        """

        # TODO: make the config controllert to be the one to se the release time == to
        # the simulation start time when not specified

        release_start = find_value(self.population_config, 'population.seeding.release_start', {})
        if not release_start:
            raise MissingConfigurationParameter('"release_start" is not defined in the population configuration.')
        self.release_start = release_start


class SeedingStrategy(ABC):
    """
    Abstract base class for seeding strategies.
    """

    @abstractmethod
    def seed(self, config: SeedingConfig, **kwargs) -> List[Tuple[int, float, float]]:
        """
        Asociates quantity of particles to a seeding locations for a given strategy.

        Parameters
        ----------
        config : SeedingConfig
            Configuration object containing the seeding parameters.
        **kwargs :
            Additional keyword arguments that may be used by specific strategies.

        Returns
        -------
        list[Tuple[int, float, float]]
            A list of tuples where each tuple contains:
            - int: The quantity of particles to  be releases at a location.
            - float: The x-coordinate of the release location.
            - float: The y-coordinate of the release location.
        """
        pass


class PointStrategy(SeedingStrategy):
    """
    Seeding strategy to release particles at specific locations (x,y).
    """

    def seed(self, config: SeedingConfig, **kwargs) -> list[Tuple[int, float, float]]:
        locations = find_value(config.population_config, 'population.seeding.strategy.point.locations', [])
        if not locations:
            raise MissingConfigurationParameter('"locations" must be provided for PointStrategy.')
        if config.quantity is None:
            raise MissingConfigurationParameter('"quantity" must be an integer for PointStrategy.')
        quantity = int(config.quantity)
        seed_locations = []
        for loc_str in locations:
            try:
                x_str, y_str = loc_str.split(',')
                x = float(x_str.strip())
                y = float(y_str.strip())
                seed_locations.append((quantity, x, y))
            except Exception as e:
                raise ValueError(f"Invalid location string '{loc_str}': {e}") from e
        return seed_locations


class RandomStrategy(SeedingStrategy):
    """
    Seeding strategy to release particles at at random locations (x,y) within an area constraint by a bounding box 'xmin,ymin xmax,ymax'.
    """

    def seed(self, config: SeedingConfig, **kwargs) -> list[Tuple[int, float, float]]:
        bbox = find_value(config.population_config, 'population.seeding.strategy.random.bbox', {})
        print(bbox)
        if not bbox:
            raise MissingConfigurationParameter('"bbox" must be provided for RandomStrategy.')
        if config.quantity is None:
            raise MissingConfigurationParameter('"quantity" must be an integer for RandomStrategy.')
        quantity = int(config.quantity)
        seed_locations = []

        _bbox = bbox.replace(',', ' ').split()  # separates values with whitespaces. Order is xmin, ymin, xmax, ymax
        for _ in range(quantity):
            x = random.uniform(float(_bbox[0]), float(_bbox[2]))
            y = random.uniform(float(_bbox[1]), float(_bbox[3]))
            seed_locations.append((quantity, x, y))
        return seed_locations


class GridStrategy(SeedingStrategy):
    """
    Seeding strategy to release particles that follows regular grid pattern. The grid is defined by the distance between particles (dx, dy) and the simulation domain size. If dx and dy have the same value, a square grid is created.
    The origin of the grid is at the bottom left corner of the bounding box
    """

    def seed(self, config: SeedingConfig, **kwargs) -> list[Tuple[int, float, float]]:
        """
        Parameters
        ----------
        config : SeedingConfig
            Configuration object containing the seeding parameters.
        bbox : Dict
            Bounding box to constrain the grid area. If not provided, compuation will fail.
            Expects a dicstionary with keys 'xmin', 'xmax', 'ymin', 'ymax'.
        """
        bbox = kwargs.get('bbox')
        if bbox is None:
            raise RuntimeError('Bounding box must be provided for GridStrategy.')

        grid = find_value(config.population_config, 'population.seeding.strategy.grid', {})

        if not grid or not all(k in grid.get('separation') for k in ['dx', 'dy']):
            raise MissingConfigurationParameter('"grid" must be provided for GridStrategy.')
        if config.quantity is None:
            raise MissingConfigurationParameter('"quantity" must be an integer for GridStrategy.')
        quantity = int(config.quantity)
        seed_locations = []
        dx = find_value(grid, 'separation.dx')
        dy = find_value(grid, 'separation.dy')
        xvals = []
        yvals = []
        x = bbox['xmin']
        while x <= bbox['xmax']:
            if x > bbox['xmax']:
                break
            xvals.append(x)
            x += dx
        if xvals and xvals[-1] > bbox['xmax']:
            xvals.pop()
        y = bbox['ymin']
        while y <= bbox['ymax']:
            if y > bbox['ymax']:
                break
            yvals.append(y)
            y += dy
        if yvals and yvals[-1] > bbox['ymax']:
            yvals.pop()
        for x in xvals:
            for y in yvals:
                seed_locations.append((quantity, x, y))
        return seed_locations


class TransectStrategy(SeedingStrategy):
    """
    Seeding strategy to release particles along straight line segments.
    A line segment is defined by two points (x1, y1) and (x2, y2).
    Particles along each segment are equally spaced, and the distance between particles is defined by the number of release locations per segment (k).
    """

    def seed(self, config: SeedingConfig, **kwargs) -> list[Tuple[int, float, float]]:
        # expect to return a dictionary with keys 'segments', 'k'
        transect = find_value(config.population_config, 'population.seeding.strategy.transect', {})
        if not transect:
            raise MissingConfigurationParameter('"transect" must be provided for TransectStrategy.')

        segments = transect.get('segments', [])
        if not segments:
            raise MissingConfigurationParameter('"segments" must be provided for TransectStrategy.')
        k = transect.get('k', {})
        if not k:
            raise MissingConfigurationParameter('"k" must be provided for TransectStrategy.')
        if config.quantity is None:
            raise MissingConfigurationParameter('"quantity" must be an integer for TransectStrategy.')
        quantity = int(config.quantity)
        seed_locations = []

        # Process each segment
        for segment_str in segments:
            try:
                # Parse segment string like '1000,2000 3000,4000'
                points = segment_str.strip().split()
                if len(points) != 2:
                    raise ValueError(f'Segment must contain exactly 2 points, got {len(points)}')

                # Parse first point (x1, y1)
                x1_str, y1_str = points[0].split(',')
                x1, y1 = float(x1_str.strip()), float(y1_str.strip())

                # Parse second point (x2, y2)
                x2_str, y2_str = points[1].split(',')
                x2, y2 = float(x2_str.strip()), float(y2_str.strip())

                # Generate k equally spaced points along the segment
                for i in range(k):
                    frac = i / (k - 1) if k > 1 else 0
                    x = x1 + frac * (x2 - x1)
                    y = y1 + frac * (y2 - y1)
                    seed_locations.append((quantity, x, y))

            except Exception as e:
                raise ValueError(f"Invalid segment string '{segment_str}': {e}") from e

        return seed_locations


class ParticleFactory:
    @staticmethod
    def create_particles(
        config: SeedingConfig,
        strategy: SeedingStrategy,
        particle_type: str,
        release_time: Optional[int] = None,
        *args,
        **kwargs,
    ) -> list[Particle]:
        """
        Create a list of particles of the specified type using a seeding strategy.

        Parameters
        ----------
        config : SeedingConfig
            Configuration object containing the seeding parameters.
        strategy : SeedingStrategy
            The seeding strategy to use for generating positions.
        particle_type : str
            The type of particle to create ('sand', 'mud', 'passive').
        release_time : Optional[int]
            The release time for the particles. If not provided, will use config.release_start or default to 0.
        *args, **kwargs
            Additional arguments that may be required by specific strategies (e.g., bbox for GridStrategy).

        Returns
        -------
        list[Particle]
            List of created particles with positions and release times set.
        """
        from sedtrails.particle_tracer.particle import Sand, Mud, Passive

        type_map = {'sand': Sand, 'mud': Mud, 'passive': Passive}
        if particle_type.lower() not in type_map:
            raise ValueError(f'Unknown particle type: {particle_type}')
        ParticleClass = type_map[particle_type.lower()]

        # Call the strategy's seed method with additional arguments
        positions = strategy.seed(config, *args, **kwargs)
        particles = []
        for qty, x, y in positions:
            for _ in range(qty):
                p = ParticleClass()
                p.x = x
                p.y = y
                # Use config.release_start if available, else release_time arg, else 0
                if hasattr(config, 'release_start'):
                    p.release_time = int(getattr(config, 'release_start', 0))
                elif release_time is not None:
                    p.release_time = int(release_time)
                else:
                    p.release_time = 0
                particles.append(p)

        return particles


# TODO:  save the final positions of particles to a vector based files (e.g., shapefile?, GeoJSON?)

if __name__ == '__main__':
    # Example usage
    config_point = SeedingConfig(
        {
            'population': {
                'seeding': {
                    'strategy': {'point': {'locations': ['1.0,2.0', '3.0,4.0']}},
                    'quantity': 1,
                }
            }
        }
    )

    particles = ParticleFactory.create_particles(config_point, PointStrategy(), particle_type='sand', release_time=0)
    print(
        'Created particles:', particles[-5:]
    )  # Should print the created particles with their positions and release times

    # Measure memory size of particles
    import sys

    total_memory = sum(sys.getsizeof(particle) for particle in particles)
    print(f'Total memory size of {len(particles)} particles: {total_memory} bytes ({total_memory / 1024:.2f} KB)')
