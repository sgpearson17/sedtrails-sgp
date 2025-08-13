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
        _strategy = self.population_config.get('population', {}).get('seeding', {}).get('strategy', {}).keys()
        if not _strategy:
            raise MissingConfigurationParameter('"strategy" is not defined as seeding parameter.')
        self.strategy = next(iter(_strategy))

        _quantity = self.population_config.get('population', {}).get('seeding', {}).get('quantity', {})
        if not _quantity:
            raise MissingConfigurationParameter('"quantity" is not defined as seeding parameter.')
        self.quantity = _quantity

    def _extract_release_times(self) -> None:
        """
        Extract release time from the configuration.
        """

        # TODO: make the config controllert to be the one to se the release time == to
        # the simulation start time when not specified

        release_start = self.population_config.get('population', {}).get('seeding', {}).get('release_start', {})
        if not release_start:
            raise MissingConfigurationParameter('"release_start" is not defined in the population configuration.')
        self.release_start = release_start


class SeedingStrategy(ABC):
    """
    Abstract base class for seeding strategies.
    """

    @abstractmethod
    def seed(self, config: SeedingConfig) -> list[Tuple[int, float, float]]:
        """
        Asociates quantity of particles to a seeding locations for a given strategy.

        Parameters
        ----------
        config : SeedingConfig
            Configuration object containing the seeding parameters.

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

    def seed(self, config: SeedingConfig) -> list[Tuple[int, float, float]]:
        # Expect config.population_config['population']['seeding']['locations'] = ['x,y', ...]
        locations = config.population_config.get('population', {}).get('seeding', {}).get('locations', [])
        if not locations:
            raise MissingConfigurationParameter('"locations" must be provided for PointStrategy.')
        if config.quantity is None:
            raise MissingConfigurationParameter('"quantity" must be an integer for PointStrategy.')
        quantity = int(config.quantity)
        parsed_locations = []
        for loc_str in locations:
            try:
                x_str, y_str = loc_str.split(',')
                x = float(x_str.strip())
                y = float(y_str.strip())
                parsed_locations.append((quantity, x, y))
            except Exception as e:
                raise ValueError(f"Invalid location string '{loc_str}': {e}") from e
        return parsed_locations


class RandomStrategy(SeedingStrategy):
    """
    Seeding strategy to release particles at at random locations (x,y) within an area constraint by a bounding box (xmin, xmax, ymin, ymax).
    """

    def seed(self, config: SeedingConfig) -> list[Tuple[int, float, float]]:
        # Expect config.population_config['population']['seeding']['bbox'] = {'xmin':..., 'xmax':..., 'ymin':..., 'ymax':...}
        bbox = config.population_config.get('population', {}).get('seeding', {}).get('bbox', {})
        if not bbox or not all(k in bbox for k in ['xmin', 'xmax', 'ymin', 'ymax']):
            raise MissingConfigurationParameter('"bbox" must be provided for RandomStrategy.')
        if config.quantity is None:
            raise MissingConfigurationParameter('"quantity" must be an integer for RandomStrategy.')
        quantity = int(config.quantity)
        particles = []
        for _ in range(quantity):
            x = random.uniform(bbox['xmin'], bbox['xmax'])
            y = random.uniform(bbox['ymin'], bbox['ymax'])
            particles.append((1, x, y))
        return particles


class GridStrategy(SeedingStrategy):
    """
    Seeding strategy to release particles that follows regular grid pattern. The grid is defined by the distance between particles (dx, dy) and the simulation domain size. If dx and dy have the same value, a square grid is created.
    """

    def seed(self, config: SeedingConfig) -> list[Tuple[int, float, float]]:
        # Expect config.population_config['population']['seeding']['grid'] = {'xmin':..., 'xmax':..., 'ymin':..., 'ymax':..., 'dx':..., 'dy':...}
        grid = config.population_config.get('population', {}).get('seeding', {}).get('grid', {})
        if not grid or not all(k in grid for k in ['xmin', 'xmax', 'ymin', 'ymax', 'dx', 'dy']):
            raise MissingConfigurationParameter('"grid" must be provided for GridStrategy.')
        if config.quantity is None:
            raise MissingConfigurationParameter('"quantity" must be an integer for GridStrategy.')
        quantity = int(config.quantity)
        particles = []
        xvals = [grid['xmin'] + i * grid['dx'] for i in range(int((grid['xmax'] - grid['xmin']) / grid['dx']) + 1)]
        yvals = [grid['ymin'] + j * grid['dy'] for j in range(int((grid['ymax'] - grid['ymin']) / grid['dy']) + 1)]
        for x in xvals:
            for y in yvals:
                particles.append((quantity, x, y))
        return particles


class TransectStrategy(SeedingStrategy):
    """
    Seeding strategy to release particles along straight line segments.
    A line segment is defined by two points (x1, y1) and (x2, y2).
    Particles along each segment are equally spaced, and the distance between particles is defined by the number of release locations per segment (k).
    """

    def seed(self, config: SeedingConfig) -> list[Tuple[int, float, float]]:
        # Expect config.population_config['population']['seeding']['transect'] = {'x1':..., 'y1':..., 'x2':..., 'y2':..., 'k':...}
        transect = config.population_config.get('population', {}).get('seeding', {}).get('transect', {})
        if not transect or not all(k in transect for k in ['x1', 'y1', 'x2', 'y2', 'k']):
            raise MissingConfigurationParameter('"transect" must be provided for TransectStrategy.')
        x1, y1, x2, y2, k = transect['x1'], transect['y1'], transect['x2'], transect['y2'], transect['k']
        if config.quantity is None:
            raise MissingConfigurationParameter('"quantity" must be an integer for TransectStrategy.')
        quantity = int(config.quantity)
        particles = []
        for i in range(k):
            frac = i / (k - 1) if k > 1 else 0
            x = x1 + frac * (x2 - x1)
            y = y1 + frac * (y2 - y1)
            particles.append((quantity, x, y))
        return particles


class ParticleFactory:
    @staticmethod
    def create_particles(
        config: SeedingConfig, strategy: SeedingStrategy, particle_type: str, release_time: Optional[int] = None
    ) -> list[Particle]:
        """
        Create a list of particles of the specified type using a seeding strategy.
        """
        from sedtrails.particle_tracer.particle import Sand, Mud, Passive

        type_map = {'sand': Sand, 'mud': Mud, 'passive': Passive}
        if particle_type.lower() not in type_map:
            raise ValueError(f'Unknown particle type: {particle_type}')
        ParticleClass = type_map[particle_type.lower()]

        positions = strategy.seed(config)
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
