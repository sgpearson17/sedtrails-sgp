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
from sedtrails.particle_tracer.particle import Particle
from typing import List


class SeedingStrategy(ABC):
    """
    Abstract base class for seeding strategies.
    """
    def __init__(self, **config):
        self.config = config

    @abstractmethod
    def seed(self) -> list[Particle]:
        """
        Seed particles at the specified location.

        Parameters
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

    def _validate_initial_positions(self, initial_positions: List[tuple] | tuple, count: int):
        """
        Validate the initial positions and count of particles.
        
        Parameters
        ----------
        initial_positions : list[tuple] | tuple
            List of tuples containing the (x,y) coordinates of the particles.
            If a single tuple is provided, it will be used for all particles.
        count : int
            Number of particles to seed.

        Raises
        ------
        TypeError
            If initial_positions is not a list or tuple.
        ValueError
            If the number of initial positions does not match the count.
        """
        result = True
        if not isinstance(initial_positions, (list, tuple)):
            result = False
            raise TypeError(f"Expected 'initial_positions' to be a list or tuple, got {type(initial_positions).__name__}")
        if len(initial_positions) != count:
            result = False
            raise ValueError(f"Number of initial positions ({len(initial_positions)}) does not match the count ({count}).")
        return result
    

    # Continue here

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
        ------"""
        
         if isinstance(release_times, int):
            release_times = [release_times] * count
        elif not isinstance(release_times, list):
            raise TypeError(f"Expected 'release_times' to be a list or int, got {type(release_times).__name__}")
        if len(release_times) != count:
            raise ValueError(f"Number of release times ({len(release_times)}) does not match the count ({count}).")
        particles = [] # List to store the particles

    

class XYSeeding(SeedingStrategy):
    """
    Seeding strategy to release particles at a specific location (x,y).
    """

    def seed(self, initial_positions=List[tuple]| tuple,  count: int = 1, release_times:List|int=1) -> list[Particle]:
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
            If a single integer is provided, it will be used for all particles. Default is 1. 

        Returns
        -------
        list[Particle]
            List of seeded particles.
        """



        # Validate release times
       

        #TODO: Add logic to create a list of particles objects using the Particle class and the 'count', and 
        # assign the initial position (x,y) to each particle, and the release time (t) to each particle.
        
        return particles



class GridSeeding(SeedingStrategy):
    """
    Seeding strategy to release particles at a regular grid pattern.
    """

    def seed(self, spatial_domain, count: int = 1, release_times : List|int = 1, mask = None) -> list[Particle]:
        #TODO: agree ont he data type for spatial_domain and mask

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

    

        particles = [] # List to store the particles

        #TODO: Add logic to create a list of particles objects using the Particle class and the 'count', and 
        # assign the initial position (x,y) to each particle, and the release time (t) to each particle.
        
        return particles



if __name__ == "__main__":

    SeedingStrategy()