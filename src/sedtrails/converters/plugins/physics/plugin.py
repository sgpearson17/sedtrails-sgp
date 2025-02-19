"""
A common interface for physics plugins.
Plugins must inherit from this class and implement the convert method.
"""

from abc import ABC, abstractmethod

class BasePhysicsPlugin(ABC):
    """
    Abstract base class for physics plugins.
    """

    @abstractmethod
    def convert(self, *args, **kwargs):
        """
        Computes the physics of particles in the simulation based on particle types.
        """
        pass
