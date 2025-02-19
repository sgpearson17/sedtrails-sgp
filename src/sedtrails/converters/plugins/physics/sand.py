"""
Physics plugin for sand particles
"""

from sedtrails.converters import BasePhysicsPlugin

class PhysicsPlugin(BasePhysicsPlugin): # All plugins must inherit from the BasePhysicsPlugin class and be named as PhysicsPlugin
    """
    Physics plugin for sand particles.
    """

    def convert(self, *args, **kwargs):
        """
        Computes the physics of sand particles in the simulation.
        """
        print ("Computing physics for sand particles")



if __name__ == "__main__":
    sand = PhysicsPlugin()
    sand.convert()
