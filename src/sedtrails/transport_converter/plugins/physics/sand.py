"""
Physics plugin for sand particles
"""

from sedtrails.transport_converter.plugins import BasePhysicsPlugin


class PhysicsPlugin(BasePhysicsPlugin): # All plugins must inherit from the BasePhysicsPlugin class and be named as PhysicsPlugin
    """
    Physics plugin for sand particles.
    """

    def __init__(self):
        super().__init__()

    def convert(self, *args, **kwargs):
        """
        Computes the physics of sand particles in the simulation.
        """

        print ("Computing physics for sand particles")

        # logic for  converting sand particles
        # If there are multiple methods in which the coversion for 
        # a type of particle happens, we could: 
        # 1. pass the method as an argument, or
        # 2. implement a library that is loaded here. 

