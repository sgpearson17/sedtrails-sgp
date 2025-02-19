"""
Computes the physics of particles in the simulation based on particle types.
"""

import importlib

# type of physics plugin to use.
## latet the value for this variable can be set based on the
## modeler's choice in the configuration file.
PHYSICS_PLUGINS = "plugins.physics.sand"

# import converter dynamically
physics_converter = importlib.import_module(PHYSICS_PLUGINS, ".")

# this guarantees that we can invoke the convertr
# independent of its name
converter = physics_converter.PhysicsPlugin()

# do convertion
converter.convert()



