import os
import importlib.util
import inspect
import pytest
from sedtrails.transport_converter.plugins.physics.plugin import BasePhysicsPlugin

# TODO: Is there a better way to get the plugin directory?
PLUGIN_DIR = os.path.dirname(__file__).replace(
    'tests/transport_converter/plugins', 'src/sedtrails/transport_converter/plugins/physics'
    )

def get_plugin_classes():
    """
    Dynamically discovers and imports all plugin classes in the physics plugin directory,
    excluding the base plugin and dunder files.

    Returns:
        list: List of plugin classes found in the directory.
    """    
    plugin_classes = []
    for fname in os.listdir(PLUGIN_DIR):
        if fname.endswith('.py') and fname != 'plugin.py' and not fname.startswith('__'):
            module_path = os.path.join(PLUGIN_DIR, fname)
            module_name = f"sedtrails.transport_converter.plugins.physics.{fname[:-3]}"
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for _, obj in inspect.getmembers(module, inspect.isclass):
                # Only consider classes defined in this module (not imported ones)
                if obj.__module__ == module_name:
                    plugin_classes.append(obj)
    return plugin_classes

@pytest.mark.integration
def test_all_plugins_inherit_base_and_implement_add_physics():
    """
    Integration test to ensure all plugins in the physics plugin directory:
    - Inherit from BasePhysicsPlugin
    - Implement a callable add_physics method
    """    
    plugin_classes = get_plugin_classes()
    assert plugin_classes, "No plugin classes found in the plugins directory."
    for cls in plugin_classes:
        assert issubclass(cls, BasePhysicsPlugin), f"{cls.__name__} does not inherit from BasePhysicsPlugin"
        assert hasattr(cls, "add_physics"), f"{cls.__name__} does not have an add_physics method"
        method = cls.add_physics
        assert callable(method), f"{cls.__name__}.add_physics is not callable"