import os
import importlib.util
import inspect
import pytest
import sys
from sedtrails.transport_converter.plugins.physics.plugin import BasePhysicsPlugin

# Get the project root directory and construct path to plugins
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = _current_dir
# Walk up to find project root (contains src directory)
while not os.path.exists(os.path.join(_project_root, 'src')) and _project_root != os.path.dirname(_project_root):
    _project_root = os.path.dirname(_project_root)

PLUGIN_DIR = os.path.join(_project_root, 'src', 'sedtrails', 'transport_converter', 'plugins', 'physics')


def get_plugin_classes():
    """
    Dynamically discovers and imports all plugin classes in the physics plugin directory,
    excluding the base plugin and dunder files.

    Returns:
        list: List of plugin classes found in the directory.
    """
    plugin_classes = []
    
    # Get list of plugin files, excluding problematic ones
    plugin_files = []
    try:
        for fname in os.listdir(PLUGIN_DIR):
            if (fname.endswith('.py') and 
                fname != 'plugin.py' and 
                not fname.startswith('__')):
                plugin_files.append(fname)
    except (OSError, FileNotFoundError):
        # If plugin directory doesn't exist, return empty list
        return plugin_classes
    
    for fname in plugin_files:
        try:
            module_path = os.path.join(PLUGIN_DIR, fname)
            module_name = f'sedtrails.transport_converter.plugins.physics.{fname[:-3]}'
            
            # Skip if module is already loaded to avoid recursion
            if module_name in sys.modules:
                module = sys.modules[module_name]
            else:
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is None or spec.loader is None:
                    continue
                    
                module = importlib.util.module_from_spec(spec)
                # Add to sys.modules before execution to prevent recursion
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
            
            # Find classes defined in this module
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Only consider classes defined in this module (not imported ones)
                # and that inherit from BasePhysicsPlugin
                if (obj.__module__ == module_name and 
                    name != 'BasePhysicsPlugin' and
                    issubclass(obj, BasePhysicsPlugin)):
                    plugin_classes.append(obj)
                    
        except Exception as e:
            # Plugin loading failures should cause test failures, not silent skips
            raise ImportError(f"Failed to load plugin {fname}: {e}") from e
    
    return plugin_classes


@pytest.fixture
def plugin_classes():
    """Fixture to provide plugin classes for testing."""
    classes = get_plugin_classes()
    if not classes:
        pytest.skip("No plugin classes found")
    return classes


@pytest.mark.integration
def test_plugin_inherits_base(plugin_classes):
    """
    Integration test to ensure each plugin inherits from BasePhysicsPlugin.
    """
    for plugin_class in plugin_classes:
        module_parts = plugin_class.__module__.split('.')
        plugin_file = module_parts[-1] + '.py' if module_parts else 'unknown'
        assert issubclass(plugin_class, BasePhysicsPlugin), f'{plugin_file} does not inherit from BasePhysicsPlugin'


@pytest.mark.integration
def test_plugin_has_add_physics(plugin_classes):
    """
    Integration test to ensure each plugin has an add_physics method.
    """
    for plugin_class in plugin_classes:
        module_parts = plugin_class.__module__.split('.')
        plugin_file = module_parts[-1] + '.py' if module_parts else 'unknown'
        assert hasattr(plugin_class, 'add_physics'), f'{plugin_file} does not have an add_physics method'


@pytest.mark.integration
def test_plugin_add_physics_callable(plugin_classes):
    """
    Integration test to ensure each plugin's add_physics method is callable.
    """
    for plugin_class in plugin_classes:
        module_parts = plugin_class.__module__.split('.')
        plugin_file = module_parts[-1] + '.py' if module_parts else 'unknown'
        method = getattr(plugin_class, 'add_physics', None)
        assert callable(method), f'add_physics method in {plugin_file} is not callable'


@pytest.mark.integration
def test_plugin_add_physics_accepts_sedtrails_data(plugin_classes):
    """
    Integration test to ensure each plugin's add_physics method accepts sedtrails_data as parameter.
    """
    for plugin_class in plugin_classes:
        module_parts = plugin_class.__module__.split('.')
        plugin_file = module_parts[-1] + '.py' if module_parts else 'unknown'

        method = getattr(plugin_class, 'add_physics', None)
        assert method is not None, f'{plugin_file} does not have an add_physics method'

        sig = inspect.signature(method)
        params = list(sig.parameters.values())

        # The first parameter should be 'self', the second should be 'sedtrails_data'
        assert len(params) >= 2, (
            f"add_physics method in {plugin_file} should accept at least 'self' and 'sedtrails_data' as arguments"
        )
        assert params[1].name == 'sedtrails_data', (
            f"add_physics method in {plugin_file} should take `sedtrails_data` as second parameter, got '{params[1].name}'"
        )
