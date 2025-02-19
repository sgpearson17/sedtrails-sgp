"""
A common interface for format plugins.
Plugins must inherit from this class and implement the convert method.
"""

from abc import ABC, abstractmethod

class BaseFormatPlugin(ABC):
    """
    Abstract base class for format plugins.
    """

    @abstractmethod
    def convert(self, *args, **kwargs):
        """
        Converts flow-field data from varios formats to the sedtrails format.
        """
        pass
