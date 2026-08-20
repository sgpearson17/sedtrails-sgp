"""
A common interface for format plugins.
Plugins must inherit from this class and implement the convert method.
"""

from abc import ABC, abstractmethod
from sedtrails.transport_converter.sedtrails_data import SedtrailsData


class BaseFormatPlugin(ABC):
    """
    Abstract base class for format plugins.
    """

    @abstractmethod
    def convert(self, *args, **kwargs) -> SedtrailsData:
        """
        Converts flow-field data from varios formats to the sedtrails format.

        Returns
        -------
        SedtrailsData
            The converted SedtrailsData object.

        Parameters
        ----------
        *args : object
            Additional positional arguments passed through to the implementation.
        **kwargs : object
            Additional keyword arguments passed through to the implementation.
        """
        pass

    def _add_runtime_coordinate_metadata(self, metadata) -> None:
        """Add explicitly configured runtime geometry metadata.

        Parameters
        ----------
        metadata : object
            SedTRAILS metadata container exposing ``add(name, value)``.
        """
        for name in (
            'runtime_geometry',
            'surface_model',
            'earth_radius_m',
            'longitude_wrap',
            'velocity_basis',
        ):
            value = getattr(self, name, None)
            if value is not None:
                metadata.add(name, value)
