"""A plugin for converting Delft3D4 TRIM format to SedTRAILS format."""

from sedtrails.transport_converter.plugins import BaseFormatPlugin
from sedtrails.transport_converter.sedtrails_data import SedtrailsData


class FormatPlugin(BaseFormatPlugin):
    """
    Plugin for converting Delft3D4 TRIM format to SedTRAILS format.
    """

    def convert(self, *args, **kwargs) -> SedtrailsData:
        """
        Converts  from Delft3D4 TRIM format.

        Parameters:
        -----------
        *args : tuple
            Positional arguments for conversion.
        **kwargs : dict
            Keyword arguments for conversion.

        Parameters
        ----------
        *args : object
            Additional positional arguments passed through to the implementation.
        **kwargs : object
            Additional keyword arguments passed through to the implementation.

        Returns
        -------
        SedtrailsData
            Converted SedTRAILS data object.
        """
        raise NotImplementedError('Delft3D trim file conversion is not yet implemented.')
