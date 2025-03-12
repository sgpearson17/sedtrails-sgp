"""
NetCDF Reader

Reads NetCDF files produced by the SedTrails Particle Tracer System.
"""

import xugrid as xu
from pathlib import Path


class NetCDFReader:
    """
    A class for reading NetCDF files produced by the SedTrails Particle Tracer System.

    Attributes:
    -----------
        file_path: str
            The path to the CF complaint NetCDF file.
    """

    def __init__(self, file_path: str) -> None:
        self.file_path: str = file_path
        self.data: xu.UgridDataset = self._read_file()
            
    def _validate_file(self) -> bool:
        """
        Validates the file path to ensure it is a NetCDF file.
        """
        FILE_EXTENSIONS = ['.nc']
        path = Path(self.file_path)
        return path.is_file() and path.suffix in FILE_EXTENSIONS

    def _read_file(self):
        """
        Reads the NetCDF file using the lazy loader from xugrid.
        """
        if not self._validate_file():
            raise ValueError("Invalid file path. Ensure the file is a NetCDF file.")
        else:
            self.data = xu.open_mfdataset(self.file_path)
            # TODO:  this retuns a hadler for the dataset. What this class shoudl do with it?
    
    def __str__(self):
        """
        Returns the string representation of the NetCDFReader object.
        """

        return self.data


if __name__ == "__main__":
    reader = NetCDFReader('sample-data/dfm_sedtrails.nc')
    print(reader)

    # # dfm dataset:
    # fm = 'sample-data/dfm_sedtrails.nc'
    # d3 = 'sample-data/d3d4.dat'
    # # 
    # uds = xu.open_mfdataset(fm)  # return Ugrid dataset

    # uds = xu.open_dataset(fm)  # return Ugrid dataset


    # print(uds)

    # elev = uds['elevation']
    # print(elev)

    # elev.plot()
