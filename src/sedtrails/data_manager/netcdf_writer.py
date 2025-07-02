"""
NetCDF Writer
=============
Writes NetCDF files produced by the SedTrails Particle Tracer System.
"""

import xugrid as xu
from pathlib import Path
from datetime import datetime

class NetCDFWriter:
    """
    A class for writing NetCDF files for the SedTrails Particle Tracer System.

    Attributes
    ----------
    output_dir : pathlib.Path
        The directory where output files (NetCDF, images, etc.) are stored.
    """

    def __init__(self, output_dir):
        output_dir = Path(output_dir)
        # If the output directory already exists, we add a timestamp to avoid overwriting
        if output_dir.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = output_dir.parent / f"{output_dir.name}_{timestamp}"
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _validate_filename(self, filename):
        """
        Validates the file name to ensure it is a NetCDF file.
        """        
        if not filename.endswith(".nc"):
            raise ValueError("Output file must have a .nc extension.")

    def write(self, ugrid_dataset, filename):
        """
        Write a xu.UgridDataset to a NetCDF file in the output directory.

        Parameters
        ----------
        ugrid_dataset : xu.UgridDataset
            The UGRID-compliant dataset to write.
        filename : str
            The name of the NetCDF file to write (should end with .nc).
        """
        self._validate_filename(filename)
        output_path = self.output_dir / filename
        if not isinstance(ugrid_dataset, xu.UgridDataset):
            raise TypeError("Input must be a xu.UgridDataset.")
        ugrid_dataset.to_netcdf(output_path)
        print(f"NetCDF file written to {output_path}")