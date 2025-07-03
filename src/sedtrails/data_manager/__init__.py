"""A directory for the Data Management Module"""

from .netcdf_reader import NetCDFReader
from .netcdf_writer import NetCDFWriter


__all__ = [
    'NetCDFReader',
    'NetCDFWriter',
]
