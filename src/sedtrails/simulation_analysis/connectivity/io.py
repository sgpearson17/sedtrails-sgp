"""NetCDF IO for SedTRAILS connectivity adjacency matrices."""

from __future__ import annotations

from pathlib import Path

import xarray as xr


def write_adjacency_netcdf(dataset: xr.Dataset, path: str | Path) -> Path:
    """Write a connectivity adjacency dataset to NetCDF."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(path)
    return path


def read_adjacency_netcdf(path: str | Path) -> xr.Dataset:
    """Read a SedTRAILS connectivity adjacency NetCDF file."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Connectivity NetCDF file '{path}' not found.")
    ds = xr.open_dataset(path)
    if ds.attrs.get('sedtrails_file_kind') != 'connectivity_adjacency':
        raise ValueError("Expected a SedTRAILS connectivity adjacency NetCDF file.")
    return ds
