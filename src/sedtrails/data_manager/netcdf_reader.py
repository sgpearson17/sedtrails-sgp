"""
NetCDF Reader

Reads NetCDF files produced by the SedTrails Particle Tracer System.
"""

import xugrid as xu

ds = xu.data.adh_san_diego(xarray=True)
print(ds)

uds = xu.UgridDataset(ds)
print(uds)

elev = uds['elevation']
print(elev)

elev.plot()
