# Input Formats

This page collects the practical differences between the input model formats SedTRAILS can work with. The goal is to document the assumptions that matter for loading data, choosing variables, and interpreting model output.

Note: the format converter interpolates data at all U/V points to cell centers and flattens structured grids to 1D spatial vectors so downstream modules can treat all inputs consistently.

## D-Flow FM

D-Flow FM input is the most direct path through SedTRAILS. The seeding GUI and the simulation workflow expect a NetCDF-based flow field with cell-center coordinates and a bathymetry variable that can be used to preview the domain.

Typical variables used by the GUI and runtime are:

- `net_xcc`
- `net_ycc`
- `bedlevel` or `bed_level`

## Delft3D 4

By default, the map output from Delft3D-4 is written to binary `trim-*.dat` files. At present it is not possible to directly read these files in with Python, so it is better to instead write the map output as `*.nc` files. To enable `*.nc` output in Delft3D-4, add the following lines to the `*.mdf` file:
```
FlNcdf= #maphis#
ncFormat=4
```
The resulting  `*.nc` file can then be directly read by SedTRAILS using the `d3d4_netcdf.py` format converter plugin.

When in doubt, check the converter and the example configuration for the exact variable names and coordinate conventions.

## XBeach

We wrote this explicitly to work with `*_mean` variables only. Otherwise, especially in surfbeat mode, the global output provides randomly-phased long wave flow fields, which don't provide meaningful information on flow or transport for SedTRAILS- just instantaneous snapshots. This means that we work with `zb_mean`, which is the average bed level over that time avg interval (defined by `tintg` in the params file).

The variables expected in the output section of the XBeach `params.txt` file are:

```text
nmeanvar =13
thetamean
zb
hh
ue
ve
taubx
tauby
Susg
Svsg
Subg
Svbg
ua
cctot
```

In practice, the GUI uses the spatial coordinates from `globalx` and `globaly`, and bathymetry preview defaults to `zb_mean` when it is available.

## SFINCS

SFINCS input is supported through the same general configuration flow, but it has its own conventions for reference dates, grid layout, and file naming. The main thing to watch is whether the file contains the coordinate and time variables required by the converter and by any visualization path you plan to use.

For configuration details, start from the SFINCS example and the converter-specific documentation, then verify the exact variable names in your NetCDF file.

