# Luminescence workflow

This page documents SedTRAILS offline luminescence and light-exposure helpers, including assumptions, standalone usage, and plotting.

## Scope

The current workflow provides:

- suspended-sediment-driven attenuation estimates
- particle-level light intensity and cumulative exposure diagnostics
- clear-sky incoming radiation estimates from timestamps and location
- MATLAB-style diagnostic plotting

The bleaching-probability and OSL signal terms are currently placeholders in the returned results.

## Core functions

### Attenuation and exposure

From `sedtrails.simulation_analysis.light_exposure`:

- `attenuation_from_storlazzi`
- `combine_attenuation_components`
- `particle_depth_below_surface`
- `light_intensity_at_particle`
- `integrate_exposure`
- `compute_light_exposure`
- `rouse_centroid_depth`
- `surface_light_series`

### Solar forcing

From `sedtrails.simulation_analysis.solar_radiation`:

- `compute_clear_sky_surface_light`

### Plotting

From `sedtrails.simulation_analysis.light_exposure_plotting`:

- `plot_particle_bleaching_potential`

## Standalone usage

### 1) Build attenuation and incoming-light forcing

```python
import numpy as np

from sedtrails.simulation_analysis import (
    attenuation_from_storlazzi,
    combine_attenuation_components,
    compute_clear_sky_surface_light,
)

time = np.array([
    "2020-06-21T00:00:00",
    "2020-06-21T01:00:00",
    "2020-06-21T02:00:00",
], dtype="datetime64[s]")

# Example concentration timeseries [kg m-3]
conc = np.array([[0.2], [0.3], [0.15]])

components = attenuation_from_storlazzi(sed_conc_mud=conc, sed_conc_sand=conc)
kd = combine_attenuation_components(components["mud"], components["sand"], strategy="sand")

surface_light = compute_clear_sky_surface_light(
    time,
    latitude_deg=53.45,
    longitude_deg=5.73,
    utc_offset_hours=1.0,
    atmospheric_transmissivity=0.75,
    output_unit="umol_photons_m2_s",
)
```

### 2) Compute particle exposure

```python
from sedtrails.simulation_analysis import compute_light_exposure

water_depth = np.array([[4.0], [4.1], [4.0]])
particle_height_above_bed = np.array([[0.5], [0.6], [0.4]])
burial_depth = np.array([[0.0], [0.02], [0.0]])

result = compute_light_exposure(
    time,
    water_depth,
    kd,
    z=particle_height_above_bed,
    z_convention="height_above_bed",
    surface_light=surface_light,
    burial_depth=burial_depth,
    burial_depth_threshold=1e-6,
    light_threshold=0.01,
    reference_surface_light=max(float(np.nanmax(surface_light)), 1.0),
)

print("Final cumulative dose:", result.cumulative_light_dose[-1, 0])
print("Final equivalent sunlight hours:", result.equivalent_sunlight_hours[-1, 0])
```

### 3) Plot MATLAB-style diagnostics

```python
from sedtrails.simulation_analysis.light_exposure_plotting import plot_particle_bleaching_potential

fig, axes = plot_particle_bleaching_potential(
    time,
    water_depth[:, 0],
    particle_height_above_bed[:, 0],
    {"Total": conc[:, 0]},
    surface_light,
    kd[:, 0],
    result,
)
```

## Notebook forcing modes

The offline workflow notebook supports:

- `surface_light_mode = "clear_sky"`
- `surface_light_mode = "constant_daylight"`

Run in this order after changing mode/options:

1. workflow options cell
2. compute cell
3. MATLAB-style plot cell

## Assumptions summary

For explicit model assumptions and equations, see [Luminescence method references](../references/luminescence-methods.md).

## Practical recommendations

- Use clear-sky forcing for realistic diurnal and seasonal shape in exploratory analyses.
- For absolute irradiance studies, replace clear-sky forcing with measured or externally modeled radiation products.
- Keep track of units: attenuation in m-1, light in either W m-2 or umol photons m-2 s-1.
