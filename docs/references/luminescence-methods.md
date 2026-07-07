# Luminescence Method References

This page lists the equations and assumptions used by the current SedTRAILS offline luminescence workflow.

## 1) Clear-sky solar radiation

Implemented in `sedtrails.simulation_analysis.solar_radiation.compute_clear_sky_surface_light`.

### Geometry and time correction

- Fractional year angle and trigonometric declination approximation
- Equation of time correction
- Solar hour angle from civil time, longitude, and UTC offset

Equivalent forms are commonly documented in NOAA solar-calculator notes.

### Top-of-atmosphere irradiance

The workflow computes horizontal top-of-atmosphere irradiance as:

$$
I_{toa} = S_0 E_0 \max(\cos\theta_z, 0)
$$

where:

- $S_0$ is the solar constant
- $E_0$ is Earth-Sun distance correction
- $\theta_z$ is solar zenith angle

### Air mass and transmissivity

Relative air mass is estimated with the Kasten-Young approximation, then clear-sky attenuation is applied as:

$$
I_{surf} = I_{toa}\,\tau^m
$$

where $\tau$ is bulk atmospheric transmissivity and $m$ is relative air mass.

### PAR conversion

When requested, W m-2 is converted to PAR using a fixed scalar factor (default 2.1 umol photons J-1).

## 2) Water-column attenuation from concentration

Implemented in `sedtrails.simulation_analysis.light_exposure.attenuation_from_storlazzi`.

Linear attenuation components for mud and sand are based on coefficients digitized from:

- Storlazzi et al. (2015)

## 3) Particle vertical depth and exposure integration

Implemented in `sedtrails.simulation_analysis.light_exposure`:

- particle depth conversion from z conventions
- Beer-Lambert attenuation at particle depth
- burial and beaching rules
- cumulative integration over saved timesteps

## 4) MacDonald-style centroid depth option

Implemented in `sedtrails.simulation_analysis.light_exposure.rouse_centroid_depth` and used in the workflow notebook for `macdonald_zc` mode.

## References

- Storlazzi, C. D., et al. (2015). (attenuation coefficients used in this workflow).
- Kasten, F., and Young, A. T. (1989). Revised optical air mass tables and approximation formula.
- NOAA Solar Calculator methodology notes (declination/equation-of-time closed-form approximations).

Note: This workflow is intended for first-order clear-sky forcing in sensitivity and exploratory analyses. For absolute irradiance studies, use observed or full meteorological radiation products.
