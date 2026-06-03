# Looping Eulerian Forcing Fields

## Overview

SedTRAILS can run simulations longer than the available flow-model forcing by looping the input Eulerian fields in time. This mirrors a common workflow in Lagrangian particle tracking where a representative cycle (tidal, spring-neap, seasonal) is repeated to extend trajectories. This was a "widely used" feature of the old MATLAB sedtrails code.

A concise statement of the approach appears in van Sebille et al. (2018): looping the model data in time “permits particles to be advected for longer time scales than available from the raw data,” provided the fields are stationary and there are no large discontinuities at the wraparound. See the paper for broader context and examples: https://doi.org/10.1016/j.ocemod.2017.11.008.

OLD TEXT COPIED FROM ERIC:

_"The maximum integration time in Eq. (21) is limited to the run time of a given model simulation. A number of oceanic processes, however, have time scales that exceed these run times (e.g., England, 1995, Stouffer, 2004, Danabasoglu, 2004). Using Lagrangian particles to temporally resolve for example the meridional overturning circulation (Blanke, Arhan, Madec, Roche, 1999, Thomas, Tréguier, Blanke, Deshayes, Voldoire, 2015) or inter-basin connectivity (Blanke and Speich, 2002) can be difficult with many state of the art climate models. To address this problem, a commonly employed ad hoc method is to loop the model data in time such that the velocity and tracer fields are returned to the first time step once the end has been reached (e.g., Döös, Nycander, Coward, 2008, van Sebille, Johns, Beal, 2012, Thomas, Tréguier, Blanke, Deshayes, Voldoire, 2015). This approach thus permits particles to be advected for longer time scales than available from the raw data. However, particle looping can only work if the model has no drift in the velocity or tracer fields, that there are no large unphysical jumps in the fields between the end and the beginning of the model run, and that any unphysical jumps will have a small net effect on the particle pathways."_

## When to Use Looping

Use looping when:
- Your forcing data represent a repeatable cycle.
- You want to study transport over longer periods than the forcing duration.

Avoid looping when:
- There is a trend or drift in the input flow fields.
- The end and start of the forcing period have large, unphysical jumps.

## Configuration

Enable looping by setting `inputs.repeat_eulerian_fields` to `true`.

```yaml
inputs:
  data: ./sample-data/inlet_sedtrails.nc
  read_interval: 10D
  repeat_eulerian_fields: true

# Run longer than the input forcing duration
# (example: 60 days of particles from a 15 day forcing file)

time:
  start: 2016-09-21 19:20:00
  duration: 60D
  timestep: 60S
```

If `repeat_eulerian_fields` is `false`, SedTRAILS will keep using the final available Eulerian fields once the end of the forcing period is reached.

## How Time Mapping Works

Internally, SedTRAILS maps each particle time step to an Eulerian forcing timestamp. This is handled by the helper `_map_eulerian_field_time`, which:
- Wraps the simulation time into the forcing window when looping is enabled.
- Clamps the time to the final forcing timestamp when looping is disabled.

This time mapping is applied consistently across velocity and tracer fields so that particle advection remains synchronized with the Eulerian data.

## Morfac Decompression

When `general.input_model.morfac` is greater than 1, the forcing time axis is decompressed before mapping. This means:
- A morphologically accelerated input model is interpreted at its “real time” scale.
- Looping and interpolation operate on the decompressed timeline.

This keeps the Eulerian time base consistent with the Lagrangian integration when morphological acceleration has been applied to the input model output.

## Practical Tips

- Test for discontinuities by plotting a few grid-point time series across the final and initial forcing steps.
- For long loops, confirm that the cycle length (e.g., 14.76 days for spring-neap) is represented accurately in the input data.
- Consider comparing short (single-cycle) and long (multi-cycle) runs to quantify sensitivity to looping.
