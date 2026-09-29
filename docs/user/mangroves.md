# Mangrove Tracers

SedTRAILS supports mangrove propagules as a modular tracer mode with:

- passive horizontal advection based on `depth_avg_flow_velocity`
- optional sticky-depth settlement
- optional release-relative lifespan
- optional windage forcing
- mangrove-specific vertical position handling

Use `particle_type: mangrove` together with `tracer_methods.mangrove`.

## Minimal Structure

```yaml
particles:
  populations:
    - name: avicennia_sources
      particle_type: mangrove
      characteristics:
        species: avicennia_marina
        diffusion_coefficient: 0.0
      tracer_methods:
        mangrove: {}
      seeding:
        release_start: 2016-09-21 19:30:00
        quantity: 1
        strategy:
          random:
            bbox: "39400,16800 40600,17800"
            seed: 42
            nlocations: 10
```

If `flow_field_name` is omitted, SedTRAILS defaults mangrove populations to `depth_avg_flow_velocity`.

## Settlement

Sticky-depth settlement is configured under `tracer_methods.mangrove.settlement`.

```yaml
tracer_methods:
  mangrove:
    settlement:
      enabled: true
      method: sticky_depth
      depth_threshold: 0.1
      depth_mode: instantaneous
```

Supported depth modes:

- `instantaneous`: settle when the interpolated local `water_depth` at the current timestep is at or below the threshold.
- `max_over_simulation`: settle using `max_water_depth`, intended for legacy/thesis-style behavior.

When a mangrove particle settles:

- `status_settled = true`
- `status_buried = true`
- `status_mobile = false`
- `settlement_time` is written once, at first settlement

For mangroves, `status_buried` is an output compatibility flag representing settled/rooted state. For sediment tracers, `status_buried` still means burial below the active layer.

## Lifespan

Lifespan is measured relative to each particle's `release_start`.

```yaml
tracer_methods:
  mangrove:
    lifespan:
      enabled: true
      duration: 7D
```

The runtime rule is:

```text
status_alive = in_domain_and_not_left_boundary AND current_time < release_time + lifespan
```

`status_released` and `status_alive` remain separate.

## Windage

Windage adds a scaled wind component to the hydrodynamic velocity:

```text
u_particle = u_flow + P * u_wind
v_particle = v_flow + P * v_wind
```

with `P = windage.coefficient`.

By default, direction uses the meteorological `from` convention:

```text
u_wind = -speed * sin(direction_degrees)
v_wind = -speed * cos(direction_degrees)
```

### Constant windage

```yaml
tracer_methods:
  mangrove:
    windage:
      enabled: true
      coefficient: 0.02
      direction_convention: from_meteorological
      forcing:
        type: constant
        speed: 4.4
        direction: 292.5
```

You may also provide constant vector components directly:

```yaml
forcing:
  type: constant
  u: 4.0
  v: -1.5
```

### Point time series windage

```yaml
tracer_methods:
  mangrove:
    windage:
      enabled: true
      coefficient: 0.02
      forcing:
        type: timeseries_point
        times: [0S, 6H, 12H]
        speed_values: [3.0, 5.0, 4.0]
        direction_values: [270.0, 292.5, 315.0]
```

`times` may be numeric seconds or SedTRAILS duration strings.

### Spatial wind fields

If wind is already available on the Eulerian grid, reference either a vector field or separate scalar `u`/`v` fields:

```yaml
tracer_methods:
  mangrove:
    windage:
      enabled: true
      coefficient: 0.02
      forcing:
        type: field
        flow_field_name: wind_velocity
```

or:

```yaml
forcing:
  type: field
  u_field_name: wind_u
  v_field_name: wind_v
```

## Vertical Position

Mangrove particles use a different `z` convention from sediment tracers:

- floating/mobile mangroves: `z = bed_level + water_depth`
- settled mangroves: `z = bed_level`

This allows surface-following floating propagules and bed-fixed settled propagules within the same output structure.

## Example

See `examples/sedtrails-example-mangrove.yaml` for a complete tracked example.
