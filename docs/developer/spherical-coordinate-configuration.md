# Geodetic coordinate runtime

This document defines the coordinate contract used by projected, regional
geographic, and ocean-scale SedTRAILS simulations.

## Configuration

All settings are below `general.input_model`.

| Setting | Default | Allowed values | Meaning |
| ------- | ------- | -------------- | ------- |
| `coordinate_system` | `auto` | `auto`, `projected`, `geographic`, `spherical` | Representation of model x/y values |
| `source_crs` | `EPSG:4326` | Non-empty pyproj CRS text | CRS of geographic source coordinates |
| `runtime_geometry` | `auto` | `auto`, `planar`, `geodetic` | Horizontal geometry backend |
| `metric_crs` | `auto_utm` | CRS text or `null` | Metre CRS for planar geographic runs |
| `surface_model` | `sphere` | `from_crs`, `sphere`, `ellipsoid` | Requested Earth surface model |
| `earth_radius_m` | `6371008.8` | Positive number | Radius used by the spherical backend |
| `longitude_wrap` | `auto` | `auto`, `-180_180`, `0_360` | Public longitude convention |
| `velocity_basis` | `auto` | `auto`, `east_north`, `source_xy`, `grid_aligned` | Input horizontal vector basis |

`coordinate_system: spherical` is an alias for geographic longitude/latitude.
It is not a runtime backend name.

`runtime_geometry: auto` preserves projected input as planar geometry. For
geographic input it selects local UTM geometry when the domain fits one safe
zone and selects geodetic geometry otherwise.

## Runtime paths

```text
flowchart LR
    source["Read coordinates and metadata"] --> resolve["Resolve coordinate descriptor"]
    resolve --> choice{"Runtime geometry"}
    choice -->|"Projected source"| planar["Planar metre backend"]
    choice -->|"Regional lon/lat"| utm["Project once to local metre CRS"]
    choice -->|"Wide, polar, or seam-crossing lon/lat"| geodetic["Geodetic spherical backend"]
    geodetic --> ecef["ECEF topology and tangent-vector integration"]
    ecef --> native["Wrapped lon/lat output and restart"]
```

The planar path is the compatibility fast path. Geographic coordinates are
projected once during geometry initialization and inverse transformed only at
public output boundaries.

The geodetic path keeps particle positions as longitude/latitude and performs
topology, interpolation, advection, diffusion, and distance calculations on a
sphere. ECEF geometry removes the antimeridian seam and remains finite at the
poles. Vector fields must resolve to geographic east/north before this backend
is created.

The operational geodetic backend currently supports `surface_model: sphere`.
`from_crs` and `ellipsoid` are accepted by the general descriptor for planar
work but are rejected by geodetic geometry rather than silently approximated.

## Topology requirement

Geodetic geometry prefers authoritative triangular connectivity from the input
format. Planar Delaunay triangulation is not used for a global or
antimeridian-crossing mesh. When a format exposes only field points, the
converter may build outward spherical Delaunay faces from the ECEF convex hull
only for a bounded small mesh. Ocean-scale inputs without authoritative
topology fail with an actionable error instead of starting an unbounded Qhull
operation. Degenerate point sets fail clearly before particle creation.

Boundary behavior is topological. Each outer mesh edge may be classified as
`open`, `land`, or `unclassified`. Face walking follows intermediate faces so a
step that crosses several cells receives the class of the outer edge it
actually reaches.

## Scale behavior

- Static node and face geometry is built once and shared by populations.
- Connectivity and adjacency use 32-bit indices when the mesh permits it.
- Neighbor construction is array based and rejects non-manifold edges.
- Cached-face point location and RK4 stage interpolation use compiled,
  array-oriented ECEF kernels. Tree candidates are bounded and used only for
  unresolved points.
- Forcing slices carry an explicit generation token. Writable source arrays
  are converted to ECEF once per time slice and shared across particle chunks
  and populations without relying on array writeability.
- Format readers receive the exact source-field set needed by active runtime
  plans and an `inputs.max_eulerian_memory_mb` byte budget. They retain the
  interpolation bracket while shortening oversized requested time windows.
- Static grid metadata is cached by complete geometry and transform content in
  a count- and byte-bounded LRU. Repeated forcing windows reuse it safely.
- Particle movement and NetCDF coordinate transforms run in bounded chunks.
- Plotting samples particles and avoids loading unnecessary trajectory data.

Large production meshes must supply native connectivity. Set
`inputs.max_eulerian_memory_mb` for the memory available to one forcing window;
the default is 2048 MiB. The `netcdf.particle_chunk` setting controls output
chunking; particle stepping uses the same bounded default of 65536 particles
per operation.

## Seeding and output

Point and file seed coordinates use native source coordinates. Geodetic random
boxes are sampled uniformly in spherical surface area, including boxes where
`xmin > xmax` to cross the antimeridian. Geodetic transects follow the shortest
great-circle arc. Grid spacing is specified in metres and converted per
latitude.

Trajectory files and checkpoints carry coordinate descriptor version 2,
including the source CRS, runtime geometry, surface model, radius, longitude
convention, velocity basis, and projected runtime CRS when applicable.
Restarts reject incompatible descriptors. Writers transform planar geographic
runtime coordinates back to native lon/lat in particle chunks. Geodetic
runtime coordinates are already native and are serialized with the configured
longitude convention.

## Validation checklist

Changes to this subsystem should run:

1. Coordinate schema and transform tests.
2. Geodetic primitive and grid tests, including the seam and poles.
3. Seeding, boundary, writer, restart, dashboard, and trajectory tests.
4. Converter tests for every affected input format.
5. Ruff and trailing-whitespace checks on changed files.

Scale changes also require a bounded synthetic benchmark that records
construction memory, particle chunk size, and update throughput on the target
machine.
