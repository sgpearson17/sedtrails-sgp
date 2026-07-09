# Visualization

This page is a placeholder for SedTRAILS visualization guidance beyond the live monitoring dashboard. It can be expanded with examples for plotting model input fields, output trajectories, particle statistics, and publication-style figures.

For the real-time run dashboard, see [Monitoring Dashboard](dashboard.md). For output file inspection and trajectory plotting commands, see [Output](output.md).

## Bathymetry colormaps

SedTRAILS has reusable bathymetry colormaps for input-data checks and visualization tools. These are currently used by the seeding setup GUI and can also be reused from Python plotting scripts.

Available named colormaps:

- `SEAWAD`: a blue-white-brown-green bathymetry palette. Copyright: (c) Stuart G. Pearson, 2022. (CC BY 4.0).
- `Vintage`: a vintage Wadden Sea map-inspired bathymetry palette. Copyright: (c) Stuart G. Pearson, 2022. (CC BY 4.0), based on Rijkswaterstaat Studiedienst Hoorn (1943); see Elias et al. 2019.

In the seeding setup GUI, use the colormap selector to switch between `SEAWAD` and `Vintage`. The color-limit boxes can be used to change the displayed bathymetry range without changing the underlying input data.

Python placeholder:

```python
from sedtrails.pathway_visualizer.colormaps import bathymetry_colormap

cmap, norm = bathymetry_colormap("Vintage", vmin=-12, vmax=6)
```

## Trajectory plots

Reusable trajectory utilities are available from `sedtrails.pathway_visualizer.sedtrails_plotting`.
Bathymetry backgrounds should use `SEAWAD` by default. Particle-age and source-distance plots use ordinary sequential colormaps such as `viridis`.

```python
from sedtrails.pathway_visualizer.trajectories import read_netcdf
from sedtrails.pathway_visualizer.sedtrails_plotting import (
    load_from_xarray,
    plot_trajectories_by_age,
    plot_trajectories_by_baseline,
    animate_particles,
)

ds = read_netcdf("sedtrails_results.nc")
tr = load_from_xarray(ds)

plot_trajectories_by_age(tr, cmap="viridis")
plot_trajectories_by_baseline(tr, rotation_deg=0.0, cmap="viridis")
fig, anim = animate_particles(tr, color_mode="baseline", cmap="viridis")
```

The helper `source_distance_from_baseline` computes the baseline-distance values directly. By default, the baseline origin is inferred from the minimum model-domain `x` and `y` coordinates, and users can pass a rotation angle.

Trajectory statistics similar to the legacy MATLAB `analyze_pathways.m` workflow can be exported with:

```python
from sedtrails.pathway_visualizer.sedtrails_plotting import compute_particle_stats, stats_to_csv

stats = compute_particle_stats(tr)
stats_to_csv(stats, "particle_statistics.csv")
```

## Connectivity analysis

SedTRAILS can compile a source-to-sink connectivity adjacency matrix from a trajectory NetCDF file.
The default method follows the current sediment-connectivity workflow:

- source nodes are based on initial particle positions;
- particles with identical initial positions are grouped into one source node;
- adjacency weights are raw particle-position counts;
- repeated visits by the same particle are counted;
- self-links are included;
- all saved timesteps are counted.

From the command line:

```text
sedtrails network adjacency --input sedtrails_results.nc --output sedtrails_connectivity_adjacency.nc
```

Useful alternatives:

```text
sedtrails network adjacency --mode final
sedtrails network adjacency --mode time
sedtrails network adjacency --polygon-mode n_cells --n-cells 8
sedtrails network adjacency --unique-visits
sedtrails network adjacency --zero-self-links
sedtrails network adjacency --weight probability
```

From Python:

```python
from sedtrails.simulation_analysis.connectivity import (
    adjacency_to_digraph,
    compile_adjacency_from_results,
    compute_network_metrics,
    read_adjacency_netcdf,
)

summary = compile_adjacency_from_results(
    "sedtrails_results.nc",
    "sedtrails_connectivity_adjacency.nc",
)
adjacency = read_adjacency_netcdf(summary.output_file)
graph = adjacency_to_digraph(adjacency)
metrics = compute_network_metrics(graph)
```

Common node metrics can also be plotted on the source-cell coordinates:

```python
from sedtrails.simulation_analysis.connectivity import (
    compute_node_metrics,
    detect_communities,
    plot_node_metric_map,
    plot_node_metric_bars,
    plot_community_map,
)

node_metrics = compute_node_metrics(graph)
community_labels, modularity = detect_communities(graph)

plot_node_metric_map(graph, node_metrics["pagerank"], metric_name="PageRank")
plot_node_metric_bars(node_metrics["weighted_in_degree"], metric_name="Weighted in-degree")
plot_community_map(graph, community_labels)
```

The connectivity tools use NetworkX for graph analysis and Shapely for polygon operations. This keeps the analysis close to widely used, citable Python packages rather than custom graph implementations.

Example notebooks are available in `examples/notebooks/trajectory_visualization.ipynb` and `examples/notebooks/connectivity_analysis.ipynb`.
