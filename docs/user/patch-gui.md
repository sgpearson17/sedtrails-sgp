# Patch-coloured trajectory GUI

Use this desktop viewer to classify existing particle seeds into coloured square
patches and trace their pathways without changing the simulation configuration.
It uses the same bathymetry loader and metre-coordinate conversion as the seeding
GUI. XBeach, Delft3D4, D-Flow FM and SFINCS inputs are supported.
Python's Tkinter desktop support is required (included in standard Windows Python).

## Start

```bash
sedtrails viz patches
sedtrails viz patches --config path/to/case.yaml
sedtrails viz patches --config path/to/case.yaml --file path/to/sedtrails_results.nc
sedtrails viz patches --session path/to/case-patches.json
```

`--config` and `--session` are mutually exclusive. Without either option, open a
YAML or saved JSON using the file buttons. `--file` optionally loads or overrides
the output referenced by a saved session.

Relative input, seed and output paths are looked up beside the YAML first, then
in the current working directory, then in the YAML's ancestor directories. This
also supports cases that reference paths from a shared modeling-project root.
The existing seeding GUI bathymetry loader uses the same lookup order.

## Workflow

1. **Open SedTRAILS YAML**. First-timestep bathymetry and all configured seed
   particles appear. Bathymetry colours automatically span the input elevation
   range, including shallow laboratory domains. Seed strategies, quantities, population order and
   `file_points` filtering are evaluated using the simulation's own seeder.
   Opening another YAML keeps your patch names, positions, sizes and colours,
   background colour and rendering settings. The map refits the new grid and
   previous trajectories are cleared; load the new run's trajectories separately.
2. Set the patch **name**, lower-left **x/y**, **side length n** (defaults to
   1.5 m), and **RGB** using separate **R**, **G** and **B** boxes.
   The lower-left x/y fields are spin boxes: use their
   up/down arrows to nudge the value by 0.1 m, or type any value directly.
   Coordinates and side lengths are in metres, not grid-cell indices.
   For geographic grids, the viewer uses the YAML's metric CRS (or safe regional
   automatic UTM selection). Configure an explicit suitable metric CRS for
   wide, polar or antimeridian-crossing domains.
3. Set **Series row spacing (+y)** and **Series column spacing (+x)**, both
   defaulting to 1 m, then press **Add square / row-column series**. With
   rows/columns both 1, this adds one square. Larger values create an n-by-n
   series of squares offset by the configured spacing, with columns increasing
   x and rows increasing y; generated names include 1-based `[row,column]`
   indices. Set the spacing equal to the side length for touching squares, or
   larger for gaps between them. Add multiple series at different x/y origins
   as needed.
4. Select a patch row to edit its position, size or colour, then press
   **Update selected square**. Multiple rows can be selected for deletion.
   Names must be unique. Series members can be recoloured individually.
   The update, delete and **Live update** controls sit above the series fields.
   Enable **Live update**, select one row or click a square on the map, then
   use the keyboard arrow keys while the table or map has focus to move the
   square in **0.1 m increments**: left/right change x and down/up change y.
   Seed and trajectory colours and legend counts update with each move.
   The x/y spin-box arrows also apply immediately in live mode.
   Text-entry keys keep their normal editing behavior. Turn off toolbar
   pan/zoom to select or move squares on the map. Disable live mode to return
   to manual updates. Overlapping squares are selected using the same
   last-row-wins rule as classification.
5. Enter **Background point RGB** and press **Apply colours / rendering**.
   It also has separate **R**, **G** and **B** boxes. Each channel must be an
   integer in **0..255**, for example R = 255, G = 80, B = 0.
   This background colour applies to unclassified particles, not the map itself.
   Choose a **Bathymetry colour map** of **SEAWAD**, **Vintage** or
   **Grayscale** (darker at lower values); the map updates immediately.
   Uncheck **Background seeding points and trajectories** to hide unclassified
   (not patch-assigned) seeds and tracks, for example to focus on classified
   patches only.
6. Choose **Load associated trajectories** to open
   `outputs.directory/sedtrails_results.nc`, or choose a NetCDF manually.
7. Pan/zoom with the Matplotlib toolbar. Toggle seeding points, trajectories,
   square overlays or background seeds/trajectories independently. The legend
   reports seed-particle counts.
8. **Save patch JSON** stores the YAML/output paths, patch geometry and colours,
   background point colour, colour map choice, layer visibility, particle
   limit and current map extent. **Open patch JSON** restores these and
   reloads the source data.
   JSON stores references, not copies of the underlying YAML or NetCDF files;
   keep those files available. Relative paths in hand-edited JSON resolve beside
   the JSON. The simulation YAML and seed files are never edited.
9. **Export map PNG** exports the current map extent, visible layers, legend
   and bathymetry colourbar at 180 dpi, without desktop controls. The control
   panel has a scrollbar for smaller displays.

## Classification and trajectory identity

Square edges are included. When squares overlap or share an edge, the **last
patch in the table wins**. Delete/re-add a patch to give it highest precedence.
Unclassified seeds use the background point colour.

Each trajectory is matched from its initial saved coordinates back to the exact
YAML seed location, allowing for NetCDF float32 rounding. Its entire path keeps
that seed's patch colour, even after crossing another patch. Changing a patch
reclassifies origins, never later positions. Missing coordinates break a path;
they are not joined across gaps.

Choose the associated full trajectory file, including its simulation-start
timestep. Checkpoints, restarted outputs starting later, missing initial
positions and positions inconsistent with the configured seeds are rejected
rather than silently assigning incorrect origin colours. Stochastic seed
strategies must reproduce the simulation's original locations.

For responsiveness, trajectories use the existing viewer's deterministic
sampling and a 2,000,000-track-point rendering budget. The default particle limit
is 10,000; change **Maximum plotted trajectories** and apply to reload the
sample. The status reports displayed versus total trajectories. All YAML seeds
are still classified and displayed independently of trajectory sampling.
