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
sedtrails viz patches --config path/to/another-case.yaml --session path/to/case-patches.json
```

Combine `--config` and `--session` to apply a saved patch layout to a run.
Without either option, open a YAML or patch JSON using the file buttons.
Portable patch JSONs do not reference source files: opening one without
`--config` loads its patches, ready for you to choose a YAML.
`--file` optionally loads trajectories; a YAML must also be available.

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
   background colour, rendering settings, custom title and current x/y limits.
   Previous trajectories are cleared; load the new run's trajectories separately.
   The new run starts at bathymetry frame 0. Use **Fit map** if you want to
   refit the view to its grid instead of keeping your existing field of view.
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
   Enable **Live update**, then select one or more rows: **Ctrl-click** toggles
   individual rows and **Shift-click** selects a range. Use the keyboard arrow
   keys while the table or map has focus to move **all selected squares together**
   in **0.1 m increments**: left/right change x and down/up change y.
   Their relative spacing, sizes and colours are preserved, and the selection
   stays active for repeated nudges. Seed and trajectory colours and legend
   counts update immediately. Clicking a square on the map selects that square
   alone. The editor shows the first selected row; **Update selected square**
   and the live x/y spin-box arrows still require a single selected row.
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
   **Bathymetry colour range [m]** defaults to **Auto**, which scales colours to
   the current bed frame and shows its limits in the disabled Min/Max fields.
   Choose **Manual**, enter **Min** and **Max** elevations in metres, then press
   **Apply colours / rendering**. Limits must be finite with Min < Max.
   Manual limits stay fixed across bathymetry timesteps, palette changes and
   YAML loads. Values outside the range remain visible in the endpoint colours;
   colourbar end triangles indicate clipping. Switch back to **Auto** to resume
   automatic scaling immediately.
   Uncheck **Background seeding points and trajectories** to hide unclassified
   (not patch-assigned) seeds and tracks, for example to focus on classified
   patches only.
6. Choose **Load associated trajectories** to open
   `outputs.directory/sedtrails_results.nc`, or choose a NetCDF manually.
7. Pan/zoom with the Matplotlib toolbar. Toggle seeding points, trajectories,
   square overlays or background seeds/trajectories independently. The legend
   reports seed-particle counts. The **Plot view** x/y limit fields follow
   toolbar navigation. Type both endpoints for each axis and press
   **Apply limits / title** to set an exact field of view. Limits must be finite
   and different; reversed limits are allowed. Loading YAMLs or trajectories,
   editing patches, or changing rendering settings does not reset the view.
   **Fit map** deliberately resets the limits to the current run's grid.
   Enter a custom **Plot title** and apply it; leave it blank to use the YAML
   filename. Custom titles persist when changing YAMLs.
8. Use **Bathymetry timestep (0-based)** to choose a saved bed frame. Spin-box
   arrows update immediately; after typing an index, press Enter or
   **Show bathymetry timestep**. The label shows the bathymetry variable,
   its source time (when available), and the number of frames. For XBeach,
   `zb_mean` is preferred over `zb`, so the selector follows `meantime` when
   present, not the trajectory or `globaltime` axis. Static or single-frame
   inputs have a disabled spin box. Each change loads just the selected bed
   frame and, in **Auto** mode, rescales its colourbar to that frame's elevation
   range. **Manual** colour limits remain unchanged. Seeds,
   trajectories, their colours, title and zoom are unchanged; trajectories
   are not truncated or animated by this control. Invalid frames report an
   error and keep the previous display.
9. **Save patch JSON** stores **only patch definitions** (names, geometry and
   RGB colours) and the **background point RGB**, plus a format version.
   YAML/trajectory paths, axis/colour limits, title, timestep, visibility and rendering
   preferences are not saved. **Open patch JSON** replaces the current patches
   and background colour without changing the loaded run or plot preferences.
   You can open and save layouts before choosing a YAML.
   Older version-1 JSONs remain readable: the GUI's **Open patch JSON** imports
   only their patch data. For backward compatibility, launching with
   `--session old-session.json` still opens its referenced run and rendering
   settings. An explicit `--config` overrides those source paths; add `--file`
   to load that run's trajectories. Relative legacy source paths resolve beside
   the JSON. Saving always writes the portable version-2 format.
   The simulation YAML and seed files are never edited.
10. **Export map PNG** exports the current map extent, title, selected bathymetry,
   visible layers, legend
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
