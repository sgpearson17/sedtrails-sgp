"""Desktop analysis of seed patches and trajectories coloured by their origins."""

from __future__ import annotations

from copy import copy, deepcopy
from dataclasses import asdict, dataclass, field, replace
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

import numpy as np

from sedtrails.application_interfaces.seeding_gui import (
    BathymetryViewData,
    SeedingGuiError,
    _build_gui_coordinate_display,
    _create_masked_triangulation,
    _resolve_relative_path,
    load_bathymetry_view_data,
    load_config,
)

RGB = tuple[int, int, int]


def parse_rgb(value: str | list[int] | tuple[int, ...]) -> RGB:
    """Read an RGB triplet in the unambiguous integer range 0..255."""
    parts = value.replace(',', ' ').split() if isinstance(value, str) else value
    if not isinstance(parts, (list, tuple)) or len(parts) != 3:
        raise ValueError('RGB must contain three integers in 0..255, for example 255, 80, 0.')
    if any(isinstance(v, bool) or str(v).strip() not in {str(i) for i in range(256)} for v in parts):
        raise ValueError('RGB channels must be integers in 0..255.')
    return int(parts[0]), int(parts[1]), int(parts[2])


@dataclass(frozen=True)
class SquarePatch:
    name: str
    x: float
    y: float
    size: float
    rgb: RGB

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError('Patch name must not be empty.')
        if not all(np.isfinite(v) for v in (self.x, self.y, self.size)) or self.size <= 0:
            raise ValueError('Patch coordinates must be finite and side length n must be positive.')
        if not np.isfinite(self.x + self.size) or not np.isfinite(self.y + self.size):
            raise ValueError('Patch extent must be finite.')
        object.__setattr__(self, 'rgb', parse_rgb(self.rgb))


def classify_points(points: np.ndarray, patches: list[SquarePatch]) -> np.ndarray:
    """Return patch indices (-1 for background); the last square wins overlaps."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('Seed points must be a finite (N, 2) array.')
    labels = np.full(len(points), -1, dtype=int)
    for index, patch in enumerate(patches):
        inside = (
            (points[:, 0] >= patch.x)
            & (points[:, 0] <= patch.x + patch.size)
            & (points[:, 1] >= patch.y)
            & (points[:, 1] <= patch.y + patch.size)
        )
        labels[inside] = index
    return labels


def point_colours(labels: np.ndarray, patches: list[SquarePatch], background: RGB) -> np.ndarray:
    palette = np.asarray([background, *(patch.rgb for patch in patches)], dtype=float) / 255.0
    return palette[labels + 1]


@dataclass
class PatchSession:
    config_path: Path | None = None
    results_path: Path | None = None
    patches: list[SquarePatch] = field(default_factory=list)
    background: RGB = (120, 120, 120)
    show_seeds: bool = True
    show_tracks: bool = True
    show_patches: bool = True
    show_background: bool = True
    colormap: str = 'SEAWAD'
    max_particles: int = 10_000
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    title: str = ''
    bathymetry_index: int = 0
    bathymetry_clim: tuple[float, float] | None = None
    population_visibility: dict[str, bool] = field(default_factory=dict)
    population_marker_sizes: dict[str, float] = field(default_factory=dict)
    show_endpoints: bool = False

    def validate(self) -> None:
        from sedtrails.pathway_visualizer.colormaps import available_bathymetry_colormaps

        self.background = parse_rgb(self.background)
        if not isinstance(self.title, str):
            raise ValueError('Plot title must be text.')
        if type(self.bathymetry_index) is not int or self.bathymetry_index < 0:
            raise ValueError('Bathymetry timestep must be a nonnegative integer.')
        if type(self.max_particles) is not int or self.max_particles < 1:
            raise ValueError('Maximum plotted trajectories must be a positive integer.')
        for value in (self.show_seeds, self.show_tracks, self.show_patches, self.show_background, self.show_endpoints):
            if type(value) is not bool:
                raise ValueError('Layer visibility values must be true or false.')
        if any(not isinstance(name, str) or type(value) is not bool for name, value in self.population_visibility.items()):
            raise ValueError('Population visibility must map names to true or false.')
        for name, size in self.population_marker_sizes.items():
            if (
                not isinstance(name, str) or isinstance(size, bool)
                or not isinstance(size, (int, float)) or not np.isfinite(size) or size <= 0
            ):
                raise ValueError('Population marker sizes must be finite positive numbers (points squared).')
        available = available_bathymetry_colormaps()
        if not isinstance(self.colormap, str) or self.colormap.lower() not in {name.lower() for name in available}:
            raise ValueError(f'colormap must be one of: {", ".join(available)}.')
        names = [patch.name for patch in self.patches]
        if len(set(names)) != len(names):
            raise ValueError('Patch names must be unique.')
        for limits in (self.xlim, self.ylim):
            if limits is not None and (len(limits) != 2 or not np.isfinite(limits).all() or limits[0] == limits[1]):
                raise ValueError('Map limits must contain two different finite coordinates.')
        if self.bathymetry_clim is not None:
            limits = self.bathymetry_clim
            if len(limits) != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]:
                raise ValueError('Bathymetry colour limits must contain two finite values with minimum < maximum.')
            if (
                not np.isfinite(float(limits[1]) - float(limits[0]))
                or not (np.diff(np.linspace(*limits, 33)) > 0).all()
            ):
                raise ValueError('Bathymetry colour limits must span 33 distinct finite contour levels.')


def save_session(path: str | Path, session: PatchSession) -> None:
    session.validate()
    payload = {'version': 2, 'patches': [asdict(patch) for patch in session.patches], 'background': session.background}
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_session(path: str | Path) -> PatchSession:
    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError('Unsupported patch session; expected a JSON object with version 1 or 2.')
    version = payload.pop('version', None)
    if type(version) is not int or version not in (1, 2):
        raise ValueError('Unsupported patch session; expected version 1 or 2.')
    try:
        if version == 2 and set(payload) - {'patches', 'background'}:
            raise ValueError('Version 2 stores only patches and background RGB.')
        for key in ('config_path', 'results_path'):
            value = payload.get(key)
            payload[key] = (path.parent / value).resolve() if value is not None else None
        payload['patches'] = [SquarePatch(**item) for item in payload.get('patches', [])]
        for key in ('xlim', 'ylim', 'bathymetry_clim'):
            if payload.get(key) is not None:
                payload[key] = tuple(payload[key])
        session = PatchSession(**payload)
        session.validate()
    except (TypeError, KeyError, ValueError) as exc:
        raise ValueError(f'Invalid patch session: {exc}') from exc
    return session


class PatchModel:
    """Load model data independently of the desktop widgets for headless use."""

    def __init__(self, session: PatchSession):
        from sedtrails.particle_tracer.particle_seeder import ParticleFactory, PopulationConfig

        session.validate()
        if session.config_path is None:
            raise ValueError('Open a SedTRAILS YAML file first.')
        self.config_path = Path(session.config_path)
        self.session = session
        self.config = load_config(self.config_path)
        self.bathymetry = load_bathymetry_view_data(self.config_path, time_index=session.bathymetry_index)
        self.coordinates = _build_gui_coordinate_display(
            self.bathymetry.x,
            self.bathymetry.y,
            coordinate_system=self.bathymetry.coordinate_system,
            source_crs=self.bathymetry.source_crs,
            metric_crs=self.bathymetry.metric_crs,
        )
        self.map_x, self.map_y = self.coordinates.source_to_display(self.bathymetry.x, self.bathymetry.y)
        populations = self.config.get('particles', {}).get('populations')
        if not isinstance(populations, list) or not populations:
            raise ValueError('The YAML has no particles.populations entries.')
        seed_arrays = []
        self.population_names = [population.get('name', f'Population {i + 1}') for i, population in enumerate(populations)]
        if len(set(self.population_names)) != len(self.population_names):
            raise ValueError('Population names must be unique for plotting.')
        for population in populations:
            config = PopulationConfig(deepcopy(population))
            for key in ('path', 'poly'):
                value = config.strategy_settings.get(key)
                if isinstance(value, str):
                    config.strategy_settings[key] = str(_resolve_relative_path(value, self.config_path.parent))
            if self.coordinates.transform is not None:
                config.strategy_settings['_coordinate_transform'] = self.coordinates.transform
            arrays = ParticleFactory.create_particle_arrays(
                config,
                reference_date=self.config.get('general', {})
                .get('input_model', {})
                .get('reference_date', '1970-01-01'),
            )
            seed_arrays.append(np.column_stack((arrays['x'], arrays['y'])))
        self.seeds = np.concatenate(seed_arrays)
        self.seed_population_ids = np.repeat(np.arange(len(populations)), [len(points) for points in seed_arrays])
        if not len(self.seeds) or not np.isfinite(self.seeds).all():
            raise ValueError('The YAML did not produce finite seeding points.')
        self.tracks: np.ndarray | None = None
        self.track_seed_indices = np.empty(0, dtype=int)
        self.track_population_ids = np.empty(0, dtype=int)
        self.track_times = np.empty(0, dtype=float)
        self.track_valid = np.empty((0, 0), dtype=bool)
        self.track_reference: np.datetime64 | None = None
        self.total_tracks = 0
        if session.results_path is not None:
            self.load_tracks(session.results_path)

    def associated_results(self) -> Path:
        directory = self.config.get('outputs', {}).get('directory')
        if not directory:
            raise ValueError('The YAML has no outputs.directory; choose a NetCDF output file.')
        return _resolve_relative_path(directory, self.config_path.parent) / 'sedtrails_results.nc'

    def load_tracks(self, path: str | Path) -> None:
        from scipy.spatial import cKDTree
        from sedtrails.pathway_visualizer.trajectories import (
            _decode_population_names,
            _reference_datetime64,
            _sample_dataset,
            _time_values_as_seconds,
            _trajectory_arrays,
            read_netcdf,
        )

        self.session.validate()
        with read_netcdf(Path(path)) as dataset:
            if 'x' not in dataset or 'y' not in dataset or dataset['x'].ndim != 2:
                raise ValueError('Choose a trajectory output, not a single-timestep checkpoint.')
            start = self.config.get('time', {}).get('start')
            if start is not None and 'time' in dataset:
                reference = _reference_datetime64(dataset, dataset['time'])
                if reference is None:
                    reference = np.datetime64(self.config['general']['input_model']['reference_date'])
                expected = float((np.datetime64(start) - reference) / np.timedelta64(1, 's'))
                times = _time_values_as_seconds(dataset, dataset['time'])
                if not times.size or not np.isclose(times.flat[0], expected, rtol=0, atol=1e-6):
                    raise ValueError('Output must include the YAML simulation start, not a restart timestep.')
            particle_dim = dataset['x'].dims[1]
            indexed = dataset.assign_coords(_patch_particle_index=(particle_dim, np.arange(dataset.sizes[particle_dim])))
            selected, total, _ = _sample_dataset(indexed, max_particles=self.session.max_particles)
            x, y, time_values = _trajectory_arrays(selected)
            x, y = self.coordinates.source_to_display(x, y)
            origins = np.column_stack((x[:, 0], y[:, 0]))
            if not len(origins) or not np.isfinite(origins).all():
                raise ValueError('Output must contain finite initial positions for all plotted particles.')
            if 'population_id' in selected:
                raw_ids = np.asarray(selected['population_id'].values)
                if raw_ids.shape != (len(origins),) or not np.issubdtype(raw_ids.dtype, np.integer):
                    raise ValueError('Output population_id must contain one integer per particle.')
                if 'population_name' in dataset:
                    count = dataset['population_name'].shape[0]
                    names = _decode_population_names(dataset, count)
                    if len(set(names)) != len(names):
                        raise ValueError('Output population names must be unique.')
                    if all(name in self.population_names for name in names):
                        mapping = np.array([self.population_names.index(name) for name in names])
                    elif names == [f'population_{i}' for i in range(len(self.population_names))]:
                        # Some writers store generated names rather than YAML names.
                        expected_counts = np.bincount(self.seed_population_ids, minlength=len(self.population_names))
                        if 'population_count' not in dataset or not np.array_equal(
                            dataset['population_count'].values, expected_counts,
                        ):
                            raise ValueError('Output population counts do not match the YAML populations.')
                        mapping = np.arange(len(self.population_names))
                    else:
                        raise ValueError('Output population names do not match the YAML populations.')
                else:
                    mapping = np.arange(len(self.population_names))
                if np.any(raw_ids < 0) or np.any(raw_ids >= len(mapping)):
                    raise ValueError('Output contains an unknown population_id.')
                population_ids = mapping[raw_ids]
            elif total == len(self.seeds):
                population_ids = self.seed_population_ids[selected['_patch_particle_index'].values]
            elif len(self.population_names) == 1:
                population_ids = np.zeros(len(origins), dtype=int)
            else:
                raise ValueError('Cannot identify trajectory populations: output needs population metadata or YAML seed order.')
            distances = np.empty(len(origins))
            indices = np.empty(len(origins), dtype=int)
            for population_id in np.unique(population_ids):
                rows = np.flatnonzero(population_ids == population_id)
                seeds = np.flatnonzero(self.seed_population_ids == population_id)
                if not seeds.size:
                    raise ValueError('Output references a population with no YAML seeds.')
                distances[rows], local_indices = cKDTree(self.seeds[seeds]).query(origins[rows])
                indices[rows] = seeds[local_indices]
            # NetCDF coordinates may be float32: match back to exact YAML seeds before classifying.
            tolerance = np.maximum(1e-5, np.max(np.abs(origins), axis=1) * np.finfo(np.float32).eps * 4)
            if np.any(distances > tolerance):
                raise ValueError(
                    'Output initial positions do not match the YAML seeds. Choose the associated full '
                    'trajectory file, including its initial timestep (not a restart or a different run).'
                )
            tracks = np.stack((x, y), axis=-1)
            track_times = time_values[0].copy()
            if not np.isfinite(track_times).all():
                raise ValueError('Output trajectory times must be finite.')
            track_reference = _reference_datetime64(dataset, dataset['time']) if 'time' in dataset else None
            track_valid = np.isfinite(tracks).all(axis=-1)
            for name in ('status_alive', 'status_domain', 'status_released', 'status_left_domain'):
                if name not in selected:
                    continue
                status = selected[name]
                if dict(status.sizes) != dict(selected['x'].sizes):
                    raise ValueError(f'Output {name} must have the same dimensions as trajectory coordinates.')
                values = status.transpose(*selected['x'].dims).values.T
                track_valid &= values == (0 if name == 'status_left_domain' else 1)
        self.tracks = tracks
        self.track_seed_indices = indices
        self.track_population_ids = population_ids
        self.track_times = track_times
        self.track_valid = track_valid
        self.track_reference = track_reference
        self.total_tracks = total
        self.session.results_path = Path(path).resolve()

    def population_mask(self, population_ids: np.ndarray) -> np.ndarray:
        visible = np.array([self.session.population_visibility.get(name, True) for name in self.population_names])
        return visible[population_ids]

    def marker_sizes(self, population_ids: np.ndarray) -> np.ndarray:
        sizes = np.array([self.session.population_marker_sizes.get(name, 9.0) for name in self.population_names])
        return sizes[population_ids]


class PatchPlot:
    """Render only the map, so PNG export excludes desktop controls."""

    def __init__(self, figure: Any, model: PatchModel):
        import matplotlib.tri as mtri

        self.figure = figure
        self.model = model
        self.ax = figure.add_subplot(111)
        self.colormap_name = ''
        self._bathymetry_clim: tuple[float, float] | None = None
        self.triangulation = _create_masked_triangulation(model.map_x, model.map_y, mtri_module=mtri)
        self.bathymetry_artist: Any = None
        self.colorbar: Any = None
        self._draw_bathymetry(model.session.colormap)
        self.ax.set(xlabel='x [m]', ylabel='y [m]')
        self.ax.set_aspect('equal', adjustable='box')
        self.home_xlim, self.home_ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.ax.set_autoscale_on(False)
        self.artists: list[Any] = []
        self.draw()
        if model.session.xlim is not None:
            self.ax.set_xlim(model.session.xlim)
        if model.session.ylim is not None:
            self.ax.set_ylim(model.session.ylim)

    def _draw_bathymetry(
        self,
        colormap_name: str,
        *,
        bathymetry: BathymetryViewData | None = None,
        triangulation: Any = None,
    ) -> None:
        from sedtrails.pathway_visualizer.colormaps import bathymetry_colormap

        limits = self.model.session.bathymetry_clim
        if colormap_name == self.colormap_name and limits == self._bathymetry_clim and bathymetry is None:
            return
        data = self.model.bathymetry if bathymetry is None else bathymetry
        mesh = self.triangulation if triangulation is None else triangulation
        if limits is None:
            vmin, vmax = float(data.values.min()), float(data.values.max())
            if vmin == vmax:
                vmin, vmax = vmin - 0.5, vmax + 0.5
            levels = 32
            extend = 'neither'
        else:
            vmin, vmax = limits
            levels = np.linspace(vmin, vmax, 33)
            extend = 'both'
        cmap, norm = bathymetry_colormap(colormap_name, vmin=vmin, vmax=vmax)
        artist = self.ax.tricontourf(mesh, data.values, levels=levels, cmap=cmap, norm=norm, extend=extend)
        # Updating only the normalization leaves old contour boundaries and extension triangles.
        if self.colorbar is not None:
            self.colorbar.remove()
        self.colorbar = self.figure.colorbar(artist, ax=self.ax, label=f'{data.variable} [m]')
        if self.bathymetry_artist is not None:
            self.bathymetry_artist.remove()
        self.bathymetry_artist = artist
        self.triangulation = mesh
        self.vmin, self.vmax = vmin, vmax
        self.colormap_name = colormap_name
        self._bathymetry_clim = limits

    def set_bathymetry(self, time_index: int) -> None:
        import matplotlib.tri as mtri

        model = self.model
        data = load_bathymetry_view_data(model.config_path, time_index=time_index)
        map_x, map_y = model.coordinates.source_to_display(data.x, data.y)
        triangulation = _create_masked_triangulation(map_x, map_y, mtri_module=mtri)
        self._draw_bathymetry(model.session.colormap, bathymetry=data, triangulation=triangulation)
        model.bathymetry = data
        model.map_x, model.map_y = map_x, map_y
        model.session.bathymetry_index = time_index
        self.figure.canvas.draw_idle()

    def set_colormap(self, colormap_name: str) -> None:
        xlim, ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self._draw_bathymetry(colormap_name)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.figure.canvas.draw_idle()

    def draw(self) -> None:
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D

        model, session = self.model, self.model.session
        session.validate()
        self.ax.set_title(session.title or model.config_path.name)
        labels = classify_points(model.seeds, session.patches)
        colours = point_colours(labels, session.patches, session.background)
        for artist in self.artists:
            artist.remove()
        self.artists.clear()
        self.track_artist = None
        self.seed_artist = None
        self.endpoint_artist = None
        visible_seeds = model.population_mask(model.seed_population_ids)
        self.track_order = np.empty(0, dtype=int)
        if session.show_patches:
            for patch in session.patches:
                rectangle = Rectangle(
                    (patch.x, patch.y),
                    patch.size,
                    patch.size,
                    facecolor=np.asarray(patch.rgb) / 255,
                    edgecolor=np.asarray(patch.rgb) / 255,
                    alpha=0.25,
                    zorder=2,
                )
                self.artists.append(self.ax.add_patch(rectangle))
        if (session.show_tracks or session.show_endpoints) and model.tracks is not None:
            track_labels = labels[model.track_seed_indices]
            track_positions = np.flatnonzero(model.population_mask(model.track_population_ids))
            if not session.show_background:
                track_positions = track_positions[track_labels[track_positions] != -1]
            self.track_order = track_positions[np.argsort(track_labels[track_positions], kind='stable')]
        if session.show_tracks and model.tracks is not None:
            self.track_artist = LineCollection(
                np.ma.masked_invalid(model.tracks[self.track_order]),
                colors=colours[model.track_seed_indices[self.track_order]],
                linewidths=0.7,
                zorder=3,
            )
            self.ax.add_collection(self.track_artist, autolim=False)
            self.artists.append(self.track_artist)
        if session.show_endpoints and model.tracks is not None:
            endpoints = model.tracks[self.track_order, -1]
            self.endpoint_artist = self.ax.scatter(
                *endpoints.T, c=colours[model.track_seed_indices[self.track_order]],
                s=model.marker_sizes(model.track_population_ids[self.track_order]),
                edgecolors='none', zorder=5,
            )
            self.artists.append(self.endpoint_artist)
        if session.show_seeds:
            seed_mask = visible_seeds & (True if session.show_background else labels != -1)
            self.seed_artist = self.ax.scatter(
                *model.seeds[seed_mask].T, c=colours[seed_mask],
                s=model.marker_sizes(model.seed_population_ids[seed_mask]), edgecolors='none', zorder=4
            )
            self.artists.append(self.seed_artist)
        handles = []
        if session.show_background:
            handles.append(
                Line2D(
                    [],
                    [],
                    marker='o',
                    linestyle='',
                    color=np.asarray(session.background) / 255,
                    label=f'Background ({np.count_nonzero((labels == -1) & visible_seeds)})',
                )
            )
        handles.extend(
            Line2D(
                [],
                [],
                marker='s',
                linestyle='',
                color=np.asarray(p.rgb) / 255,
                label=f'{p.name} ({np.count_nonzero((labels == i) & visible_seeds)})',
            )
            for i, p in enumerate(session.patches)
        )
        self.ax.legend(handles=handles, loc='upper right', fontsize='small')
        self.figure.canvas.draw_idle()

    def remember_view(self) -> None:
        self.model.session.xlim = tuple(self.ax.get_xlim())
        self.model.session.ylim = tuple(self.ax.get_ylim())

    def export_png(self, path: str | Path) -> None:
        self.figure.savefig(path, format='png', dpi=180, facecolor=self.figure.get_facecolor())

    def draw_animation_frame(self, frame: int) -> tuple[Any, ...]:
        """Update moving markers and optional growing trails on the existing map."""
        model = self.model
        if model.tracks is None:
            raise ValueError('Load trajectories before exporting a GIF.')
        order = self.track_order
        visible = model.track_valid[order, frame]
        labels = classify_points(model.seeds, model.session.patches)
        colours = point_colours(labels, model.session.patches, model.session.background)
        artists = []
        if self.endpoint_artist is not None:
            rows = order[visible]
            self.endpoint_artist.set_offsets(model.tracks[rows, frame])
            self.endpoint_artist.set_facecolors(colours[model.track_seed_indices[rows]])
            self.endpoint_artist.set_sizes(model.marker_sizes(model.track_population_ids[rows]))
            artists.append(self.endpoint_artist)
        if self.track_artist is not None:
            positions = model.tracks[order, :frame + 1]
            valid = model.track_valid[order, :frame + 1]
            self.track_artist.set_segments(np.where(valid[:, :, None], positions, np.nan))
            artists.append(self.track_artist)
        seconds = model.track_times[frame]
        elapsed = (seconds - model.track_times[0]) / 3600
        time_label = f't = {elapsed:.2f} h'
        if model.track_reference is not None:
            timestamp = model.track_reference + np.timedelta64(int(round(seconds * 1e9)), 'ns')
            time_label = f'{np.datetime_as_string(timestamp, unit="s")}  ({time_label})'
        title = self.ax.set_title(
            f'{model.session.title or model.config_path.name}\n'
            f'{time_label}; {np.count_nonzero(visible):,} plotted particles in domain'
        )
        return (*artists, title)

    def export_gif(self, path: str | Path, *, frame_step: int = 5, fps: int = 10) -> None:
        """Animate the sampled tracks on an isolated copy of the current map view."""
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        if type(frame_step) is not int or frame_step < 1:
            raise ValueError('GIF frame step must be a positive integer.')
        if type(fps) is not int or not 1 <= fps <= 100:
            raise ValueError('GIF frames per second must be an integer in 1..100.')
        if self.model.tracks is None:
            raise ValueError('Load trajectories before exporting a GIF.')
        snapshot = copy(self.model)
        snapshot.session = replace(deepcopy(self.model.session), show_endpoints=True)
        figure = Figure(figsize=(8, 8), layout='constrained')
        FigureCanvasAgg(figure)
        try:
            plot = PatchPlot(figure, snapshot)
            plot.ax.set_xlim(self.ax.get_xlim())
            plot.ax.set_ylim(self.ax.get_ylim())
            if not plot.track_order.size:
                raise ValueError('No trajectories match the selected populations and background filter.')
            frames = np.arange(0, len(snapshot.track_times), frame_step)
            if frames[-1] != len(snapshot.track_times) - 1:
                frames = np.append(frames, len(snapshot.track_times) - 1)
            animation = FuncAnimation(
                figure, plot.draw_animation_frame, frames=frames, interval=1000 / fps,
                blit=False, repeat=False,
            )
            animation.save(str(path), writer=PillowWriter(fps=fps), dpi=100)
        finally:
            figure.clear()


class PatchGuiApp:
    """Tk desktop controls with a Matplotlib map and navigation toolbar."""

    def __init__(self, root: Any, session: PatchSession):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.session = session
        self.model: PatchModel | None = None
        self.plot: PatchPlot | None = None
        self.canvas: Any = None
        self.toolbar: Any = None
        root.title('SedTRAILS - Patch colours and trajectories')
        root.geometry('1400x900')
        root.minsize(1100, 760)
        self.map_frame = ttk.Frame(root)
        self.map_frame.pack(side='left', fill='both', expand=True)
        sidebar = ttk.Frame(root)
        sidebar.pack(side='right', fill='y')
        panel_canvas = tk.Canvas(sidebar, width=410, highlightthickness=0)
        scrollbar = ttk.Scrollbar(sidebar, orient='vertical', command=panel_canvas.yview)
        scrollbar.pack(side='right', fill='y')
        panel_canvas.pack(side='left', fill='y', expand=True)
        panel_canvas.configure(yscrollcommand=scrollbar.set)
        panel = ttk.Frame(panel_canvas, padding=10)
        panel_canvas.create_window((0, 0), window=panel, anchor='nw')
        panel.bind('<Configure>', lambda _event: panel_canvas.configure(scrollregion=panel_canvas.bbox('all')))
        self.fields: dict[str, Any] = {}
        self.visibility: dict[str, Any] = {}

        def button(parent, text, command):
            widget = ttk.Button(parent, text=text, command=lambda: self._run(command))
            widget.pack(fill='x', pady=2)
            return widget

        def rgb_fields(parent, colour):
            frame = ttk.Frame(parent)
            variables = {}
            for channel, initial in zip('rgb', colour, strict=True):
                ttk.Label(frame, text=channel.upper()).pack(side='left', padx=(0, 2))
                value = tk.StringVar(value=str(initial))
                ttk.Entry(frame, textvariable=value, width=4).pack(side='left', padx=(0, 5))
                variables[channel] = value
            return frame, variables

        files = ttk.LabelFrame(panel, text='Files', padding=5)
        files.pack(fill='x')
        button(files, 'Open SedTRAILS YAML...', self.open_yaml)
        button(files, 'Open patch JSON...', self.open_json)
        self.file_label = ttk.Label(files, text='No YAML loaded', wraplength=340)
        self.file_label.pack(fill='x')
        button(files, 'Load associated trajectories', self.load_associated)
        button(files, 'Choose trajectory NetCDF...', self.choose_tracks)

        view = ttk.LabelFrame(panel, text='Plot view', padding=5)
        view.pack(fill='x', pady=7)
        limits = ttk.Frame(view)
        limits.pack(fill='x')
        self.view_limits: dict[str, Any] = {}
        for row, axis in enumerate(('x', 'y')):
            ttk.Label(limits, text=f'{axis} limits [m]').grid(row=row, column=0, sticky='w')
            for column, endpoint in enumerate(('min', 'max'), start=1):
                value = tk.StringVar()
                ttk.Entry(limits, textvariable=value, width=13).grid(row=row, column=column, padx=2, pady=2)
                self.view_limits[f'{axis}{endpoint}'] = value
        ttk.Label(view, text='Plot title (blank = YAML filename)').pack(anchor='w')
        self.plot_title = tk.StringVar(value=session.title)
        ttk.Entry(view, textvariable=self.plot_title).pack(fill='x')
        button(view, 'Apply limits / title', self.apply_view)
        button(view, 'Fit map', self.fit_map)

        patches = ttk.LabelFrame(panel, text='Patches (last row wins overlaps)', padding=5)
        patches.pack(fill='x', pady=7)
        self.table = ttk.Treeview(patches, columns=('x', 'y', 'n', 'rgb'), height=6, selectmode='extended')
        self.table.heading('#0', text='Name')
        self.table.column('#0', width=90)
        for column in ('x', 'y', 'n', 'rgb'):
            self.table.heading(column, text=column)
            self.table.column(column, width=60 if column != 'rgb' else 95)
        self.table.pack(fill='x')
        self.table.bind('<<TreeviewSelect>>', self.select_patch)
        self.table.bind('<KeyPress>', self._on_patch_arrow)
        self.live_update = tk.BooleanVar(value=False)
        editor = ttk.Frame(patches)
        editor.pack(fill='x')
        defaults = (
            ('name', 'Name / series prefix', 'Patch'),
            ('x', 'Lower-left x [m]', '0'),
            ('y', 'Lower-left y [m]', '0'),
            ('size', 'Side length n [m]', '1.5'),
        )
        for row, (key, label, default) in enumerate(defaults):
            ttk.Label(editor, text=label).grid(row=row, column=0, sticky='w', pady=2)
            value = tk.StringVar(value=default)
            if key in ('x', 'y'):
                ttk.Spinbox(
                    editor,
                    textvariable=value,
                    from_=-1_000_000,
                    to=1_000_000,
                    increment=0.1,
                    width=20,
                    command=lambda: self._run(self._live_position_changed),
                ).grid(row=row, column=1, sticky='ew')
            else:
                ttk.Entry(editor, textvariable=value, width=22).grid(row=row, column=1, sticky='ew')
            self.fields[key] = value
        ttk.Label(editor, text='RGB (0..255)').grid(row=4, column=0, sticky='w', pady=2)
        rgb_frame, self.patch_rgb = rgb_fields(editor, (255, 0, 0))
        rgb_frame.grid(row=4, column=1, sticky='ew')
        button(patches, 'Update selected square', self.update_patch)
        button(patches, 'Delete selected square(s)', self.delete_patches)
        ttk.Checkbutton(
            patches,
            text='Live update (arrow keys: 0.1 m)',
            variable=self.live_update,
            command=lambda: self._run(self.toggle_live_update),
        ).pack(fill='x', pady=2)
        series = ttk.Frame(patches)
        series.pack(fill='x')
        for row, (key, label) in enumerate(
            (
                ('rows', 'Series rows (+y)'),
                ('columns', 'Series columns (+x)'),
                ('row_spacing', 'Series row spacing (+y) [m]'),
                ('column_spacing', 'Series column spacing (+x) [m]'),
            )
        ):
            ttk.Label(series, text=label).grid(row=row, column=0, sticky='w', pady=2)
            value = tk.StringVar(value='1')
            ttk.Entry(series, textvariable=value, width=22).grid(row=row, column=1, sticky='ew')
            self.fields[key] = value
        button(patches, 'Add square / row-column series', self.add_patches)

        self.population_controls = ttk.LabelFrame(panel, text='Populations', padding=5)
        self.population_controls.pack(fill='x', pady=7)
        self.population_visibility: dict[str, Any] = {}
        self.population_marker_sizes: dict[str, Any] = {}
        ttk.Label(self.population_controls, text='Open a YAML to select populations.').pack(anchor='w')

        layers = ttk.LabelFrame(panel, text='Layers and rendering', padding=5)
        layers.pack(fill='x')
        ttk.Label(layers, text='Background point RGB (0..255)').pack(anchor='w')
        background_frame, self.background = rgb_fields(layers, session.background)
        background_frame.pack(fill='x')
        self.maximum = tk.StringVar(value=str(session.max_particles))
        ttk.Label(layers, text='Maximum plotted trajectories').pack(anchor='w')
        ttk.Entry(layers, textvariable=self.maximum).pack(fill='x')
        from sedtrails.pathway_visualizer.colormaps import available_bathymetry_colormaps

        self.colormap = tk.StringVar(value=session.colormap)
        ttk.Label(layers, text='Bathymetry colour map').pack(anchor='w')
        ttk.Combobox(
            layers,
            textvariable=self.colormap,
            values=available_bathymetry_colormaps(),
            state='readonly',
        ).pack(fill='x')
        self._colormap_trace_suspended = False
        self.colormap.trace_add(
            'write', lambda *_args: None if self._colormap_trace_suspended else self._run(self.apply_layers)
        )
        ttk.Label(layers, text='Bathymetry colour range [m]').pack(anchor='w')
        self.bathymetry_scale = tk.StringVar(value='Auto' if session.bathymetry_clim is None else 'Manual')
        self.bathymetry_scale_box = ttk.Combobox(
            layers, textvariable=self.bathymetry_scale, values=('Auto', 'Manual'), state='readonly'
        )
        self.bathymetry_scale_box.pack(fill='x')
        self.bathymetry_scale_box.bind('<<ComboboxSelected>>', lambda _event: self._run(self._change_bathymetry_scale))
        colour_limits = ttk.Frame(layers)
        colour_limits.pack(fill='x')
        self.bathymetry_colour_limits: dict[str, Any] = {}
        self.bathymetry_colour_entries: dict[str, Any] = {}
        for key, initial in zip(('min', 'max'), session.bathymetry_clim or ('', ''), strict=True):
            ttk.Label(colour_limits, text=key.capitalize()).pack(side='left')
            value = tk.StringVar(value=str(initial))
            entry = ttk.Entry(
                colour_limits,
                textvariable=value,
                width=13,
                state='disabled' if session.bathymetry_clim is None else 'normal',
            )
            entry.pack(side='left', padx=(2, 5))
            self.bathymetry_colour_limits[key] = value
            self.bathymetry_colour_entries[key] = entry
        ttk.Label(layers, text='Bathymetry timestep (0-based)').pack(anchor='w')
        self.bathymetry_index = tk.StringVar(value='0')
        self.bathymetry_spinner = ttk.Spinbox(
            layers,
            textvariable=self.bathymetry_index,
            from_=0,
            to=0,
            increment=1,
            state='disabled',
            command=lambda: self._run(self.apply_bathymetry),
        )
        self.bathymetry_spinner.pack(fill='x')
        self.bathymetry_spinner.bind('<Return>', lambda _event: self._run(self.apply_bathymetry))
        button(layers, 'Show bathymetry timestep', self.apply_bathymetry)
        self.bathymetry_label = ttk.Label(layers, text='No YAML loaded', wraplength=340)
        self.bathymetry_label.pack(fill='x')
        for key, label in (
            ('show_seeds', 'Seeding points'),
            ('show_tracks', 'Trajectories'),
            ('show_endpoints', 'Trajectory endpoints'),
            ('show_patches', 'Patch squares'),
            ('show_background', 'Background seeding points and trajectories'),
        ):
            value = tk.BooleanVar(value=getattr(session, key))
            self.visibility[key] = value
            ttk.Checkbutton(layers, text=label, variable=value, command=lambda: self._run(self.apply_layers)).pack(
                anchor='w'
            )
        button(layers, 'Apply colours / rendering', self.apply_layers)
        button(panel, 'Save patch JSON...', self.save_json)
        button(panel, 'Export map PNG...', self.export_png)
        gif_controls = ttk.LabelFrame(panel, text='GIF animation', padding=5)
        gif_controls.pack(fill='x', pady=7)
        self.gif_frame_step = tk.StringVar(value='5')
        self.gif_fps = tk.StringVar(value='10')
        for text, variable in (('Frame step (loaded samples)', self.gif_frame_step), ('Frames per second', self.gif_fps)):
            row = ttk.Frame(gif_controls)
            row.pack(fill='x')
            ttk.Label(row, text=text).pack(side='left')
            ttk.Entry(row, textvariable=variable, width=8).pack(side='right')
        button(gif_controls, 'Export animation GIF...', self.export_gif)
        self.status = ttk.Label(panel, text='Open a YAML file or a patch layout JSON.', wraplength=350)
        self.status.pack(fill='x', pady=6)
        self._refresh_table()
        if session.config_path is not None:
            self._install_model(PatchModel(session))

    def _run(self, action) -> None:
        from tkinter import messagebox

        try:
            action()
        except Exception as exc:
            # GUI callbacks must surface failures without terminating the event loop.
            self.status.configure(text=f'Error: {exc}')
            messagebox.showerror('SedTRAILS patch GUI', str(exc), parent=self.root)

    def _install_model(self, model: PatchModel) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        from matplotlib.figure import Figure

        figure = Figure(figsize=(9, 8), layout='constrained')
        plot = PatchPlot(figure, model)
        for widget in self.map_frame.winfo_children():
            widget.destroy()
        self.model, self.session, self.plot = model, model.session, plot
        self._update_population_controls()
        plot.ax.callbacks.connect('xlim_changed', self._sync_view)
        plot.ax.callbacks.connect('ylim_changed', self._sync_view)
        self._sync_view()
        self.plot_title.set(self.session.title)
        self._update_bathymetry_controls()
        self.canvas = FigureCanvasTkAgg(figure, master=self.map_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.map_frame, pack_toolbar=False)
        self.toolbar.pack(side='bottom', fill='x')
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.canvas.get_tk_widget().bind('<KeyPress>', self._on_patch_arrow, add='+')
        self.canvas.mpl_connect('button_press_event', self._on_map_click)
        self.canvas.draw()
        self.file_label.configure(text=str(self.session.config_path))
        for channel, value in zip('rgb', self.session.background, strict=True):
            self.background[channel].set(str(value))
        self.maximum.set(str(self.session.max_particles))
        self._colormap_trace_suspended = True
        self.colormap.set(self.session.colormap)
        self._colormap_trace_suspended = False
        for key, variable in self.visibility.items():
            variable.set(getattr(self.session, key))
        self.table.selection_remove(*self.table.selection())
        self._refresh_table()
        self._report()

    def _require_model(self) -> PatchModel:
        if self.model is None:
            raise ValueError('Open a SedTRAILS YAML file first.')
        return self.model

    def _update_population_controls(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        for widget in self.population_controls.winfo_children():
            widget.destroy()
        self.population_visibility.clear()
        self.population_marker_sizes.clear()
        ttk.Label(self.population_controls, text='Show population / marker area [pt²]').pack(anchor='w')
        for name in self._require_model().population_names:
            row = ttk.Frame(self.population_controls)
            row.pack(fill='x')
            visible = tk.BooleanVar(value=self.session.population_visibility.get(name, True))
            size = tk.StringVar(value=str(self.session.population_marker_sizes.get(name, 9.0)))
            self.population_visibility[name] = visible
            self.population_marker_sizes[name] = size
            ttk.Checkbutton(
                row, text=name, variable=visible, command=lambda: self._run(self.apply_layers),
            ).pack(side='left', fill='x', expand=True)
            entry = ttk.Entry(row, textvariable=size, width=8)
            entry.pack(side='right')
            entry.bind('<Return>', lambda _event: self._run(self.apply_layers))
        ttk.Label(
            self.population_controls, text='Sizes apply to seeds and endpoints. Enter or Apply to update.',
            wraplength=340,
        ).pack(anchor='w')

    def _require_plot(self) -> PatchPlot:
        if self.plot is None:
            raise ValueError('Open a SedTRAILS YAML file first.')
        return self.plot

    def _sync_view(self, _axes: Any = None) -> None:
        if self.plot is None:
            return
        self.plot.remember_view()
        for axis, values in (('x', self.plot.ax.get_xlim()), ('y', self.plot.ax.get_ylim())):
            for endpoint, value in zip(('min', 'max'), values, strict=True):
                self.view_limits[f'{axis}{endpoint}'].set(str(float(value)))

    def apply_view(self) -> None:
        model = self._require_model()
        plot = self._require_plot()
        candidate = deepcopy(self.session)
        candidate.xlim = (float(self.view_limits['xmin'].get()), float(self.view_limits['xmax'].get()))
        candidate.ylim = (float(self.view_limits['ymin'].get()), float(self.view_limits['ymax'].get()))
        candidate.title = self.plot_title.get()
        candidate.validate()
        xlim, ylim = candidate.xlim, candidate.ylim
        model.session = self.session = candidate
        plot.ax.set_xlim(xlim)
        plot.ax.set_ylim(ylim)
        plot.ax.set_title(candidate.title or model.config_path.name)
        self.canvas.draw_idle()

    def fit_map(self) -> None:
        plot = self._require_plot()
        plot.ax.set_xlim(plot.home_xlim)
        plot.ax.set_ylim(plot.home_ylim)
        self.canvas.draw_idle()

    def _update_bathymetry_controls(self) -> None:
        data = self._require_model().bathymetry
        self.bathymetry_index.set(str(data.time_index))
        self.bathymetry_spinner.configure(to=data.time_count - 1, state='normal' if data.time_count > 1 else 'disabled')
        self.bathymetry_label.configure(text=f'{data.variable}: {data.time_label} ({data.time_count} frames)')
        self._update_bathymetry_colour_controls()

    def _update_bathymetry_colour_controls(self) -> None:
        plot = self._require_plot()
        manual = self.session.bathymetry_clim is not None
        self.bathymetry_scale.set('Manual' if manual else 'Auto')
        for key, value in (('min', plot.vmin), ('max', plot.vmax)):
            self.bathymetry_colour_limits[key].set(str(float(value)))
            self.bathymetry_colour_entries[key].configure(state='normal' if manual else 'disabled')

    def _change_bathymetry_scale(self) -> None:
        for entry in self.bathymetry_colour_entries.values():
            entry.configure(state='normal' if self.bathymetry_scale.get() == 'Manual' else 'disabled')
        if self.model is not None:
            self.apply_layers()

    def apply_bathymetry(self) -> None:
        plot = self._require_plot()
        try:
            plot.set_bathymetry(int(self.bathymetry_index.get()))
        finally:
            self._update_bathymetry_controls()
        self._report()

    def _report(self) -> None:
        model = self._require_model()
        shown = 0 if model.tracks is None else len(model.tracks)
        visible_seeds = np.count_nonzero(model.population_mask(model.seed_population_ids))
        visible_tracks = np.count_nonzero(model.population_mask(model.track_population_ids))
        crs = model.coordinates.metric_crs or 'native projected coordinates'
        self.status.configure(
            text=f'{visible_seeds:,}/{len(model.seeds):,} seeds in selected populations; '
            f'{visible_tracks:,}/{shown:,} loaded trajectories in selected populations '
            f'({model.total_tracks:,} total).\n'
            f'{crs}. Trajectories retain their seed colour. Large outputs are sampled '
            'deterministically (at most 2,000,000 plotted track points).'
        )

    def _refresh_table(self) -> None:
        for row in self.table.get_children()[len(self.session.patches) :]:
            self.table.delete(row)
        for index, patch in enumerate(self.session.patches):
            values = (patch.x, patch.y, patch.size, ','.join(map(str, patch.rgb)))
            if self.table.exists(str(index)):
                self.table.item(str(index), text=patch.name, values=values)
            else:
                self.table.insert('', 'end', iid=str(index), text=patch.name, values=values)

    def open_yaml(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(parent=self.root, filetypes=[('SedTRAILS YAML', '*.yaml *.yml')])
        if path:
            candidate = self._read_layers()
            candidate.config_path = Path(path)
            candidate.results_path = None
            candidate.bathymetry_index = 0
            self._install_model(PatchModel(candidate))

    def open_json(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(parent=self.root, filetypes=[('Patch layout', '*.json')])
        if path:
            layout = load_session(path)
            candidate = deepcopy(self.session)
            candidate.patches, candidate.background = layout.patches, layout.background
            candidate.validate()
            self.session = candidate
            for channel, value in zip('rgb', candidate.background, strict=True):
                self.background[channel].set(str(value))
            self.table.selection_remove(*self.table.selection())
            self._refresh_table()
            if self.model is not None:
                self.model.session = candidate
                self._require_plot().draw()
                self._report()
            else:
                self.status.configure(text='Patch layout loaded. Open a SedTRAILS YAML to display it.')

    def _read_layers(self) -> PatchSession:
        candidate = deepcopy(self.session)
        candidate.background = parse_rgb(' '.join(value.get() for value in self.background.values()))
        candidate.max_particles = int(self.maximum.get())
        candidate.colormap = self.colormap.get()
        for name, variable in self.population_visibility.items():
            candidate.population_visibility[name] = variable.get()
            candidate.population_marker_sizes[name] = float(self.population_marker_sizes[name].get())
        if self.bathymetry_scale.get() == 'Auto':
            candidate.bathymetry_clim = None
        elif self.bathymetry_scale.get() == 'Manual':
            candidate.bathymetry_clim = (
                float(self.bathymetry_colour_limits['min'].get()),
                float(self.bathymetry_colour_limits['max'].get()),
            )
        else:
            raise ValueError('Bathymetry colour range must be Auto or Manual.')
        for key, variable in self.visibility.items():
            setattr(candidate, key, variable.get())
        candidate.validate()
        return candidate

    def apply_layers(self) -> None:
        model = self._require_model()
        candidate = self._read_layers()
        if candidate.max_particles != self.session.max_particles and candidate.results_path is not None:
            # Load before committing state, so a failed reload leaves the old plot/session intact.
            old_session = model.session
            model.session = candidate
            try:
                model.load_tracks(candidate.results_path)
            except Exception:
                model.session = old_session
                raise
        model.session = self.session = candidate
        self.plot.set_colormap(candidate.colormap)
        self.plot.draw()
        self._update_bathymetry_colour_controls()
        self._report()

    def _read_patch(self) -> SquarePatch:
        return SquarePatch(
            self.fields['name'].get().strip(),
            float(self.fields['x'].get()),
            float(self.fields['y'].get()),
            float(self.fields['size'].get()),
            parse_rgb(' '.join(value.get() for value in self.patch_rgb.values())),
        )

    def add_patches(self) -> None:
        self._require_model()
        patch = self._read_patch()
        rows, columns = int(self.fields['rows'].get()), int(self.fields['columns'].get())
        row_spacing, column_spacing = (
            float(self.fields['row_spacing'].get()),
            float(self.fields['column_spacing'].get()),
        )
        if rows < 1 or columns < 1 or rows * columns > 10_000:
            raise ValueError('Series rows and columns must be positive, with at most 10,000 patches.')
        if row_spacing <= 0 or column_spacing <= 0:
            raise ValueError('Series row and column spacing must be positive.')
        additions = [
            SquarePatch(
                patch.name if rows * columns == 1 else f'{patch.name}[{row + 1},{column + 1}]',
                patch.x + column * column_spacing,
                patch.y + row * row_spacing,
                patch.size,
                patch.rgb,
            )
            for row in range(rows)
            for column in range(columns)
        ]
        candidate = deepcopy(self.session)
        candidate.patches.extend(additions)
        candidate.validate()
        self.session.patches = candidate.patches
        self._refresh_table()
        self.plot.draw()

    def select_patch(self, _event=None) -> None:
        selection = self.table.selection()
        if not selection:
            return
        patch = self.session.patches[int(selection[0])]
        for key in ('name', 'x', 'y', 'size'):
            self.fields[key].set(str(getattr(patch, key)))
        for channel, value in zip('rgb', patch.rgb, strict=True):
            self.patch_rgb[channel].set(str(value))
        self.fields['rows'].set('1')
        self.fields['columns'].set('1')
        self.fields['row_spacing'].set('1')
        self.fields['column_spacing'].set('1')

    def update_patch(self) -> None:
        selection = self.table.selection()
        if len(selection) != 1:
            raise ValueError('Select one patch to update.')
        self._commit_patches({int(selection[0]): self._read_patch()})

    def _commit_patches(self, patches: dict[int, SquarePatch]) -> None:
        candidate = deepcopy(self.session)
        for index, patch in patches.items():
            candidate.patches[index] = patch
        candidate.validate()
        self.session.patches = candidate.patches
        self._refresh_table()
        self._require_plot().draw()

    def toggle_live_update(self) -> None:
        if self.live_update.get():
            self.table.focus_set()
            self.status.configure(
                text='Live update: select patch rows with Ctrl/Shift-click, or click a square on the map. '
                'Arrow keys move all selected patches by 0.1 m. Turn off toolbar pan/zoom first.'
            )
        elif self.model is not None:
            self._report()

    def _live_position_changed(self) -> None:
        if self.live_update.get():
            self.update_patch()

    def _on_patch_arrow(self, event: Any) -> str | None:
        offsets = {'Left': (-0.1, 0), 'Right': (0.1, 0), 'Down': (0, -0.1), 'Up': (0, 0.1)}
        if not self.live_update.get() or event.keysym not in offsets:
            return None
        if self.toolbar is not None and self.toolbar.mode:
            return None
        self._run(lambda: self._move_selected_patches(*offsets[event.keysym]))
        return 'break'

    def _move_selected_patches(self, dx: float, dy: float) -> None:
        selection = self.table.selection()
        if not selection:
            raise ValueError('Select at least one patch for live movement.')
        delta_x, delta_y = Decimal(str(dx)), Decimal(str(dy))
        patches = {}
        for row in selection:
            index = int(row)
            patch = self.session.patches[index]
            patches[index] = replace(
                patch,
                x=float(Decimal(str(patch.x)) + delta_x),
                y=float(Decimal(str(patch.y)) + delta_y),
            )
        self._commit_patches(patches)
        moved = patches[int(selection[0])]
        self.fields['x'].set(str(moved.x))
        self.fields['y'].set(str(moved.y))

    def _on_map_click(self, event: Any) -> None:
        if (
            not self.live_update.get()
            or self.plot is None
            or not self.session.show_patches
            or self.toolbar.mode
            or event.button != 1
            or event.inaxes is not self.plot.ax
            or event.xdata is None
            or event.ydata is None
        ):
            return
        index = classify_points(np.array([[event.xdata, event.ydata]]), self.session.patches)[0]
        if index >= 0:
            row = str(index)
            self.table.selection_set(row)
            self.table.focus(row)
            self.table.see(row)
            self.select_patch()
            self.canvas.get_tk_widget().focus_set()

    def delete_patches(self) -> None:
        selected = {int(index) for index in self.table.selection()}
        if not selected:
            raise ValueError('Select the patch(es) to delete.')
        self.session.patches = [p for i, p in enumerate(self.session.patches) if i not in selected]
        self.table.selection_remove(*self.table.selection())
        self._refresh_table()
        self.plot.draw()

    def load_associated(self) -> None:
        self._load_tracks(self._require_model().associated_results())

    def choose_tracks(self) -> None:
        from tkinter import filedialog

        self._require_model()
        path = filedialog.askopenfilename(parent=self.root, filetypes=[('SedTRAILS output', '*.nc')])
        if path:
            self._load_tracks(Path(path))

    def _load_tracks(self, path: Path) -> None:
        model = self._require_model()
        candidate = self._read_layers()
        old_session = model.session
        model.session = candidate
        try:
            model.load_tracks(path)
        except Exception:
            model.session = old_session
            raise
        self.session = candidate
        plot = self._require_plot()
        plot.set_colormap(candidate.colormap)
        plot.draw()
        self._update_bathymetry_colour_controls()
        self._report()

    def save_json(self) -> None:
        from tkinter import filedialog

        candidate = deepcopy(self.session)
        candidate.background = parse_rgb(' '.join(value.get() for value in self.background.values()))
        candidate.validate()
        path = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension='.json',
            filetypes=[('Patch layout', '*.json')],
            initialfile=(self.session.config_path.stem + '-patches.json')
            if self.session.config_path
            else 'patches.json',
        )
        if path:
            save_session(path, candidate)
            self.session = candidate
            if self.model is not None:
                self.model.session = candidate
                self._require_plot().draw()
            self.status.configure(text=f'Saved patch configuration: {path}')

    def export_png(self) -> None:
        from tkinter import filedialog

        self.apply_layers()
        path = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension='.png',
            filetypes=[('PNG image', '*.png')],
            initialfile=self.session.config_path.stem + '-patches.png',
        )
        if path:
            self.plot.export_png(path)
            self.status.configure(text=f'Exported current map view: {path}')

    def export_gif(self) -> None:
        from tkinter import filedialog

        self.apply_layers()
        plot = self._require_plot()
        if self._require_model().tracks is None:
            raise ValueError('Load trajectories before exporting a GIF.')
        frame_step, fps = int(self.gif_frame_step.get()), int(self.gif_fps.get())
        if frame_step < 1 or not 1 <= fps <= 100:
            raise ValueError('GIF frame step must be >= 1 and frames per second must be in 1..100.')
        path = filedialog.asksaveasfilename(
            parent=self.root, defaultextension='.gif', filetypes=[('GIF animation', '*.gif')],
            initialfile=self.session.config_path.stem + '-patches.gif',
        )
        if not path:
            return
        self.status.configure(text='Exporting GIF; please wait. Large animations may take several minutes.')
        self.root.configure(cursor='watch')
        self.root.update_idletasks()
        try:
            plot.export_gif(path, frame_step=frame_step, fps=fps)
        finally:
            self.root.configure(cursor='')
            self._report()
        self.status.configure(text=f'Exported animation: {path}')


def launch_patch_gui(
    config_path: str | Path | None = None,
    session_path: str | Path | None = None,
    results_path: str | Path | None = None,
) -> None:
    import tkinter as tk

    session = load_session(session_path) if session_path else PatchSession()
    if config_path is not None:
        session.config_path = Path(config_path).resolve()
        session.results_path = None
        session.bathymetry_index = 0
    if results_path is not None:
        if session.config_path is None:
            raise SeedingGuiError('A SedTRAILS YAML (--config) is required to load trajectories.')
        session.results_path = Path(results_path).resolve()
    root = tk.Tk()
    root.protocol('WM_DELETE_WINDOW', root.quit)
    try:
        app = PatchGuiApp(root, session)
        root.mainloop()
        del app
    finally:
        root.destroy()
