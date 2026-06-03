"""Small desktop helper for choosing SedTRAILS seed points from input data."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from sedtrails.application_interfaces.validator import SedtrailsYamlLoader, YAMLConfigValidator

SEEDING_MODES = ('points', 'transect', 'random', 'grid')


class SeedingGuiError(RuntimeError):
    """Raised for user-facing setup GUI errors."""


@dataclass(frozen=True)
class BathymetryViewData:
    """Spatial data needed by the seeding GUI map."""

    x: np.ndarray
    y: np.ndarray
    values: np.ndarray
    variable: str
    input_file: Path


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a SedTRAILS YAML file without converting timestamp strings."""

    path = Path(config_path)
    if not path.exists():
        raise SeedingGuiError(f'Configuration file not found: {path}')

    try:
        with path.open('r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=SedtrailsYamlLoader)
    except Exception as exc:
        raise SeedingGuiError(f'Could not read configuration: {exc}') from exc

    if not isinstance(config, dict):
        raise SeedingGuiError('Configuration file did not contain a YAML object.')

    return config


def get_population_names(config: dict[str, Any]) -> list[str]:
    """Return configured population names in file order."""

    populations = _get_populations(config)
    return [str(pop.get('name', f'population_{idx + 1}')) for idx, pop in enumerate(populations)]


def add_population_from_existing(
    config: dict[str, Any],
    *,
    source_population_name: str | None,
    new_population_name: str | None = None,
) -> tuple[dict[str, Any], str]:
    """Return a copied config with a new population cloned from an existing one."""

    updated = deepcopy(config)
    populations = _get_populations(updated)
    source_population = _select_population(populations, source_population_name)
    source_index = populations.index(source_population)
    resolved_name = _unique_population_name(
        get_population_names(updated),
        new_population_name or f"{source_population.get('name', 'population')}_copy",
    )
    new_population = deepcopy(source_population)
    new_population['name'] = resolved_name
    populations.insert(source_index + 1, new_population)
    return updated, resolved_name


def rename_population(config: dict[str, Any], *, old_name: str, new_name: str) -> dict[str, Any]:
    """Return a copied config with one population renamed."""

    cleaned_name = new_name.strip()
    if not cleaned_name:
        raise SeedingGuiError('Population name cannot be empty.')

    updated = deepcopy(config)
    populations = _get_populations(updated)
    existing = [str(pop.get('name')) for pop in populations if pop.get('name') != old_name]
    if cleaned_name in existing:
        raise SeedingGuiError(f"Population '{cleaned_name}' already exists.")
    population = _select_population(populations, old_name)
    population['name'] = cleaned_name
    return updated


def remove_population(config: dict[str, Any], *, population_name: str) -> dict[str, Any]:
    """Return a copied config with one population removed."""

    updated = deepcopy(config)
    populations = _get_populations(updated)
    if len(populations) <= 1:
        raise SeedingGuiError('Cannot remove the only population.')
    population = _select_population(populations, population_name)
    populations.remove(population)
    return updated


def default_seeded_config_path(config_path: str | Path) -> Path:
    """Return the default copied-config path beside the source YAML file."""

    source = Path(config_path)
    suffix = source.suffix or '.yaml'
    return source.with_name(f'{source.stem}-seeded{suffix}')


def load_bathymetry_view_data(
    config_path: str | Path,
    *,
    format_override: str | None = None,
    variable: str | None = None,
) -> BathymetryViewData:
    """Load first-timestep map data for the GUI from a SedTRAILS config."""

    config_file = Path(config_path)
    config = load_config(config_file)
    input_model = config.get('general', {}).get('input_model', {})
    input_format = format_override or input_model.get('format')
    if input_format != 'fm_netcdf':
        raise SeedingGuiError(f"Unsupported input format for setup GUI: {input_format!r}. Only 'fm_netcdf' is supported.")

    data_path = config.get('inputs', {}).get('data')
    if not data_path:
        raise SeedingGuiError('Configuration is missing inputs.data.')

    input_file = _resolve_relative_path(data_path, config_file.parent)
    if not input_file.exists():
        raise SeedingGuiError(f'Input data file not found: {input_file}')

    try:
        import xarray as xr

        dataset = xr.open_dataset(input_file, decode_timedelta=True)
    except Exception as exc:
        raise SeedingGuiError(f'Could not load input data: {exc}') from exc

    missing_coordinates = [name for name in ('net_xcc', 'net_ycc') if name not in dataset]
    if missing_coordinates:
        missing = ', '.join(missing_coordinates)
        raise SeedingGuiError(f'Missing coordinate variable(s): {missing}')

    variable_name = _resolve_bathymetry_variable(dataset, variable)
    try:
        values = _first_timestep_values(dataset[variable_name])
        x = np.asarray(dataset['net_xcc'].values, dtype=float).reshape(-1)
        y = np.asarray(dataset['net_ycc'].values, dtype=float).reshape(-1)
    except Exception as exc:
        raise SeedingGuiError(f'Could not extract map data: {exc}') from exc
    finally:
        close = getattr(dataset, 'close', None)
        if close is not None:
            close()

    values = np.asarray(values, dtype=float).reshape(-1)
    if not (len(x) == len(y) == len(values)):
        raise SeedingGuiError(
            f'Map arrays have incompatible lengths: x={len(x)}, y={len(y)}, {variable_name}={len(values)}.'
        )

    return BathymetryViewData(x=x, y=y, values=values, variable=variable_name, input_file=input_file)


def update_config_for_file_points(
    config: dict[str, Any],
    *,
    population_name: str | None,
    points_path: str,
) -> dict[str, Any]:
    """Return a copied config with one population using the file_points strategy."""

    updated = deepcopy(config)
    populations = _get_populations(updated)
    population = _select_population(populations, population_name)
    seeding = population.setdefault('seeding', {})
    seeding['strategy'] = {
        'file_points': {
            'path': points_path,
            'has_header': False,
            'x_col': 0,
            'y_col': 1,
            'deduplicate': False,
            'dropna': False,
            'stride': 1,
        }
    }
    return updated


def write_points_file(points: list[tuple[float, float]], output_path: str | Path) -> Path:
    """Write selected seed points as two whitespace-separated columns."""

    if not points:
        raise SeedingGuiError('No seed points selected.')

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for x, y in points:
            f.write(f'{float(x):.12g} {float(y):.12g}\n')
    return path


def generate_transect_points(vertices: list[tuple[float, float]], points_per_segment: int) -> list[tuple[float, float]]:
    """Generate equally spaced points from clicked endpoint pairs."""

    if points_per_segment < 1:
        raise SeedingGuiError('Transect points per segment must be at least 1.')
    if len(vertices) < 2 or len(vertices) % 2 != 0:
        raise SeedingGuiError('Transect mode needs an even number of clicked endpoints.')

    points: list[tuple[float, float]] = []
    for start, end in zip(vertices[0::2], vertices[1::2], strict=True):
        x1, y1 = start
        x2, y2 = end
        if points_per_segment == 1:
            points.append((float(x1), float(y1)))
            continue
        for idx in range(points_per_segment):
            fraction = idx / (points_per_segment - 1)
            points.append((float(x1 + fraction * (x2 - x1)), float(y1 + fraction * (y2 - y1))))
    return points


def generate_random_points_in_polygon(
    polygon: list[tuple[float, float]],
    *,
    nlocations: int,
    seed: int,
) -> list[tuple[float, float]]:
    """Generate uniformly sampled random points inside a polygon."""

    if nlocations < 1:
        raise SeedingGuiError('Random nlocations must be at least 1.')
    path = _polygon_path(polygon)
    vertices = np.asarray(polygon, dtype=float)
    xmin, ymin = np.min(vertices, axis=0)
    xmax, ymax = np.max(vertices, axis=0)
    rng = np.random.default_rng(seed)
    points: list[tuple[float, float]] = []
    attempts = 0
    max_attempts = max(1000, nlocations * 500)
    while len(points) < nlocations and attempts < max_attempts:
        attempts += 1
        candidate = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)])
        if path.contains_point(candidate):
            points.append((float(candidate[0]), float(candidate[1])))

    if len(points) < nlocations:
        raise SeedingGuiError(f'Only generated {len(points)} of {nlocations} random points before attempt limit.')
    return points


def generate_grid_points_in_polygon(
    polygon: list[tuple[float, float]],
    *,
    dx: float,
    dy: float,
) -> list[tuple[float, float]]:
    """Generate a regular dx/dy grid clipped to a polygon."""

    if dx <= 0 or dy <= 0:
        raise SeedingGuiError('Grid dx and dy must be positive.')
    path = _polygon_path(polygon)
    vertices = np.asarray(polygon, dtype=float)
    xmin, ymin = np.min(vertices, axis=0)
    xmax, ymax = np.max(vertices, axis=0)
    xs = np.arange(xmin, xmax + dx * 0.5, dx)
    ys = np.arange(ymin, ymax + dy * 0.5, dy)
    candidates = np.array([(x, y) for x in xs for y in ys], dtype=float)
    if candidates.size == 0:
        return []
    mask = path.contains_points(candidates)
    return [(float(x), float(y)) for x, y in candidates[mask]]


def clip_points_by_elevation(
    points: list[tuple[float, float]],
    *,
    field_x: np.ndarray,
    field_y: np.ndarray,
    field_values: np.ndarray,
    threshold: float,
    delete: str,
) -> list[tuple[float, float]]:
    """Delete selected points above or below an elevation using nearest source cell values."""

    if delete not in {'above', 'below'}:
        raise SeedingGuiError("Clipping delete mode must be 'above' or 'below'.")
    if not points:
        return []

    from scipy.spatial import cKDTree

    source_xy = np.column_stack((np.asarray(field_x, dtype=float).reshape(-1), np.asarray(field_y, dtype=float).reshape(-1)))
    values = np.asarray(field_values, dtype=float).reshape(-1)
    tree = cKDTree(source_xy)
    _, indices = tree.query(np.asarray(points, dtype=float))
    point_values = values[indices]
    if delete == 'above':
        keep = point_values <= threshold
    else:
        keep = point_values >= threshold
    return [(float(x), float(y)) for (x, y), keep_point in zip(points, keep, strict=True) if keep_point]


def save_seeded_config(
    *,
    source_config_path: str | Path,
    output_config_path: str | Path,
    points_output_path: str | Path,
    points: list[tuple[float, float]],
    population_name: str | None = None,
    config_data: dict[str, Any] | None = None,
    population_points: dict[str, list[tuple[float, float]]] | None = None,
) -> tuple[Path, Path]:
    """Write the point file and copied YAML config, then validate the YAML."""

    source_path = Path(source_config_path)
    output_path = Path(output_config_path)
    points_path = Path(points_output_path)

    config = deepcopy(config_data) if config_data is not None else load_config(source_path)
    nonempty_population_points = {
        name: pop_points for name, pop_points in (population_points or {}).items() if pop_points
    }
    if nonempty_population_points:
        updated_config = deepcopy(config)
        single_population = len(nonempty_population_points) == 1
        last_points_path = points_path
        for point_population_name, point_values in nonempty_population_points.items():
            pop_points_path = points_path if single_population else _population_points_path(points_path, point_population_name)
            write_points_file(point_values, pop_points_path)
            last_points_path = pop_points_path
            updated_config = update_config_for_file_points(
                updated_config,
                population_name=point_population_name,
                points_path=_yaml_path_value(pop_points_path, output_path.parent),
            )
        points_path = last_points_path
    else:
        write_points_file(points, points_path)
        yaml_points_path = _yaml_path_value(points_path, output_path.parent)
        updated_config = update_config_for_file_points(
            config,
            population_name=population_name,
            points_path=yaml_points_path,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        yaml.safe_dump(updated_config, f, sort_keys=False, default_flow_style=False)

    try:
        YAMLConfigValidator().validate_yaml(str(output_path))
    except Exception as exc:
        raise SeedingGuiError(f'Generated YAML did not validate: {exc}') from exc

    return output_path, points_path


def launch_seeding_gui(
    *,
    config_path: str | Path,
    output_path: str | Path | None = None,
    points_output_path: str | Path | None = None,
    population_name: str | None = None,
    format_override: str | None = None,
    variable: str | None = None,
) -> None:
    """Open the Matplotlib seed-point selection GUI."""

    app = SeedingGuiApp(
        config_path=Path(config_path),
        output_path=Path(output_path) if output_path is not None else None,
        points_output_path=Path(points_output_path) if points_output_path is not None else None,
        population_name=population_name,
        format_override=format_override,
        variable=variable,
    )
    app.show()


class SeedingGuiApp:
    """Interactive Matplotlib application for selecting SedTRAILS seed points."""

    def __init__(
        self,
        *,
        config_path: Path,
        output_path: Path | None,
        points_output_path: Path | None,
        population_name: str | None,
        format_override: str | None,
        variable: str | None,
    ) -> None:
        self.config_path = config_path
        self.output_path = output_path or default_seeded_config_path(config_path)
        self.points_output_path = points_output_path or self.output_path.with_suffix('.points.txt')
        self.population_name = population_name
        self.format_override = format_override
        self.variable = variable
        self.draft_vertices: list[tuple[float, float]] = []
        self.strategy_mode = 'points'
        self.polygon_closed = False
        self._nearest_remove_pixels = 12.0
        self.colormap_name = 'SEAWAD'
        self._bathymetry_vmin = -20.0
        self._bathymetry_vmax = 10.0
        self._max_display_points = 500_000

        self.config = load_config(self.config_path)
        self.population_names = get_population_names(self.config)
        if self.population_name is None and self.population_names:
            self.population_name = self.population_names[0]
        elif self.population_name not in self.population_names:
            raise SeedingGuiError(f'Population not found: {self.population_name}')
        self.population_points: dict[str, list[tuple[float, float]]] = {name: [] for name in self.population_names}
        self.points = self.population_points[self.population_name]

        self.view_data = load_bathymetry_view_data(
            self.config_path,
            format_override=self.format_override,
            variable=self.variable,
        )

        self._build_ui()

    def _build_ui(self) -> None:
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        from matplotlib.widgets import Button, RadioButtons, TextBox
        from sedtrails.pathway_visualizer.colormaps import bathymetry_colormap

        self.fig, self.ax = plt.subplots(figsize=(12, 7.5))
        self.fig.subplots_adjust(left=0.07, right=0.70, bottom=0.18)

        self.triangulation = mtri.Triangulation(self.view_data.x, self.view_data.y)
        self.bathymetry_cmap, self.bathymetry_norm = bathymetry_colormap(
            self.colormap_name,
            vmin=self._bathymetry_vmin,
            vmax=self._bathymetry_vmax,
        )
        self.bathymetry_field = self.ax.tricontourf(
            self.triangulation,
            self.view_data.values,
            levels=32,
            cmap=self.bathymetry_cmap,
            norm=self.bathymetry_norm,
        )
        self.colorbar = self.fig.colorbar(self.bathymetry_field, ax=self.ax, label=self.view_data.variable)
        self.point_artist = self.ax.scatter([], [], marker='x', c='red', s=42, linewidths=1.8, label='seed points')
        (self.draft_artist,) = self.ax.plot(
            [],
            [],
            '-o',
            color='#f4a261',
            markerfacecolor='white',
            markeredgecolor='#f4a261',
            linewidth=1.5,
            markersize=5,
            label='draft geometry',
        )

        self.ax.set_title(f'{self.view_data.variable} at first timestep')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_aspect('equal', adjustable='datalim')
        self.ax.legend(loc='upper right')

        self.fig.canvas.mpl_connect('button_press_event', self._on_map_click)

        axes = {
            'save': self.fig.add_axes((0.08, 0.05, 0.12, 0.06)),
            'save_as': self.fig.add_axes((0.22, 0.05, 0.12, 0.06)),
            'validate': self.fig.add_axes((0.36, 0.05, 0.12, 0.06)),
            'undo': self.fig.add_axes((0.50, 0.05, 0.12, 0.06)),
            'clear': self.fig.add_axes((0.64, 0.05, 0.12, 0.06)),
        }
        self._buttons = [
            Button(axes['save'], 'Save'),
            Button(axes['save_as'], 'Save as'),
            Button(axes['validate'], 'Validate'),
            Button(axes['undo'], 'Undo'),
            Button(axes['clear'], 'Clear'),
        ]
        self._buttons[0].on_clicked(self._save)
        self._buttons[1].on_clicked(self._save_as)
        self._buttons[2].on_clicked(self._validate)
        self._buttons[3].on_clicked(self._undo)
        self._buttons[4].on_clicked(self._clear)

        mode_ax = self.fig.add_axes((0.73, 0.74, 0.11, 0.18))
        self._mode_radio = RadioButtons(mode_ax, SEEDING_MODES, active=0)
        self._mode_radio.on_clicked(self._set_strategy_mode)

        self._population_radio_ax = self.fig.add_axes((0.86, 0.74, 0.12, 0.18))
        self._population_radio = None
        self._rebuild_population_selector()

        cmap_ax = self.fig.add_axes((0.86, 0.62, 0.12, 0.10))
        self._colormap_radio = RadioButtons(cmap_ax, ('SEAWAD', 'Vintage'), active=0)
        self._colormap_radio.on_clicked(self._set_colormap)

        self._population_name_box = TextBox(
            self.fig.add_axes((0.75, 0.57, 0.20, 0.04)),
            'pop ',
            initial=self.population_name,
        )
        self._add_population_button = Button(self.fig.add_axes((0.75, 0.51, 0.06, 0.04)), 'Add')
        self._rename_population_button = Button(self.fig.add_axes((0.82, 0.51, 0.06, 0.04)), 'Rename')
        self._remove_population_button = Button(self.fig.add_axes((0.89, 0.51, 0.06, 0.04)), 'Remove')
        self._add_population_button.on_clicked(self._add_population)
        self._rename_population_button.on_clicked(self._rename_population)
        self._remove_population_button.on_clicked(self._remove_population)

        self._transect_k_box = TextBox(self.fig.add_axes((0.75, 0.44, 0.08, 0.04)), 'k ', initial='20')
        self._random_n_box = TextBox(self.fig.add_axes((0.75, 0.38, 0.08, 0.04)), 'n ', initial='20')
        self._seed_box = TextBox(self.fig.add_axes((0.90, 0.38, 0.07, 0.04)), 'seed ', initial='42')
        self._dx_box = TextBox(self.fig.add_axes((0.75, 0.32, 0.08, 0.04)), 'dx ', initial='100')
        self._dy_box = TextBox(self.fig.add_axes((0.90, 0.32, 0.07, 0.04)), 'dy ', initial='100')

        self._generate_button = Button(self.fig.add_axes((0.75, 0.26, 0.20, 0.05)), 'Generate')
        self._generate_button.on_clicked(self._generate_from_strategy)

        self._cmin_box = TextBox(self.fig.add_axes((0.75, 0.19, 0.08, 0.04)), 'cmin ', initial=str(self._bathymetry_vmin))
        self._cmax_box = TextBox(self.fig.add_axes((0.90, 0.19, 0.07, 0.04)), 'cmax ', initial=str(self._bathymetry_vmax))
        self._color_button = Button(self.fig.add_axes((0.75, 0.13, 0.20, 0.05)), 'Apply color limits')
        self._color_button.on_clicked(self._apply_color_limits)

        self._clip_mode = 'above'
        clip_mode_ax = self.fig.add_axes((0.74, 0.01, 0.12, 0.09))
        self._clip_radio = RadioButtons(clip_mode_ax, ('above', 'below'), active=0)
        self._clip_radio.on_clicked(self._set_clip_mode)
        self._clip_elevation_box = TextBox(self.fig.add_axes((0.90, 0.05, 0.07, 0.04)), 'z ', initial='0')
        self._clip_button = Button(self.fig.add_axes((0.75, 0.00, 0.20, 0.04)), 'Clip points')
        self._clip_button.on_clicked(self._clip_selected_points)

        self.status_text = self.fig.text(0.08, 0.13, self._status_message(), fontsize=9)

    def show(self) -> None:
        import matplotlib.pyplot as plt

        plt.show()

    def _on_map_click(self, event: Any) -> None:
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return

        if self.strategy_mode == 'points':
            self._on_points_click(event)
            return

        if self.strategy_mode == 'transect':
            self._on_transect_click(event)
            return

        if self.strategy_mode in {'random', 'grid'}:
            self._on_polygon_click(event)
            return

    def _on_points_click(self, event: Any) -> None:
        if event.button == 1:
            self.points.append((float(event.xdata), float(event.ydata)))
            self._refresh_points()
            return

        if event.button == 3:
            self._remove_nearest(event)

    def _on_transect_click(self, event: Any) -> None:
        if event.button == 1:
            self.draft_vertices.append((float(event.xdata), float(event.ydata)))
            self._refresh_draft()
            return

        if event.button == 3 and self.draft_vertices:
            self.draft_vertices.pop()
            self._refresh_draft()

    def _on_polygon_click(self, event: Any) -> None:
        if event.button == 1:
            if self.polygon_closed:
                self.draft_vertices.clear()
                self.polygon_closed = False
            self.draft_vertices.append((float(event.xdata), float(event.ydata)))
            self._refresh_draft()
            return

        if event.button == 3 and len(self.draft_vertices) >= 3:
            self.polygon_closed = True
            self._refresh_draft()

    def _remove_nearest(self, event: Any) -> None:
        if not self.points:
            return

        point_pixels = self.ax.transData.transform(np.asarray(self.points))
        click_pixel = np.asarray([event.x, event.y])
        distances = np.linalg.norm(point_pixels - click_pixel, axis=1)
        nearest = int(np.argmin(distances))
        if distances[nearest] <= self._nearest_remove_pixels:
            self.points.pop(nearest)
            self._refresh_points()

    def _refresh_points(self) -> None:
        if self._should_display_points() and self.points:
            self.point_artist.set_offsets(np.asarray(self.points))
        else:
            self.point_artist.set_offsets(np.empty((0, 2)))
        self.status_text.set_text(self._status_message())
        self.fig.canvas.draw_idle()

    def _refresh_draft(self) -> None:
        vertices = list(self.draft_vertices)
        if self.polygon_closed and len(vertices) >= 3:
            vertices.append(vertices[0])
        if vertices:
            xs, ys = zip(*vertices, strict=True)
            self.draft_artist.set_data(xs, ys)
        else:
            self.draft_artist.set_data([], [])
        self.status_text.set_text(self._status_message())
        self.fig.canvas.draw_idle()

    def _set_points(self, points: list[tuple[float, float]]) -> None:
        self.points = points
        self.population_points[self.population_name] = self.points
        self._refresh_points()

    def _rebuild_population_selector(self) -> None:
        from matplotlib.widgets import RadioButtons

        self._population_radio_ax.clear()
        self.population_names = get_population_names(self.config)
        for population_name in self.population_names:
            self.population_points.setdefault(population_name, [])
        if self.population_name not in self.population_names:
            self.population_name = self.population_names[0]
        active_idx = self.population_names.index(self.population_name)
        self._population_radio = RadioButtons(self._population_radio_ax, self.population_names, active=active_idx)
        self._population_radio.on_clicked(self._set_population)
        self.fig.canvas.draw_idle()

    def _set_strategy_mode(self, label: str) -> None:
        self.strategy_mode = label
        self.draft_vertices.clear()
        self.polygon_closed = False
        self._refresh_draft()

    def _set_population(self, label: str) -> None:
        if label not in self.population_points:
            self.population_points[label] = []
        self.population_name = label
        self.points = self.population_points[label]
        self._population_name_box.set_val(label)
        self._refresh_points()
        self.status_text.set_text(self._status_message())
        self.fig.canvas.draw_idle()

    def _add_population(self, _event: Any = None) -> None:
        source_name = self.population_name
        try:
            self.config, new_name = add_population_from_existing(
                self.config,
                source_population_name=source_name,
                new_population_name=self._population_name_box.text,
            )
        except Exception as exc:
            self._show_error(str(exc))
            return

        self.population_points.setdefault(new_name, [])
        self.population_name = new_name
        self.points = self.population_points[new_name]
        self._population_name_box.set_val(new_name)
        self._rebuild_population_selector()
        self._refresh_points()
        self._show_info(
            f"Added population '{new_name}' by copying '{source_name}'.\n\n"
            'Check the generated YAML before running to confirm the copied particle settings are correct.'
        )

    def _rename_population(self, _event: Any = None) -> None:
        old_name = self.population_name
        new_name = self._population_name_box.text.strip()
        try:
            self.config = rename_population(self.config, old_name=old_name, new_name=new_name)
        except Exception as exc:
            self._show_error(str(exc))
            return

        self.population_points[new_name] = self.population_points.pop(old_name, [])
        self.population_name = new_name
        self.points = self.population_points[new_name]
        self._rebuild_population_selector()
        self._refresh_points()

    def _remove_population(self, _event: Any = None) -> None:
        removed_name = self.population_name
        try:
            self.config = remove_population(self.config, population_name=removed_name)
        except Exception as exc:
            self._show_error(str(exc))
            return

        self.population_points.pop(removed_name, None)
        self.population_names = get_population_names(self.config)
        self.population_name = self.population_names[0]
        self.points = self.population_points.setdefault(self.population_name, [])
        self._population_name_box.set_val(self.population_name)
        self._rebuild_population_selector()
        self._refresh_points()

    def _set_clip_mode(self, label: str) -> None:
        self._clip_mode = label

    def _set_colormap(self, label: str) -> None:
        self.colormap_name = label
        try:
            self._redraw_bathymetry()
        except Exception as exc:
            self._show_error(str(exc))

    def _generate_from_strategy(self, _event: Any = None) -> None:
        try:
            if self.strategy_mode == 'points':
                raise SeedingGuiError('Points mode does not need generation; left click to add points.')
            if self.strategy_mode == 'transect':
                generated = generate_transect_points(
                    self.draft_vertices,
                    self._parse_int_box(self._transect_k_box, 'transect k'),
                )
            elif self.strategy_mode == 'random':
                if not self.polygon_closed:
                    raise SeedingGuiError('Right click to close the random polygon before generating points.')
                generated = generate_random_points_in_polygon(
                    self.draft_vertices,
                    nlocations=self._parse_int_box(self._random_n_box, 'random nlocations'),
                    seed=self._parse_int_box(self._seed_box, 'random seed'),
                )
            elif self.strategy_mode == 'grid':
                if not self.polygon_closed:
                    raise SeedingGuiError('Right click to close the grid polygon before generating points.')
                generated = generate_grid_points_in_polygon(
                    self.draft_vertices,
                    dx=self._parse_float_box(self._dx_box, 'grid dx'),
                    dy=self._parse_float_box(self._dy_box, 'grid dy'),
                )
            else:
                raise SeedingGuiError(f'Unknown seeding mode: {self.strategy_mode}')
        except Exception as exc:
            self._show_error(str(exc))
            return

        self._set_points(generated)

    def _clip_selected_points(self, _event: Any = None) -> None:
        try:
            clipped_points = clip_points_by_elevation(
                self.points,
                field_x=self.view_data.x,
                field_y=self.view_data.y,
                field_values=self.view_data.values,
                threshold=self._parse_float_box(self._clip_elevation_box, 'clip elevation'),
                delete=self._clip_mode,
            )
        except Exception as exc:
            self._show_error(str(exc))
            return
        self._set_points(clipped_points)

    def _apply_color_limits(self, _event: Any = None) -> None:
        try:
            vmin = self._parse_float_box(self._cmin_box, 'color minimum')
            vmax = self._parse_float_box(self._cmax_box, 'color maximum')
            if vmin >= vmax:
                raise SeedingGuiError('Color minimum must be less than color maximum.')
        except Exception as exc:
            self._show_error(str(exc))
            return

        self._bathymetry_vmin = vmin
        self._bathymetry_vmax = vmax
        try:
            self._redraw_bathymetry()
        except Exception as exc:
            self._show_error(str(exc))

    def _redraw_bathymetry(self) -> None:
        from sedtrails.pathway_visualizer.colormaps import bathymetry_colormap

        self.bathymetry_cmap, self.bathymetry_norm = bathymetry_colormap(
            self.colormap_name,
            vmin=self._bathymetry_vmin,
            vmax=self._bathymetry_vmax,
        )
        if hasattr(self.bathymetry_field, 'remove'):
            self.bathymetry_field.remove()
        else:
            for collection in getattr(self.bathymetry_field, 'collections', []):
                collection.remove()
        levels = np.linspace(self._bathymetry_vmin, self._bathymetry_vmax, 32)
        self.bathymetry_field = self.ax.tricontourf(
            self.triangulation,
            self.view_data.values,
            levels=levels,
            cmap=self.bathymetry_cmap,
            norm=self.bathymetry_norm,
        )
        self.colorbar.update_normal(self.bathymetry_field)
        self.fig.canvas.draw_idle()

    def _parse_int_box(self, box: Any, label: str) -> int:
        value = self._parse_float_box(box, label)
        if int(value) != value:
            raise SeedingGuiError(f'{label} must be an integer.')
        return int(value)

    def _parse_float_box(self, box: Any, label: str) -> float:
        try:
            return float(str(box.text).strip())
        except ValueError as exc:
            raise SeedingGuiError(f'{label} must be numeric.') from exc

    def _save(self, _event: Any = None) -> None:
        try:
            config_path, points_path = save_seeded_config(
                source_config_path=self.config_path,
                output_config_path=self.output_path,
                points_output_path=self.points_output_path,
                points=self.points,
                population_name=self.population_name,
                config_data=self.config,
                population_points=self.population_points,
            )
        except Exception as exc:
            self._show_error(str(exc))
            return

        self._show_info(f'Saved config:\n{config_path}\n\nSaved points:\n{points_path}')
        self.status_text.set_text(self._status_message())
        self.fig.canvas.draw_idle()

    def _save_as(self, _event: Any = None) -> None:
        try:
            from tkinter import filedialog

            selected = filedialog.asksaveasfilename(
                title='Save SedTRAILS config as',
                defaultextension='.yaml',
                filetypes=[('YAML files', '*.yaml *.yml'), ('All files', '*.*')],
                initialdir=str(self.config_path.parent),
                initialfile=self.output_path.name,
            )
        except Exception as exc:
            self._show_error(f'Could not open save dialog: {exc}')
            return

        if selected:
            self.output_path = Path(selected)
            self.points_output_path = self.output_path.with_suffix('.points.txt')
            self._save()

    def _validate(self, _event: Any = None) -> None:
        try:
            YAMLConfigValidator().validate_yaml(str(self.output_path))
        except Exception as exc:
            self._show_error(f'Validation failed:\n{exc}')
            return
        self._show_info(f'Configuration validates:\n{self.output_path}')

    def _undo(self, _event: Any = None) -> None:
        if self.points:
            self.points.pop()
            self._refresh_points()

    def _clear(self, _event: Any = None) -> None:
        self.points.clear()
        self.draft_vertices.clear()
        self.polygon_closed = False
        self._refresh_points()
        self._refresh_draft()

    def _status_message(self) -> str:
        hints = {
            'points': 'Left click adds a seed point; right click near a point removes it.',
            'transect': 'Left click transect endpoints in pairs; Generate creates points along each segment.',
            'random': 'Left click polygon vertices; right click closes; Generate samples random points inside.',
            'grid': 'Left click polygon vertices; right click closes; Generate creates a dx/dy grid inside.',
        }
        warning = ''
        if not self._should_display_points() and self.points:
            warning = f' Warning: >{self._max_display_points:,} points, particles not displayed.'
        return (
            f'Mode: {self.strategy_mode}. {hints.get(self.strategy_mode, "")} '
            f'{len(self.points)} seed point(s), {len(self.draft_vertices)} draft vertex/vertices. '
            f'Population: {self.population_name}.{warning}'
        )

    def _should_display_points(self) -> bool:
        return len(self.points) <= self._max_display_points

    def _show_error(self, message: str) -> None:
        self._show_dialog('SedTRAILS seeding setup error', message, error=True)

    def _show_info(self, message: str) -> None:
        self._show_dialog('SedTRAILS seeding setup', message, error=False)

    def _show_dialog(self, title: str, message: str, *, error: bool) -> None:
        try:
            from tkinter import messagebox

            if error:
                messagebox.showerror(title, message)
            else:
                messagebox.showinfo(title, message)
        except Exception:
            print(f'{title}: {message}')


def _first_timestep_values(variable: Any) -> np.ndarray:
    data = variable
    if 'layer' in getattr(data, 'dims', ()):
        data = data.isel(layer=0)
    if 'time' in getattr(data, 'dims', ()):
        data = data.isel(time=0)
    return np.asarray(data.values)


def _polygon_path(polygon: list[tuple[float, float]]) -> Any:
    if len(polygon) < 3:
        raise SeedingGuiError('A polygon needs at least three clicked vertices.')
    from matplotlib.path import Path as MplPath

    return MplPath(np.asarray(polygon, dtype=float))


def _resolve_bathymetry_variable(dataset: Any, requested: str | None) -> str:
    candidates = [requested] if requested else ['bedlevel', 'bed_level']
    for candidate in candidates:
        if candidate and candidate in dataset:
            return candidate
    available = ', '.join(str(name) for name in dataset.data_vars)
    if requested:
        raise SeedingGuiError(f"Requested bathymetry variable '{requested}' was not found. Available variables: {available}")
    raise SeedingGuiError(f"No bathymetry variable found. Tried 'bedlevel' and 'bed_level'. Available variables: {available}")


def _get_populations(config: dict[str, Any]) -> list[dict[str, Any]]:
    populations = config.get('particles', {}).get('populations')
    if not isinstance(populations, list) or not populations:
        raise SeedingGuiError('Configuration has no particles.populations entries.')
    return populations


def _select_population(populations: list[dict[str, Any]], population_name: str | None) -> dict[str, Any]:
    if population_name is None:
        return populations[0]

    for population in populations:
        if population.get('name') == population_name:
            return population

    names = ', '.join(str(pop.get('name')) for pop in populations)
    raise SeedingGuiError(f"Population '{population_name}' was not found. Available populations: {names}")


def _unique_population_name(existing_names: list[str], requested_name: str) -> str:
    base_name = requested_name.strip() or 'population'
    if base_name not in existing_names:
        return base_name
    suffix = 2
    while f'{base_name}_{suffix}' in existing_names:
        suffix += 1
    return f'{base_name}_{suffix}'


def _population_points_path(base_path: Path, population_name: str) -> Path:
    safe_name = ''.join(char if char.isalnum() or char in {'-', '_'} else '_' for char in population_name).strip('_')
    if not safe_name:
        safe_name = 'population'
    return base_path.with_name(f'{base_path.stem}.{safe_name}{base_path.suffix}')


def _resolve_relative_path(path: str | Path, base_dir: Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    config_relative = (base_dir / candidate).resolve()
    if config_relative.exists():
        return config_relative
    return candidate.resolve()


def _yaml_path_value(points_path: Path, output_config_dir: Path) -> str:
    try:
        path_value = Path(points_path).resolve().relative_to(output_config_dir.resolve())
        normalized = path_value.as_posix()
    except ValueError:
        normalized = Path(points_path).as_posix()

    if not normalized.startswith(('.', '/')) and ':' not in normalized:
        normalized = f'./{normalized}'
    return normalized
