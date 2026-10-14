"""Desktop analysis of seed patches and trajectories coloured by their origins."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from sedtrails.application_interfaces.seeding_gui import (
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
    max_particles: int = 10_000
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None

    def validate(self) -> None:
        self.background = parse_rgb(self.background)
        if type(self.max_particles) is not int or self.max_particles < 1:
            raise ValueError('Maximum plotted trajectories must be a positive integer.')
        for value in (self.show_seeds, self.show_tracks, self.show_patches):
            if type(value) is not bool:
                raise ValueError('Layer visibility values must be true or false.')
        names = [patch.name for patch in self.patches]
        if len(set(names)) != len(names):
            raise ValueError('Patch names must be unique.')
        for limits in (self.xlim, self.ylim):
            if limits is not None and (len(limits) != 2 or not np.isfinite(limits).all() or limits[0] == limits[1]):
                raise ValueError('Map limits must contain two different finite coordinates.')


def save_session(path: str | Path, session: PatchSession) -> None:
    session.validate()
    payload = asdict(session)
    for key in ('config_path', 'results_path'):
        value = payload[key]
        payload[key] = str(value.resolve()) if value is not None else None
    payload['version'] = 1
    Path(path).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_session(path: str | Path) -> PatchSession:
    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict) or payload.pop('version', None) != 1:
        raise ValueError('Unsupported patch session; expected a JSON object with version 1.')
    try:
        for key in ('config_path', 'results_path'):
            value = payload.get(key)
            payload[key] = (path.parent / value).resolve() if value is not None else None
        payload['patches'] = [SquarePatch(**item) for item in payload.get('patches', [])]
        for key in ('xlim', 'ylim'):
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
        self.bathymetry = load_bathymetry_view_data(self.config_path)
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
        if not len(self.seeds) or not np.isfinite(self.seeds).all():
            raise ValueError('The YAML did not produce finite seeding points.')
        self.tracks: np.ndarray | None = None
        self.track_seed_indices = np.empty(0, dtype=int)
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
            selected, total, _ = _sample_dataset(dataset, max_particles=self.session.max_particles)
            x, y, _ = _trajectory_arrays(selected)
            x, y = self.coordinates.source_to_display(x, y)
            origins = np.column_stack((x[:, 0], y[:, 0]))
            if not len(origins) or not np.isfinite(origins).all():
                raise ValueError('Output must contain finite initial positions for all plotted particles.')
            distances, indices = cKDTree(self.seeds).query(origins)
            # NetCDF coordinates may be float32: match back to exact YAML seeds before classifying.
            tolerance = np.maximum(1e-5, np.max(np.abs(origins), axis=1) * np.finfo(np.float32).eps * 4)
            if np.any(distances > tolerance):
                raise ValueError(
                    'Output initial positions do not match the YAML seeds. Choose the associated full '
                    'trajectory file, including its initial timestep (not a restart or a different run).'
                )
            tracks = np.stack((x, y), axis=-1)
        self.tracks = tracks
        self.track_seed_indices = indices
        self.total_tracks = total
        self.session.results_path = Path(path).resolve()


class PatchPlot:
    """Render only the map, so PNG export excludes desktop controls."""

    def __init__(self, figure: Any, model: PatchModel):
        import matplotlib.tri as mtri
        from sedtrails.pathway_visualizer.colormaps import bathymetry_colormap

        self.figure = figure
        self.model = model
        self.ax = figure.add_subplot(111)
        vmin, vmax = float(model.bathymetry.values.min()), float(model.bathymetry.values.max())
        if vmin == vmax:
            vmin, vmax = vmin - 0.5, vmax + 0.5
        cmap, norm = bathymetry_colormap('SEAWAD', vmin=vmin, vmax=vmax)
        triangulation = _create_masked_triangulation(model.map_x, model.map_y, mtri_module=mtri)
        bathymetry = self.ax.tricontourf(triangulation, model.bathymetry.values, levels=32, cmap=cmap, norm=norm)
        figure.colorbar(bathymetry, ax=self.ax, label=f'{model.bathymetry.variable} [m]')
        self.ax.set(xlabel='x [m]', ylabel='y [m]', title=model.config_path.name)
        self.ax.set_aspect('equal', adjustable='box')
        self.home_xlim, self.home_ylim = self.ax.get_xlim(), self.ax.get_ylim()
        self.artists: list[Any] = []
        self.draw()
        if model.session.xlim is not None:
            self.ax.set_xlim(model.session.xlim)
        if model.session.ylim is not None:
            self.ax.set_ylim(model.session.ylim)

    def draw(self) -> None:
        from matplotlib.collections import LineCollection
        from matplotlib.patches import Rectangle
        from matplotlib.lines import Line2D

        model, session = self.model, self.model.session
        session.validate()
        labels = classify_points(model.seeds, session.patches)
        colours = point_colours(labels, session.patches, session.background)
        for artist in self.artists:
            artist.remove()
        self.artists.clear()
        self.track_artist = None
        self.seed_artist = None
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
        if session.show_tracks and model.tracks is not None:
            self.track_order = np.argsort(labels[model.track_seed_indices], kind='stable')
            self.track_artist = LineCollection(
                np.ma.masked_invalid(model.tracks[self.track_order]),
                colors=colours[model.track_seed_indices[self.track_order]],
                linewidths=0.7,
                zorder=3,
            )
            self.ax.add_collection(self.track_artist, autolim=False)
            self.artists.append(self.track_artist)
        if session.show_seeds:
            self.seed_artist = self.ax.scatter(*model.seeds.T, c=colours, s=9, edgecolors='none', zorder=4)
            self.artists.append(self.seed_artist)
        handles = [
            Line2D(
                [],
                [],
                marker='o',
                linestyle='',
                color=np.asarray(session.background) / 255,
                label=f'Background ({np.count_nonzero(labels == -1)})',
            )
        ]
        handles.extend(
            Line2D(
                [],
                [],
                marker='s',
                linestyle='',
                color=np.asarray(p.rgb) / 255,
                label=f'{p.name} ({np.count_nonzero(labels == i)})',
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

        files = ttk.LabelFrame(panel, text='Files', padding=5)
        files.pack(fill='x')
        button(files, 'Open SedTRAILS YAML...', self.open_yaml)
        button(files, 'Open patch JSON...', self.open_json)
        self.file_label = ttk.Label(files, text='No YAML loaded', wraplength=340)
        self.file_label.pack(fill='x')
        button(files, 'Load associated trajectories', self.load_associated)
        button(files, 'Choose trajectory NetCDF...', self.choose_tracks)

        patches = ttk.LabelFrame(panel, text='Patches (last row wins overlaps)', padding=5)
        patches.pack(fill='x', pady=7)
        self.table = ttk.Treeview(patches, columns=('x', 'y', 'n', 'rgb'), height=6)
        self.table.heading('#0', text='Name')
        self.table.column('#0', width=90)
        for column in ('x', 'y', 'n', 'rgb'):
            self.table.heading(column, text=column)
            self.table.column(column, width=60 if column != 'rgb' else 95)
        self.table.pack(fill='x')
        self.table.bind('<<TreeviewSelect>>', self.select_patch)
        editor = ttk.Frame(patches)
        editor.pack(fill='x')
        defaults = (
            ('name', 'Name / series prefix', 'Patch'),
            ('x', 'Lower-left x [m]', '0'),
            ('y', 'Lower-left y [m]', '0'),
            ('size', 'Side length n [m]', '1'),
            ('rgb', 'RGB (0..255)', '255, 0, 0'),
            ('rows', 'Series rows (+y)', '1'),
            ('columns', 'Series columns (+x)', '1'),
        )
        for row, (key, label, default) in enumerate(defaults):
            ttk.Label(editor, text=label).grid(row=row, column=0, sticky='w', pady=2)
            value = tk.StringVar(value=default)
            ttk.Entry(editor, textvariable=value, width=22).grid(row=row, column=1, sticky='ew')
            self.fields[key] = value
        button(patches, 'Add square / touching row-column series', self.add_patches)
        button(patches, 'Update selected square', self.update_patch)
        button(patches, 'Delete selected square(s)', self.delete_patches)

        layers = ttk.LabelFrame(panel, text='Layers and rendering', padding=5)
        layers.pack(fill='x')
        self.background = tk.StringVar(value=', '.join(map(str, session.background)))
        ttk.Label(layers, text='Background point RGB (0..255)').pack(anchor='w')
        ttk.Entry(layers, textvariable=self.background).pack(fill='x')
        self.maximum = tk.StringVar(value=str(session.max_particles))
        ttk.Label(layers, text='Maximum plotted trajectories').pack(anchor='w')
        ttk.Entry(layers, textvariable=self.maximum).pack(fill='x')
        for key, label in (
            ('show_seeds', 'Seeding points'),
            ('show_tracks', 'Trajectories'),
            ('show_patches', 'Patch squares'),
        ):
            value = tk.BooleanVar(value=getattr(session, key))
            self.visibility[key] = value
            ttk.Checkbutton(layers, text=label, variable=value, command=lambda: self._run(self.apply_layers)).pack(
                anchor='w'
            )
        button(layers, 'Apply colours / rendering', self.apply_layers)
        button(panel, 'Save patch JSON...', self.save_json)
        button(panel, 'Export map PNG...', self.export_png)
        self.status = ttk.Label(panel, text='Open a YAML file or a saved JSON session.', wraplength=350)
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
        self.canvas = FigureCanvasTkAgg(figure, master=self.map_frame)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.map_frame, pack_toolbar=False)
        self.toolbar.pack(side='bottom', fill='x')
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self.canvas.draw()
        self.file_label.configure(text=str(self.session.config_path))
        self.background.set(', '.join(map(str, self.session.background)))
        self.maximum.set(str(self.session.max_particles))
        for key, variable in self.visibility.items():
            variable.set(getattr(self.session, key))
        self._refresh_table()
        self._report()

    def _require_model(self) -> PatchModel:
        if self.model is None:
            raise ValueError('Open a SedTRAILS YAML file first.')
        return self.model

    def _report(self) -> None:
        model = self._require_model()
        shown = 0 if model.tracks is None else len(model.tracks)
        crs = model.coordinates.metric_crs or 'native projected coordinates'
        self.status.configure(
            text=f'{len(model.seeds):,} seeds; {shown:,}/{model.total_tracks:,} trajectories loaded.\n'
            f'{crs}. Trajectories retain their seed colour. Large outputs are sampled '
            'deterministically (at most 2,000,000 plotted track points).'
        )

    def _refresh_table(self) -> None:
        self.table.delete(*self.table.get_children())
        for index, patch in enumerate(self.session.patches):
            self.table.insert(
                '',
                'end',
                iid=str(index),
                text=patch.name,
                values=(patch.x, patch.y, patch.size, ','.join(map(str, patch.rgb))),
            )

    def open_yaml(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(parent=self.root, filetypes=[('SedTRAILS YAML', '*.yaml *.yml')])
        if path:
            self._install_model(PatchModel(PatchSession(config_path=Path(path))))

    def open_json(self) -> None:
        from tkinter import filedialog

        path = filedialog.askopenfilename(parent=self.root, filetypes=[('Patch session', '*.json')])
        if path:
            self._install_model(PatchModel(load_session(path)))

    def _read_layers(self) -> PatchSession:
        candidate = deepcopy(self.session)
        candidate.background = parse_rgb(self.background.get())
        candidate.max_particles = int(self.maximum.get())
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
        self.plot.draw()
        self._report()

    def _read_patch(self) -> SquarePatch:
        return SquarePatch(
            self.fields['name'].get().strip(),
            float(self.fields['x'].get()),
            float(self.fields['y'].get()),
            float(self.fields['size'].get()),
            parse_rgb(self.fields['rgb'].get()),
        )

    def add_patches(self) -> None:
        self._require_model()
        patch = self._read_patch()
        rows, columns = int(self.fields['rows'].get()), int(self.fields['columns'].get())
        if rows < 1 or columns < 1 or rows * columns > 10_000:
            raise ValueError('Series rows and columns must be positive, with at most 10,000 patches.')
        additions = [
            SquarePatch(
                patch.name if rows * columns == 1 else f'{patch.name}[{row + 1},{column + 1}]',
                patch.x + column * patch.size,
                patch.y + row * patch.size,
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
        self.fields['rgb'].set(', '.join(map(str, patch.rgb)))
        self.fields['rows'].set('1')
        self.fields['columns'].set('1')

    def update_patch(self) -> None:
        selection = self.table.selection()
        if len(selection) != 1:
            raise ValueError('Select one patch to update.')
        candidate = deepcopy(self.session)
        candidate.patches[int(selection[0])] = self._read_patch()
        candidate.validate()
        self.session.patches = candidate.patches
        self._refresh_table()
        self.plot.draw()

    def delete_patches(self) -> None:
        selected = {int(index) for index in self.table.selection()}
        if not selected:
            raise ValueError('Select the patch(es) to delete.')
        self.session.patches = [p for i, p in enumerate(self.session.patches) if i not in selected]
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
        self.plot.draw()
        self._report()

    def save_json(self) -> None:
        from tkinter import filedialog

        self.apply_layers()
        self.plot.remember_view()
        path = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension='.json',
            filetypes=[('Patch session', '*.json')],
            initialfile=self.session.config_path.stem + '-patches.json',
        )
        if path:
            save_session(path, self.session)
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


def launch_patch_gui(
    config_path: str | Path | None = None,
    session_path: str | Path | None = None,
    results_path: str | Path | None = None,
) -> None:
    import tkinter as tk

    if config_path is not None and session_path is not None:
        raise SeedingGuiError('Use either --config or --session, not both.')
    session = (
        load_session(session_path)
        if session_path
        else PatchSession(config_path=Path(config_path).resolve() if config_path else None)
    )
    if results_path is not None:
        session.results_path = Path(results_path).resolve()
    root = tk.Tk()
    root.protocol('WM_DELETE_WINDOW', root.quit)
    try:
        app = PatchGuiApp(root, session)
        root.mainloop()
        del app
    finally:
        root.destroy()
