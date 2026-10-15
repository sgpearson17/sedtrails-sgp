import json

import matplotlib

matplotlib.use('Agg')

from matplotlib.figure import Figure
import numpy as np
import pytest
import xarray as xr
import yaml

from sedtrails.application_interfaces.patch_gui import (
    PatchGuiApp,
    PatchModel,
    PatchPlot,
    PatchSession,
    SquarePatch,
    classify_points,
    launch_patch_gui,
    load_session,
    parse_rgb,
    point_colours,
    save_session,
)
from sedtrails.application_interfaces.seeding_gui import _resolve_relative_path


@pytest.fixture
def case(tmp_path):
    x, y = np.meshgrid(np.arange(4.0), np.arange(3.0))
    xr.Dataset(
        {
            'globalx': (('ny', 'nx'), x),
            'globaly': (('ny', 'nx'), y),
            'zb': (('time', 'ny', 'nx'), (-x - y)[None]),
        }
    ).to_netcdf(tmp_path / 'bathy.nc')
    (tmp_path / 'seeds.txt').write_text('0 0\n1 0\n2 0\n3 0\n', encoding='utf-8')
    config = {
        'general': {'input_model': {'format': 'xbeach', 'reference_date': '2000-01-01'}},
        'inputs': {'data': 'bathy.nc'},
        'outputs': {'directory': '.'},
        'time': {'start': '2000-01-01 00:00:00'},
        'particles': {
            'populations': [
                {
                    'name': 'sand',
                    'particle_type': 'passive',
                    'seeding': {
                        'quantity': 1,
                        'strategy': {
                            'file_points': {
                                'path': 'seeds.txt',
                                'has_header': False,
                                'deduplicate': False,
                            }
                        },
                    },
                }
            ]
        },
    }
    path = tmp_path / 'case.yaml'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    output = tmp_path / 'sedtrails_results.nc'
    xr.Dataset(
        {
            'x': (('time', 'n_particles'), np.array([[0, 1, 2, 3], [2, 2, 0, 0], [3, 3, 1, 1]], dtype=float)),
            'y': (('time', 'n_particles'), np.zeros((3, 4))),
        },
        coords={'time': [0.0, 60, 120]},
        attrs={'reference_date': '2000-01-01'},
    ).to_netcdf(output)
    return path, output


def test_classification_includes_edges_and_last_patch_wins():
    patches = [
        SquarePatch('red', 0, 0, 2, (255, 0, 0)),
        SquarePatch('blue', 1, 1, 1, (0, 0, 255)),
    ]
    points = np.array([[0, 0], [0, 2], [1, 1], [2, 2], [2.001, 2], [-1, 0]])
    labels = classify_points(points, patches)
    np.testing.assert_array_equal(labels, [0, 0, 1, 1, -1, -1])
    colours = point_colours(labels, patches, (0, 255, 0))
    np.testing.assert_allclose(colours[[0, 2, 4]], [[1, 0, 0], [0, 0, 1], [0, 1, 0]])


@pytest.mark.parametrize('rgb', ['255,0,80', '255 0 80', [255, 0, 80], (255, 0, 80)])
def test_rgb_input(rgb):
    assert parse_rgb(rgb) == (255, 0, 80)


@pytest.mark.parametrize('rgb', ['256,0,0', '-1,0,0', '1.0,0,0', 'nan,0,0', '1,2', [True, 0, 0]])
def test_invalid_rgb_is_rejected(rgb):
    with pytest.raises(ValueError, match='RGB'):
        parse_rgb(rgb)


@pytest.mark.parametrize('x,y,size', [(0, 0, 0), (0, 0, -1), (np.nan, 0, 1), (0, np.inf, 1)])
def test_invalid_patch_geometry(x, y, size):
    with pytest.raises(ValueError):
        SquarePatch('bad', x, y, size, (0, 0, 0))


def test_loading_uses_actual_seed_strategy_and_quantity(case):
    path, _ = case
    config = yaml.safe_load(path.read_text())
    population = config['particles']['populations'][0]
    population['seeding']['quantity'] = 2
    population['seeding']['strategy']['file_points']['stride'] = 2
    config['particles']['populations'].append(
        {
            'name': 'extra',
            'particle_type': 'passive',
            'seeding': {'quantity': 1, 'strategy': {'point': {'locations': ['0.5,0.5']}}},
        }
    )
    path.write_text(yaml.safe_dump(config))
    model = PatchModel(PatchSession(config_path=path))
    np.testing.assert_allclose(model.seeds, [[0, 0], [0, 0], [2, 0], [2, 0], [0.5, 0.5]])
    assert model.associated_results() == path.parent / 'sedtrails_results.nc'


def test_tracks_keep_origin_colour_after_crossing_patches(case):
    path, output = case
    session = PatchSession(
        config_path=path,
        results_path=output,
        background=(0, 255, 0),
        patches=[SquarePatch('red', 0, 0, 0.5, (255, 0, 0)), SquarePatch('blue', 2, 0, 0.5, (0, 0, 255))],
    )
    model = PatchModel(session)
    plot = PatchPlot(Figure(), model)
    colours = plot.track_artist.get_colors()[np.argsort(plot.track_order), :3]
    np.testing.assert_allclose(colours, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
    np.testing.assert_allclose(plot.seed_artist.get_facecolors()[:, :3], colours)
    assert model.tracks[0, -1, 0] == 3  # Red origin is now outside its square.
    session.patches[0] = SquarePatch('red', 0, 0, 0.5, (255, 255, 0))
    plot.draw()
    np.testing.assert_allclose(plot.track_artist.get_colors()[np.flatnonzero(plot.track_order == 0)[0], :3], [1, 1, 0])
    np.testing.assert_array_equal(plot.track_order[:2], [1, 3])
    assert plot.ax.collections[0].norm.vmin == model.bathymetry.values.min()
    assert plot.ax.collections[0].norm.vmax == model.bathymetry.values.max()


def test_sampling_preserves_origin_identity_and_is_deterministic(case):
    path, output = case
    model = PatchModel(PatchSession(config_path=path, results_path=output, max_particles=2))
    assert len(model.tracks) == 2
    assert model.total_tracks == 4
    np.testing.assert_allclose(model.tracks[:, 0], model.seeds[model.track_seed_indices])
    original = model.track_seed_indices.copy()
    model.load_tracks(output)
    np.testing.assert_array_equal(model.track_seed_indices, original)


def test_geographic_seeds_and_tracks_share_metre_display_coordinates(case):
    path, output = case
    config = yaml.safe_load(path.read_text())
    config['general']['input_model'].update(
        {
            'coordinate_system': 'geographic',
            'source_crs': 'EPSG:4326',
            'metric_crs': 'EPSG:32631',
        }
    )
    path.write_text(yaml.safe_dump(config))
    bathy_path = path.parent / 'bathy.nc'
    with xr.open_dataset(bathy_path) as ds:
        bathy = ds.load()
    bathy['globalx'] = 4 + bathy.globalx * 0.0001
    bathy['globaly'] = 52 + bathy.globaly * 0.0001
    bathy.to_netcdf(bathy_path, mode='w')
    with xr.open_dataset(output) as ds:
        tracks = ds.load()
    tracks['x'] = (4 + tracks.x * 0.0001).astype('float32')
    tracks['y'] = (52 + tracks.y * 0.0001).astype('float32')
    tracks.to_netcdf(output, mode='w')
    (path.parent / 'seeds.txt').write_text('4 52\n4.0001 52\n4.0002 52\n4.0003 52\n')
    model = PatchModel(PatchSession(config_path=path, results_path=output))
    assert model.coordinates.uses_metres
    assert model.seeds[1, 0] - model.seeds[0, 0] > 6
    np.testing.assert_array_equal(model.track_seed_indices, [0, 1, 2, 3])
    np.testing.assert_allclose(model.tracks[:, 0], model.seeds, atol=0.3, rtol=0)


def test_float32_origin_rounding_uses_exact_yaml_seed(case):
    path, output = case
    (path.parent / 'seeds.txt').write_text('0.60040282 0\n1 0\n2 0\n3 0\n')
    with xr.open_dataset(output) as ds:
        data = ds.load()
    data['x'][0, 0] = np.float32(0.60040282)
    data.to_netcdf(output, mode='w')
    session = PatchSession(
        config_path=path, results_path=output, patches=[SquarePatch('edge', 0.60040282, 0, 0.1, (255, 0, 0))]
    )
    model = PatchModel(session)
    assert model.track_seed_indices[0] == 0
    assert classify_points(model.seeds, session.patches)[0] == 0


@pytest.mark.parametrize('kind', ['wrong-seeds', 'missing-origin', 'restart', 'checkpoint'])
def test_invalid_output_does_not_replace_loaded_tracks(case, kind):
    path, output = case
    model = PatchModel(PatchSession(config_path=path, results_path=output))
    tracks = model.tracks
    with xr.open_dataset(output) as ds:
        data = ds.load()
    if kind == 'wrong-seeds':
        data['x'][0, 0] = 100
    elif kind == 'missing-origin':
        data['x'][0, 0] = np.nan
    elif kind == 'restart':
        data['time'] = data.time + 60
    else:
        data = data.isel(time=0)
    invalid = output.with_name('invalid.nc')
    data.to_netcdf(invalid)
    with pytest.raises(ValueError):
        model.load_tracks(invalid)
    assert model.tracks is tracks
    assert model.session.results_path == output


def test_nan_gaps_are_not_joined(case):
    path, output = case
    with xr.open_dataset(output) as ds:
        data = ds.load()
    data['x'][1, 0] = np.nan
    data.to_netcdf(output, mode='w')
    model = PatchModel(PatchSession(config_path=path, results_path=output))
    plot = PatchPlot(Figure(), model)
    assert np.isnan(plot.track_artist.get_paths()[0].vertices[1, 0])


def test_json_roundtrip_restores_plot_and_png(case, tmp_path):
    from PIL import Image

    path, output = case
    session = PatchSession(
        config_path=path,
        results_path=output,
        background=(10, 20, 30),
        patches=[SquarePatch('test', 0, 0, 1, (100, 150, 200))],
        show_patches=False,
        xlim=(-1, 2),
        ylim=(-0.5, 2.5),
    )
    saved = tmp_path / 'session.json'
    save_session(saved, session)
    reopened = load_session(saved)
    assert reopened == session
    plot = PatchPlot(Figure(), PatchModel(reopened))
    np.testing.assert_allclose(plot.ax.get_xlim(), session.xlim)
    np.testing.assert_allclose(plot.ax.get_ylim(), session.ylim)
    png = tmp_path / 'view.png'
    plot.export_png(png)
    with Image.open(png) as image:
        assert image.format == 'PNG'
        assert image.width > 500 and image.height > 500


def test_colormap_option_selects_bathymetry_palette(case):
    path, _ = case
    model = PatchModel(PatchSession(config_path=path, colormap='Grayscale'))
    plot = PatchPlot(Figure(), model)
    assert plot.colormap_name == 'Grayscale'
    plot.set_colormap('Vintage')
    assert plot.colormap_name == 'Vintage'


def test_invalid_colormap_is_rejected(case):
    path, _ = case
    with pytest.raises(ValueError, match='colormap'):
        PatchSession(config_path=path, colormap='not-a-colormap').validate()


def test_show_background_hides_unclassified_seeds_and_tracks(case):
    path, output = case
    session = PatchSession(
        config_path=path,
        results_path=output,
        patches=[SquarePatch('p', 0, 0, 1, (255, 0, 0))],
        show_background=False,
    )
    model = PatchModel(session)
    plot = PatchPlot(Figure(), model)
    labels = classify_points(model.seeds, session.patches)
    assert np.count_nonzero(labels == -1) > 0
    assert len(plot.seed_artist.get_offsets()) == np.count_nonzero(labels != -1)
    track_labels = labels[model.track_seed_indices[plot.track_order]]
    assert np.all(track_labels != -1)


def test_relative_paths_in_json_are_relative_to_json(case, tmp_path):
    path, _ = case
    saved = tmp_path / 'relative.json'
    saved.write_text(json.dumps({'version': 1, 'config_path': path.name}))
    assert load_session(saved).config_path == path


@pytest.mark.parametrize(
    'changes',
    [
        {'version': 2},
        {'background': [300, 0, 0]},
        {'max_particles': 0},
        {'show_tracks': 'false'},
        {'xlim': [1, 1]},
        {'unknown': 1},
        {'patches': [{'name': 'x', 'x': 0, 'y': 0, 'size': -1, 'rgb': [0, 0, 0]}]},
    ],
)
def test_invalid_sessions_are_rejected(tmp_path, changes):
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps({'version': 1, **changes}))
    with pytest.raises(ValueError):
        load_session(path)


def test_project_relative_paths_preserve_existing_precedence(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    folder = project / 'cases' / 'case'
    folder.mkdir(parents=True)
    cwd = tmp_path / 'working'
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    (project / 'data.nc').touch()
    assert _resolve_relative_path('data.nc', folder) == project / 'data.nc'
    (cwd / 'data.nc').touch()
    assert _resolve_relative_path('data.nc', folder) == cwd / 'data.nc'
    (folder / 'data.nc').touch()
    assert _resolve_relative_path('data.nc', folder) == folder / 'data.nc'


@pytest.fixture(scope='module')
def tk_root():
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f'Tk display unavailable: {exc}')
    root.withdraw()
    try:
        yield root
    finally:
        root.destroy()


@pytest.fixture
def desktop(case, tk_root):
    path, _ = case
    app = PatchGuiApp(tk_root, PatchSession(config_path=path))
    try:
        yield app
    finally:
        for widget in tk_root.winfo_children():
            widget.destroy()


def test_default_fields_have_expected_values(desktop):
    assert desktop.fields['size'].get() == '1.5'
    assert desktop.fields['row_spacing'].get() == '1'
    assert desktop.fields['column_spacing'].get() == '1'
    assert desktop.colormap.get() == 'SEAWAD'


def test_gui_add_series_edit_delete_and_layers(desktop):
    app = desktop
    app.fields['rows'].set('2')
    app.fields['columns'].set('3')
    app.add_patches()
    assert len(app.session.patches) == 6
    assert [(p.x, p.y) for p in app.session.patches] == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
    app.table.selection_set('0')
    app.select_patch()
    app.fields['rgb'].set('0,0,255')
    app.update_patch()
    assert app.session.patches[0].rgb == (0, 0, 255)
    app.table.selection_set('1', '2')
    app.delete_patches()
    assert len(app.session.patches) == 4
    app.background.set('10,20,30')
    app.visibility['show_seeds'].set(False)
    app.apply_layers()
    assert app.session.background == (10, 20, 30)
    assert app.plot.seed_artist is None


def test_gui_series_spacing_is_independent_of_side_length(desktop):
    app = desktop
    app.fields['size'].set('2')
    app.fields['row_spacing'].set('0.5')
    app.fields['column_spacing'].set('3')
    app.fields['rows'].set('2')
    app.fields['columns'].set('2')
    app.add_patches()
    assert [(p.x, p.y) for p in app.session.patches] == [(0, 0), (3, 0), (0, 0.5), (3, 0.5)]


def test_gui_colormap_dropdown_switches_bathymetry_palette(desktop):
    app = desktop
    app.colormap.set('Grayscale')
    assert app.plot.colormap_name == 'Grayscale'
    assert app.session.colormap == 'Grayscale'


def test_gui_background_checkbox_hides_unclassified_layers(desktop):
    app = desktop
    app.visibility['show_background'].set(False)
    app.apply_layers()
    assert app.session.show_background is False
    assert len(app.plot.seed_artist.get_offsets()) == 0


def test_gui_load_save_reopen_and_export(desktop, case, tmp_path, monkeypatch):
    app = desktop
    app.load_associated()
    assert len(app.model.tracks) == 4
    app.fields['size'].set('0.5')
    app.add_patches()
    app.plot.ax.set_xlim(-1, 4)
    saved = tmp_path / 'desktop.json'
    monkeypatch.setattr('tkinter.filedialog.asksaveasfilename', lambda **kwargs: str(saved))
    app.save_json()
    original_colours = app.plot.track_artist.get_colors().copy()
    monkeypatch.setattr('tkinter.filedialog.askopenfilename', lambda **kwargs: str(saved))
    app.open_json()
    np.testing.assert_allclose(app.plot.track_artist.get_colors(), original_colours)
    np.testing.assert_allclose(app.plot.ax.get_xlim(), (-1, 4))
    png = tmp_path / 'desktop.png'
    monkeypatch.setattr('tkinter.filedialog.asksaveasfilename', lambda **kwargs: str(png))
    app.export_png()
    assert png.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')


def test_gui_rejects_duplicate_names_without_changing_patches(desktop):
    desktop.add_patches()
    with pytest.raises(ValueError, match='unique'):
        desktop.add_patches()
    assert len(desktop.session.patches) == 1


def test_gui_callback_reports_error(desktop, monkeypatch):
    errors = []
    monkeypatch.setattr('tkinter.messagebox.showerror', lambda *args, **kwargs: errors.append(args))
    desktop.background.set('invalid')
    desktop._run(desktop.apply_layers)
    assert errors
    assert 'Error:' in desktop.status.cget('text')
    assert desktop.session.background == (120, 120, 120)


def test_launch_rejects_conflicting_inputs_before_opening_window():
    with pytest.raises(Exception, match='either --config or --session'):
        launch_patch_gui('config.yaml', 'session.json')


def test_cli_options_and_errors(monkeypatch):
    from typer.testing import CliRunner
    from sedtrails.application_interfaces.cli import app

    calls = []
    monkeypatch.setattr(
        'sedtrails.application_interfaces.patch_gui.launch_patch_gui',
        lambda **kwargs: calls.append(kwargs),
    )
    runner = CliRunner()
    result = runner.invoke(app, ['viz', 'patches', '--config', 'case.yaml', '--file', 'tracks.nc'])
    assert result.exit_code == 0
    assert calls == [{'config_path': 'case.yaml', 'session_path': None, 'results_path': 'tracks.nc'}]
    result = runner.invoke(app, ['viz', 'patches', '--session', 'saved.json'])
    assert result.exit_code == 0
    assert calls[-1]['session_path'] == 'saved.json'

    def fail(**kwargs):
        raise ValueError('bad file')

    monkeypatch.setattr('sedtrails.application_interfaces.patch_gui.launch_patch_gui', fail)
    result = runner.invoke(app, ['viz', 'patches'])
    assert result.exit_code == 1
    assert 'bad file' in result.output
