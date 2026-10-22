import json
from types import SimpleNamespace

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
from sedtrails.application_interfaces.seeding_gui import SeedingGuiError, _resolve_relative_path


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


@pytest.fixture
def evolving_case(case):
    path, output = case
    x, y = np.meshgrid(np.arange(4.0), np.arange(3.0))
    values = np.stack([-x - y, -x - y + 0.25, -x - y + 0.5])
    values[2, 0, 0] = np.nan
    xr.Dataset(
        {
            'globalx': (('ny', 'nx'), x),
            'globaly': (('ny', 'nx'), y),
            'zb_mean': (('meantime', 'ny', 'nx'), values),
            'zb': (('globaltime', 'ny', 'nx'), np.zeros((4, 3, 4))),
        },
        coords={
            'meantime': xr.DataArray([600.0, 1800.0, 3600.0], dims='meantime', attrs={'units': 's'}),
            'globaltime': [0.0, 600.0, 1200.0, 1800.0],
        },
    ).to_netcdf(path.parent / 'bathy.nc', mode='w')
    return path, output


@pytest.fixture
def population_case(case):
    path, output = case
    config = yaml.safe_load(path.read_text())
    first = config['particles']['populations'][0]
    first['name'] = 'fine'
    second = json.loads(json.dumps(first))
    second['name'] = 'coarse'
    config['particles']['populations'].append(second)
    path.write_text(yaml.safe_dump(config))
    # Both populations start at identical points; output population order is reversed.
    origins = np.tile(np.arange(4.0), 2)
    xr.Dataset(
        {
            'x': (('time', 'n_particles'), np.stack((origins, origins + 0.5))),
            'y': (('time', 'n_particles'), np.stack((np.zeros(8), np.repeat([1.0, 2.0], 4)))),
            'population_id': ('n_particles', np.repeat([0, 1], 4)),
            'population_name': ('n_populations', ['coarse', 'fine']),
            'population_start_idx': ('n_populations', [0, 4]),
            'population_count': ('n_populations', [4, 4]),
        },
        coords={'time': [0.0, 60.0]},
        attrs={'reference_date': '2000-01-01'},
    ).to_netcdf(output, mode='w')
    return path, output


def test_population_identity_survives_shared_origins_and_sampling(population_case):
    path, output = population_case
    model = PatchModel(PatchSession(config_path=path, results_path=output, max_particles=4))
    assert model.population_names == ['fine', 'coarse']
    assert set(model.track_population_ids) == {0, 1}
    np.testing.assert_array_equal(
        model.seed_population_ids[model.track_seed_indices], model.track_population_ids,
    )
    np.testing.assert_array_equal(model.tracks[:, -1, 1], np.where(model.track_population_ids == 1, 1, 2))


def test_population_filter_sizes_endpoints_and_export(population_case, tmp_path):
    path, output = population_case
    session = PatchSession(
        config_path=path, results_path=output, show_endpoints=True,
        population_visibility={'fine': False},
        population_marker_sizes={'fine': 4, 'coarse': 36},
        patches=[SquarePatch('origin', 0, 0, 0.5, (255, 0, 0))],
    )
    model = PatchModel(session)
    plot = PatchPlot(Figure(), model)
    assert len(plot.seed_artist.get_offsets()) == 4
    np.testing.assert_array_equal(plot.seed_artist.get_sizes(), [36] * 4)
    np.testing.assert_array_equal(model.track_population_ids[plot.track_order], [1] * 4)
    np.testing.assert_array_equal(plot.endpoint_artist.get_sizes(), [36] * 4)
    np.testing.assert_allclose(plot.endpoint_artist.get_offsets()[:, 1], 1)
    assert [t.get_text() for t in plot.ax.get_legend().get_texts()] == ['Background (3)', 'origin (1)']
    session.show_background = False
    plot.draw()
    assert len(plot.seed_artist.get_offsets()) == 1
    assert len(plot.track_order) == 1
    assert len(plot.endpoint_artist.get_offsets()) == 1
    session.show_tracks = False
    plot.draw()
    assert plot.track_artist is None
    assert len(plot.endpoint_artist.get_offsets()) == 1
    png = tmp_path / 'population.png'
    plot.export_png(png)
    assert png.stat().st_size > 1000
    session.population_visibility['coarse'] = False
    plot.draw()
    assert len(plot.seed_artist.get_offsets()) == 0
    assert len(plot.endpoint_artist.get_offsets()) == 0
    session.population_visibility = {}
    session.show_background = True
    plot.draw()
    assert len(plot.seed_artist.get_offsets()) == 8
    np.testing.assert_array_equal(plot.seed_artist.get_sizes(), [4] * 4 + [36] * 4)


def test_population_legacy_output_uses_yaml_order_not_nearest_shared_seed(population_case):
    path, output = population_case
    with xr.open_dataset(output) as dataset:
        legacy = dataset.drop_vars(
            ['population_id', 'population_name', 'population_start_idx', 'population_count'],
        ).load()
    legacy.to_netcdf(output, mode='w')
    model = PatchModel(PatchSession(config_path=path, results_path=output))
    np.testing.assert_array_equal(model.track_population_ids, [0] * 4 + [1] * 4)
    np.testing.assert_array_equal(model.seed_population_ids[model.track_seed_indices], model.track_population_ids)


def test_generated_output_population_names_use_verified_yaml_order(population_case):
    path, output = population_case
    with xr.open_dataset(output) as dataset:
        generated = dataset.load()
    generated['population_name'] = ('n_populations', ['population_0', 'population_1'])
    generated.to_netcdf(output, mode='w')
    model = PatchModel(PatchSession(config_path=path, results_path=output))
    np.testing.assert_array_equal(model.track_population_ids, [0] * 4 + [1] * 4)
    generated['population_count'] = ('n_populations', [3, 5])
    generated.to_netcdf(output, mode='w')
    old_tracks = model.tracks
    with pytest.raises(ValueError, match='population counts'):
        model.load_tracks(output)
    assert model.tracks is old_tracks


def test_animation_frames_use_status_flags_time_and_growing_trails(case):
    path, output = case
    with xr.open_dataset(output) as dataset:
        data = dataset.load()
    for name in ('status_domain', 'status_alive', 'status_released'):
        flags = np.ones((3, 4), dtype=np.uint8)
        flags[-1, {'status_domain': 0, 'status_alive': 1, 'status_released': 2}[name]] = 0
        data[name] = (('n_particles', 'time'), flags.T)
    data.to_netcdf(output, mode='w')
    model = PatchModel(PatchSession(config_path=path, results_path=output, show_endpoints=True))
    plot = PatchPlot(Figure(), model)
    plot.draw_animation_frame(0)
    assert len(plot.endpoint_artist.get_offsets()) == 4
    assert all(len(segment) == 1 for segment in plot.track_artist.get_segments())
    plot.draw_animation_frame(2)
    np.testing.assert_allclose(plot.endpoint_artist.get_offsets(), [[1, 0]])
    assert '2000-01-01T00:02:00' in plot.ax.get_title()
    assert 't = 0.03 h' in plot.ax.get_title()
    assert '1 plotted particles in domain' in plot.ax.get_title()
    np.testing.assert_allclose(plot.track_artist.get_segments()[-1], model.tracks[-1])


def test_gif_export_writes_first_and_final_frames_and_preserves_plot(population_case, tmp_path):
    from PIL import Image

    path, output = population_case
    session = PatchSession(
        config_path=path, results_path=output, show_tracks=False,
        population_visibility={'fine': False}, population_marker_sizes={'coarse': 36},
    )
    model = PatchModel(session)
    plot = PatchPlot(Figure(), model)
    plot.ax.set_xlim(-1, 5)
    plot.ax.set_ylim(-1, 3)
    initial_title = plot.ax.get_title()
    initial_artists = plot.artists.copy()
    gif = tmp_path / 'sorting.gif'
    plot.export_gif(gif, frame_step=5, fps=10)
    with Image.open(gif) as image:
        assert image.format == 'GIF'
        assert image.n_frames == 2
        assert image.size == (800, 800)
        assert image.info['duration'] == 100
        first = np.asarray(image.convert('RGB'))
        image.seek(1)
        assert not np.array_equal(first, np.asarray(image.convert('RGB')))
    assert model.session is session
    assert session.show_endpoints is False
    assert plot.endpoint_artist is None
    assert plot.ax.get_title() == initial_title
    assert plot.artists == initial_artists
    np.testing.assert_allclose(plot.ax.get_xlim(), [-1, 5])
    np.testing.assert_allclose(plot.ax.get_ylim(), [-1, 3])


@pytest.mark.parametrize('frame_step,fps', [(0, 10), (-1, 10), (1.5, 10), (True, 10), (1, 0), (1, 101)])
def test_gif_export_rejects_invalid_settings(case, tmp_path, frame_step, fps):
    path, output = case
    plot = PatchPlot(Figure(), PatchModel(PatchSession(config_path=path, results_path=output)))
    with pytest.raises(ValueError, match='GIF'):
        plot.export_gif(tmp_path / 'invalid.gif', frame_step=frame_step, fps=fps)
    assert not (tmp_path / 'invalid.gif').exists()


def test_gif_export_requires_loaded_visible_trajectories(case, tmp_path):
    path, output = case
    model = PatchModel(PatchSession(config_path=path))
    plot = PatchPlot(Figure(), model)
    with pytest.raises(ValueError, match='Load trajectories'):
        plot.export_gif(tmp_path / 'none.gif')
    model.load_tracks(output)
    model.session.population_visibility = {'sand': False}
    with pytest.raises(ValueError, match='No trajectories match'):
        plot.export_gif(tmp_path / 'none.gif')


def test_animation_filters_populations_background_and_left_domain(population_case):
    path, output = population_case
    with xr.open_dataset(output) as dataset:
        data = dataset.load()
    left = np.zeros((2, 8), dtype=np.uint8)
    left[-1, 0] = 1
    data['status_left_domain'] = (('time', 'n_particles'), left)
    data.to_netcdf(output, mode='w')
    session = PatchSession(
        config_path=path, results_path=output, show_endpoints=True, show_background=False,
        population_visibility={'fine': False}, population_marker_sizes={'coarse': 36},
        patches=[SquarePatch('red', 0, 0, 0.5, (255, 0, 0))],
    )
    model = PatchModel(session)
    plot = PatchPlot(Figure(), model)
    plot.draw_animation_frame(0)
    assert len(plot.endpoint_artist.get_offsets()) == 1
    np.testing.assert_array_equal(plot.endpoint_artist.get_sizes(), [36])
    np.testing.assert_allclose(plot.endpoint_artist.get_facecolors()[0, :3], [1, 0, 0])
    plot.draw_animation_frame(1)
    assert len(plot.endpoint_artist.get_offsets()) == 0


def test_gui_gif_export_settings_cancel_and_error_cleanup(desktop, case, monkeypatch, tmp_path):
    app = desktop
    with pytest.raises(ValueError, match='Load trajectories'):
        app.export_gif()
    app.load_associated()
    calls = []
    monkeypatch.setattr(app.plot, 'export_gif', lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr('tkinter.filedialog.asksaveasfilename', lambda **kwargs: '')
    app.export_gif()
    assert not calls
    target = tmp_path / 'test.gif'
    monkeypatch.setattr('tkinter.filedialog.asksaveasfilename', lambda **kwargs: str(target))
    app.gif_frame_step.set('2')
    app.gif_fps.set('5')
    app.export_gif()
    assert calls == [((str(target),), {'frame_step': 2, 'fps': 5})]
    assert 'Exported animation' in app.status.cget('text')
    app.gif_frame_step.set('0')
    with pytest.raises(ValueError, match='GIF frame step'):
        app.export_gif()
    app.gif_frame_step.set('2')

    def fail_export(*args, **kwargs):
        raise OSError('Cannot write GIF')

    monkeypatch.setattr(app.plot, 'export_gif', fail_export)
    errors = []
    monkeypatch.setattr('tkinter.messagebox.showerror', lambda *args, **kwargs: errors.append(args))
    app._run(app.export_gif)
    assert errors
    assert app.root.cget('cursor') == ''
    assert 'Cannot write GIF' in app.status.cget('text')


@pytest.mark.parametrize('size', [0, -1, np.nan, np.inf, True, 'large'])
def test_invalid_population_marker_size_is_rejected(size):
    with pytest.raises(ValueError, match='marker sizes'):
        PatchSession(population_marker_sizes={'sand': size}).validate()


def test_gui_population_controls_apply_without_reloading(population_case, tk_root, monkeypatch):
    path, output = population_case
    app = PatchGuiApp(tk_root, PatchSession(config_path=path, results_path=output))
    try:
        original_tracks = app.model.tracks
        app.population_visibility['fine'].set(False)
        app.population_marker_sizes['coarse'].set('25')
        app.visibility['show_endpoints'].set(True)
        app.apply_layers()
        assert app.model.tracks is original_tracks
        assert len(app.plot.seed_artist.get_offsets()) == 4
        np.testing.assert_array_equal(app.plot.endpoint_artist.get_sizes(), [25] * 4)
        app.plot.ax.set_xlim(-2, 5)
        old_session = app.session
        app.population_marker_sizes['coarse'].set('-1')
        errors = []
        monkeypatch.setattr('tkinter.messagebox.showerror', lambda *args, **kwargs: errors.append(args))
        app._run(app.apply_layers)
        assert errors
        assert app.session is old_session
        np.testing.assert_array_equal(app.plot.endpoint_artist.get_sizes(), [25] * 4)
        np.testing.assert_allclose(app.plot.ax.get_xlim(), [-2, 5])
        app.population_marker_sizes['coarse'].set('25')
        monkeypatch.setattr('tkinter.filedialog.askopenfilename', lambda **kwargs: str(path))
        app.open_yaml()
        assert app.population_visibility['fine'].get() is False
        assert float(app.population_marker_sizes['coarse'].get()) == 25
        assert app.model.tracks is None
    finally:
        for widget in tk_root.winfo_children():
            widget.destroy()


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
        bathymetry_clim=(-4, 2),
    )
    saved = tmp_path / 'session.json'
    save_session(saved, session)
    reopened = load_session(saved)
    payload = json.loads(saved.read_text())
    assert set(payload) == {'version', 'patches', 'background'}
    assert payload['version'] == 2
    assert reopened == PatchSession(patches=session.patches, background=session.background)
    reopened.config_path, reopened.results_path = path, output
    reopened.xlim, reopened.ylim = session.xlim, session.ylim
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


@pytest.mark.parametrize('colormap', ['SEAWAD', 'Vintage', 'Grayscale'])
@pytest.mark.parametrize('limits', [(-4, -1), (-10, -6), (1, 3)])
def test_manual_bathymetry_colour_limits_control_contours_and_colourbar(case, colormap, limits):
    path, _ = case
    model = PatchModel(PatchSession(config_path=path, colormap=colormap, bathymetry_clim=limits))
    plot = PatchPlot(Figure(), model)
    artist = plot.bathymetry_artist
    np.testing.assert_allclose((artist.norm.vmin, artist.norm.vmax), limits)
    np.testing.assert_allclose((artist.levels[0], artist.levels[-1]), limits)
    np.testing.assert_allclose((plot.colorbar.vmin, plot.colorbar.vmax), limits)
    np.testing.assert_allclose(artist.to_rgba([-100, 100]), artist.cmap([0.0, 1.0]))
    assert artist.extend == 'both'
    assert any(path.vertices.size for path in artist.get_paths())
    assert len(plot.figure.axes) == 2


@pytest.mark.parametrize(
    'limits',
    [(0, 0), (1, -1), (np.nan, 1), (0, np.inf), (-1e308, 1e308), (1, np.nextafter(1.0, np.inf)), (1,)],
)
def test_invalid_bathymetry_colour_limits_are_rejected(limits):
    with pytest.raises(ValueError, match='Bathymetry colour limits'):
        PatchSession(bathymetry_clim=limits).validate()


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
        {'version': 3},
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


@pytest.fixture
def evolving_desktop(evolving_case, tk_root):
    path, _ = evolving_case
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
    for channel, value in zip('rgb', (0, 0, 255), strict=True):
        app.patch_rgb[channel].set(str(value))
    app.update_patch()
    assert app.session.patches[0].rgb == (0, 0, 255)
    app.table.selection_set('1', '2')
    app.delete_patches()
    assert len(app.session.patches) == 4
    for channel, value in zip('rgb', (10, 20, 30), strict=True):
        app.background[channel].set(str(value))
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


def test_gui_manual_bathymetry_colour_limits_persist_and_can_return_to_auto(
    evolving_desktop, evolving_case, tmp_path, monkeypatch
):
    app = evolving_desktop
    app.add_patches()
    app.load_associated()
    seeds, tracks = app.model.seeds, app.model.tracks
    app.plot.ax.set_xlim(-1, 4)
    app.plot.ax.set_ylim(-1, 3)
    assert app.bathymetry_scale.get() == 'Auto'
    assert all(str(entry.cget('state')) == 'disabled' for entry in app.bathymetry_colour_entries.values())
    app.bathymetry_scale.set('Manual')
    app.bathymetry_scale_box.event_generate('<<ComboboxSelected>>')
    assert app.session.bathymetry_clim == (-5, 0)
    assert all(str(entry.cget('state')) == 'normal' for entry in app.bathymetry_colour_entries.values())
    for key, value in (('min', '-3.5'), ('max', '0.5')):
        app.bathymetry_colour_limits[key].set(value)
    app.apply_layers()
    app.bathymetry_index.set('1')
    app.apply_bathymetry()
    app.colormap.set('Grayscale')
    assert app.session.bathymetry_clim == (-3.5, 0.5)
    np.testing.assert_allclose((app.plot.colorbar.vmin, app.plot.colorbar.vmax), (-3.5, 0.5))
    assert app.model.seeds is seeds and app.model.tracks is tracks
    assert len(app.plot.figure.axes) == 2
    png = tmp_path / 'manual-colours.png'
    monkeypatch.setattr('tkinter.filedialog.asksaveasfilename', lambda **kwargs: str(png))
    app.export_png()
    assert png.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    np.testing.assert_allclose((app.plot.colorbar.vmin, app.plot.colorbar.vmax), (-3.5, 0.5))
    path, _ = evolving_case
    other = tmp_path / 'other.yaml'
    other.write_text(path.read_text())
    monkeypatch.setattr('tkinter.filedialog.askopenfilename', lambda **kwargs: str(other))
    app.open_yaml()
    assert app.session.bathymetry_clim == (-3.5, 0.5)
    np.testing.assert_allclose((app.plot.colorbar.vmin, app.plot.colorbar.vmax), (-3.5, 0.5))
    np.testing.assert_allclose(app.plot.ax.get_xlim(), (-1, 4))
    np.testing.assert_allclose(app.plot.ax.get_ylim(), (-1, 3))
    app.bathymetry_colour_limits['min'].set('discard this unapplied edit')
    app.bathymetry_scale.set('Auto')
    app.bathymetry_scale_box.event_generate('<<ComboboxSelected>>')
    assert app.session.bathymetry_clim is None
    np.testing.assert_allclose((app.plot.vmin, app.plot.vmax), (-5, 0))
    app.bathymetry_index.set('2')
    app.apply_bathymetry()
    data = app.model.bathymetry.values
    np.testing.assert_allclose((app.plot.vmin, app.plot.vmax), (data.min(), data.max()))
    assert [float(app.bathymetry_colour_limits[key].get()) for key in ('min', 'max')] == [app.plot.vmin, app.plot.vmax]


@pytest.mark.parametrize('minimum,maximum', [('bad', '1'), ('nan', '1'), ('0', 'inf'), ('1', '1'), ('2', '1')])
def test_gui_invalid_bathymetry_colour_limits_preserve_session_and_plot(desktop, monkeypatch, minimum, maximum):
    app = desktop
    session, artist = app.session, app.plot.bathymetry_artist
    app.bathymetry_scale.set('Manual')
    app.bathymetry_colour_limits['min'].set(minimum)
    app.bathymetry_colour_limits['max'].set(maximum)
    errors = []
    monkeypatch.setattr('tkinter.messagebox.showerror', lambda *args, **kwargs: errors.append(args))
    app._run(app.apply_layers)
    assert len(errors) == 1 and 'Error:' in app.status.cget('text')
    assert app.session is session and app.model.session is session
    assert app.plot.bathymetry_artist is artist


def test_gui_loading_tracks_applies_pending_bathymetry_colour_limits(desktop):
    app = desktop
    app.bathymetry_scale.set('Manual')
    app.bathymetry_colour_limits['min'].set('-2')
    app.bathymetry_colour_limits['max'].set('0')
    app.load_associated()
    assert app.session.bathymetry_clim == (-2, 0)
    np.testing.assert_allclose((app.plot.colorbar.vmin, app.plot.colorbar.vmax), (-2, 0))


def test_gui_background_checkbox_hides_unclassified_layers(desktop):
    app = desktop
    app.visibility['show_background'].set(False)
    app.apply_layers()
    assert app.session.show_background is False
    assert len(app.plot.seed_artist.get_offsets()) == 0


def test_gui_rgb_channels_and_action_layout(desktop):
    from tkinter import ttk

    app = desktop
    assert {channel: value.get() for channel, value in app.patch_rgb.items()} == {'r': '255', 'g': '0', 'b': '0'}
    assert {channel: value.get() for channel, value in app.background.items()} == {'r': '120', 'g': '120', 'b': '120'}
    patches = app.table.master
    children = patches.winfo_children()
    actions = [widget for widget in children if isinstance(widget, (ttk.Button, ttk.Checkbutton))]
    assert [widget.cget('text') for widget in actions] == [
        'Update selected square',
        'Delete selected square(s)',
        'Live update (arrow keys: 0.1 m)',
        'Add square / row-column series',
    ]
    series = next(
        widget
        for widget in children
        if isinstance(widget, ttk.Frame)
        and any(
            isinstance(child, ttk.Label) and child.cget('text') == 'Series rows (+y)'
            for child in widget.winfo_children()
        )
    )
    assert all(children.index(widget) < children.index(series) for widget in actions[:3])


@pytest.mark.parametrize('channel,value', [('r', '-1'), ('g', '256'), ('b', '1.5'), ('r', '')])
def test_gui_rejects_invalid_patch_rgb_channel(desktop, channel, value):
    desktop.patch_rgb[channel].set(value)
    with pytest.raises(ValueError, match='RGB'):
        desktop.add_patches()
    assert desktop.session.patches == []


def test_gui_live_movement_recolours_seeds_and_tracks_and_keeps_selection(desktop):
    app = desktop
    app.fields['size'].set('0.05')
    app.add_patches()
    app.load_associated()
    app.table.selection_set('0')
    app.select_patch()
    app.live_update.set(True)
    app.toggle_live_update()
    app.plot.ax.set_xlim(-1, 4)
    app.plot.ax.set_ylim(-1, 3)
    np.testing.assert_allclose(app.plot.seed_artist.get_facecolors()[0, :3], [1, 0, 0])

    assert app._on_patch_arrow(SimpleNamespace(keysym='Right')) == 'break'

    assert app.session.patches[0].x == 0.1
    assert app.fields['x'].get() == '0.1'
    assert app.table.selection() == ('0',)
    assert float(app.table.item('0', 'values')[0]) == 0.1
    np.testing.assert_allclose(app.plot.seed_artist.get_facecolors()[:, :3], np.full((4, 3), 120 / 255))
    np.testing.assert_allclose(app.plot.track_artist.get_colors()[:, :3], np.full((4, 3), 120 / 255))
    np.testing.assert_allclose(app.plot.ax.get_xlim(), (-1, 4))
    np.testing.assert_allclose(app.plot.ax.get_ylim(), (-1, 3))

    app._on_patch_arrow(SimpleNamespace(keysym='Left'))
    np.testing.assert_allclose(app.plot.seed_artist.get_facecolors()[0, :3], [1, 0, 0])
    origin = np.flatnonzero(app.plot.track_order == 0)[0]
    np.testing.assert_allclose(app.plot.track_artist.get_colors()[origin, :3], [1, 0, 0])


@pytest.mark.parametrize(
    'key,expected', [('Left', (-0.1, 0)), ('Right', (0.1, 0)), ('Down', (0, -0.1)), ('Up', (0, 0.1))]
)
@pytest.mark.parametrize('selection', [('0',), ('0', '2')])
def test_gui_live_arrow_directions(desktop, key, expected, selection):
    app = desktop
    app.fields['columns'].set('3')
    app.add_patches()
    original = list(app.session.patches)
    app.table.selection_set(*selection)
    app.select_patch()
    app.live_update.set(True)
    app._on_patch_arrow(SimpleNamespace(keysym=key))
    for index, (before, after) in enumerate(zip(original, app.session.patches, strict=True)):
        dx, dy = expected if str(index) in selection else (0, 0)
        assert (after.x, after.y) == (before.x + dx, before.y + dy)
        assert (after.name, after.size, after.rgb) == (before.name, before.size, before.rgb)
    assert app.table.selection() == selection


def test_gui_live_group_movement_updates_colours_once_and_keeps_spacing(desktop, monkeypatch):
    app = desktop
    app.fields['size'].set('0.05')
    app.fields['columns'].set('3')
    app.add_patches()
    app.table.selection_set('2')
    app.select_patch()
    app.patch_rgb['r'].set('0')
    app.patch_rgb['b'].set('255')
    app.update_patch()
    app.table.selection_set('0', '2')
    app.select_patch()
    app.load_associated()
    tracks, seeds = app.model.tracks, app.model.seeds
    app.live_update.set(True)
    app.toggle_live_update()
    app.plot.ax.set_xlim(-1, 4)
    app.plot.ax.set_ylim(-1, 3)
    draw = app.plot.draw
    draws = []
    errors = []
    monkeypatch.setattr('tkinter.messagebox.showerror', lambda *args, **kwargs: errors.append(args))

    def counted_draw():
        draws.append(True)
        draw()

    monkeypatch.setattr(app.plot, 'draw', counted_draw)
    app._on_patch_arrow(SimpleNamespace(keysym='Right'))
    assert errors == []
    assert draws == [True]
    assert app.table.selection() == ('0', '2')
    assert [patch.x for patch in app.session.patches] == [0.1, 1, 2.1]
    assert app.session.patches[2].x - app.session.patches[0].x == 2
    assert float(app.table.item('2', 'values')[0]) == 2.1
    assert app.fields['x'].get() == '0.1'
    expected = np.full((4, 3), 120 / 255)
    expected[1] = [1, 0, 0]
    np.testing.assert_allclose(app.plot.seed_artist.get_facecolors()[:, :3], expected)
    np.testing.assert_allclose(app.plot.track_artist.get_colors()[np.argsort(app.plot.track_order), :3], expected)
    app._on_patch_arrow(SimpleNamespace(keysym='Left'))
    assert len(draws) == 2
    expected[0], expected[2] = [1, 0, 0], [0, 0, 1]
    np.testing.assert_allclose(app.plot.seed_artist.get_facecolors()[:, :3], expected)
    np.testing.assert_allclose(app.plot.track_artist.get_colors()[np.argsort(app.plot.track_order), :3], expected)
    assert app.model.tracks is tracks and app.model.seeds is seeds
    np.testing.assert_allclose(app.plot.ax.get_xlim(), (-1, 4))
    np.testing.assert_allclose(app.plot.ax.get_ylim(), (-1, 3))


def test_gui_live_group_steps_do_not_drift_at_classification_edges(desktop):
    app = desktop
    app.fields['size'].set('0.05')
    app.fields['columns'].set('2')
    app.fields['column_spacing'].set('2')
    app.add_patches()
    app.table.selection_set('0', '1')
    app.live_update.set(True)
    for _ in range(10):
        app._on_patch_arrow(SimpleNamespace(keysym='Right'))
    assert [patch.x for patch in app.session.patches] == [1, 3]
    np.testing.assert_array_equal(classify_points(app.model.seeds, app.session.patches), [-1, 0, -1, 1])
    for _ in range(10):
        app._on_patch_arrow(SimpleNamespace(keysym='Left'))
    assert [patch.x for patch in app.session.patches] == [0, 2]
    np.testing.assert_array_equal(classify_points(app.model.seeds, app.session.patches), [0, -1, 1, -1])
    assert app.table.selection() == ('0', '1')


def test_gui_live_steps_do_not_drift_at_classification_edges(desktop):
    app = desktop
    app.fields['size'].set('0.05')
    app.add_patches()
    app.table.selection_set('0')
    app.select_patch()
    app.live_update.set(True)
    for _ in range(10):
        app._on_patch_arrow(SimpleNamespace(keysym='Right'))
    assert app.session.patches[0].x == 1
    assert classify_points(app.model.seeds, app.session.patches)[1] == 0
    for _ in range(10):
        app._on_patch_arrow(SimpleNamespace(keysym='Left'))
    assert app.session.patches[0].x == 0
    assert classify_points(app.model.seeds, app.session.patches)[0] == 0


def test_gui_live_arrows_do_not_interfere_when_disabled_or_navigating(desktop):
    app = desktop
    app.add_patches()
    app.table.selection_set('0')
    app.select_patch()
    assert app._on_patch_arrow(SimpleNamespace(keysym='Right')) is None
    app.live_update.set(True)
    assert app._on_patch_arrow(SimpleNamespace(keysym='a')) is None
    app.toolbar.pan()
    assert app._on_patch_arrow(SimpleNamespace(keysym='Right')) is None
    app.toolbar.pan()
    app.toolbar.zoom()
    assert app._on_patch_arrow(SimpleNamespace(keysym='Right')) is None
    app.toolbar.zoom()
    assert app.session.patches[0].x == 0


def test_gui_live_movement_reports_empty_selection(desktop, monkeypatch):
    app = desktop
    app.fields['columns'].set('2')
    app.add_patches()
    app.table.selection_remove(*app.table.selection())
    app.live_update.set(True)
    errors = []
    monkeypatch.setattr('tkinter.messagebox.showerror', lambda *args, **kwargs: errors.append(args))
    app._on_patch_arrow(SimpleNamespace(keysym='Right'))
    assert len(errors) == 1
    assert 'Select at least one patch' in app.status.cget('text')
    assert [patch.x for patch in app.session.patches] == [0, 1]


def test_gui_live_map_click_selects_last_overlapping_square(desktop):
    from matplotlib.backend_bases import MouseEvent

    app = desktop
    app.fields['columns'].set('2')
    app.fields['column_spacing'].set('0.5')
    app.add_patches()
    app.canvas.draw()
    x, y = app.plot.ax.transData.transform((0.75, 0.5))
    event = MouseEvent('button_press_event', app.canvas, x, y, button=1)
    app.canvas.callbacks.process('button_press_event', event)
    assert app.table.selection() == ()
    app.live_update.set(True)
    app.canvas.callbacks.process('button_press_event', event)
    assert app.table.selection() == ('1',)
    assert app.fields['name'].get() == 'Patch[1,2]'
    app._on_patch_arrow(SimpleNamespace(keysym='Up'))
    assert app.session.patches[0].y == 0
    assert app.session.patches[1].y == 0.1

    app.table.selection_remove('1')
    app.toolbar.pan()
    app.canvas.callbacks.process('button_press_event', event)
    assert app.table.selection() == ()
    app.toolbar.pan()
    app.session.show_patches = False
    app.canvas.callbacks.process('button_press_event', event)
    assert app.table.selection() == ()


@pytest.mark.parametrize('target', ['table', 'map', 'entry', 'spinbox'])
def test_gui_live_key_bindings_respect_widget_focus(desktop, target):
    from tkinter import ttk

    app = desktop
    app.add_patches()
    app.table.selection_set('0')
    app.select_patch()
    app.live_update.set(True)
    if target == 'table':
        widget = app.table
    elif target == 'map':
        widget = app.canvas.get_tk_widget()
    else:
        editor = next(child for child in app.table.master.winfo_children() if isinstance(child, ttk.Frame))
        variable = app.fields['name' if target == 'entry' else 'x']
        widget = next(
            child
            for child in editor.winfo_children()
            if isinstance(child, (ttk.Entry, ttk.Spinbox)) and child.cget('textvariable') == str(variable)
        )
    app.root.deiconify()
    try:
        app.root.update()
        widget.focus_force()
        widget.event_generate('<KeyPress-Up>' if target == 'spinbox' else '<KeyPress-Right>')
        app.root.update()
        assert app.session.patches[0].x == (0 if target == 'entry' else 0.1)
        assert app.session.patches[0].y == 0
        assert app.table.selection() == ('0',)
    finally:
        app.root.withdraw()


@pytest.mark.parametrize('modifier,selection', [('Shift', ('0', '1', '2')), ('Control', ('0', '2'))])
@pytest.mark.parametrize('target', ['table', 'map'])
def test_gui_live_group_selection_with_real_modifier_clicks_and_keys(desktop, modifier, selection, target):
    app = desktop
    app.fields['columns'].set('3')
    app.add_patches()
    app.live_update.set(True)
    app.root.deiconify()
    try:
        app.root.update()
        for row, prefix in (('0', ''), ('2', f'{modifier}-')):
            x, y, _width, height = app.table.bbox(row, '#0')
            app.table.event_generate(f'<{prefix}ButtonPress-1>', x=x + 20, y=y + height // 2)
            app.table.event_generate(f'<{prefix}ButtonRelease-1>', x=x + 20, y=y + height // 2)
            app.root.update()
        assert app.table.selection() == selection
        focus = app.table.focus()
        widget = app.table if target == 'table' else app.canvas.get_tk_widget()
        widget.focus_force()
        widget.event_generate('<KeyPress-Down>')
        app.root.update()
        assert app.table.selection() == selection
        assert app.table.focus() == focus
        for index, patch in enumerate(app.session.patches):
            assert patch.x == index
            assert patch.y == (-0.1 if str(index) in selection else 0)
    finally:
        app.root.withdraw()


def test_gui_patch_editor_still_requires_one_selected_patch(desktop):
    app = desktop
    app.fields['columns'].set('2')
    app.add_patches()
    app.table.selection_set('0', '1')
    with pytest.raises(ValueError, match='Select one patch to update'):
        app.update_patch()


def test_gui_open_yaml_preserves_patch_settings_but_replaces_run_data(desktop, case, monkeypatch):
    app = desktop
    path, _ = case
    app.add_patches()
    for channel, value in zip('rgb', (10, 20, 30), strict=True):
        app.background[channel].set(str(value))
    app.colormap.set('Grayscale')
    app.visibility['show_background'].set(False)
    app.maximum.set('2')
    app.load_associated()
    app.plot.ax.set_xlim(-50, 50)
    app.plot.ax.set_ylim(-20, 20)
    previous = app.session
    new_path = path.parent / 'new.yaml'
    config = yaml.safe_load(path.read_text())
    config['particles']['populations'][0]['seeding']['strategy']['file_points']['path'] = 'new-seeds.txt'
    (path.parent / 'new-seeds.txt').write_text('0.5 0\n1.5 0\n2.5 0\n', encoding='utf-8')
    new_path.write_text(yaml.safe_dump(config), encoding='utf-8')
    monkeypatch.setattr('tkinter.filedialog.askopenfilename', lambda **kwargs: str(new_path))

    app.open_yaml()

    assert app.session.patches == previous.patches
    assert app.session.patches is not previous.patches
    assert app.session.background == previous.background
    assert app.session.colormap == 'Grayscale'
    assert app.session.max_particles == 2
    assert app.session.show_background is False
    assert app.session.config_path == new_path
    assert app.session.results_path is None
    assert app.model.tracks is None
    assert app.session.xlim == (-50, 50) and app.session.ylim == (-20, 20)
    np.testing.assert_allclose(app.plot.ax.get_xlim(), (-50, 50))
    np.testing.assert_allclose(app.plot.ax.get_ylim(), (-20, 20))
    np.testing.assert_allclose(app.model.seeds, [[0.5, 0], [1.5, 0], [2.5, 0]])
    assert len(app.plot.seed_artist.get_offsets()) == 2


def test_gui_failed_or_cancelled_yaml_load_keeps_current_model(desktop, case, monkeypatch):
    app = desktop
    path, _ = case
    app.add_patches()
    app.load_associated()
    original = app.model
    invalid_path = path.parent / 'invalid.yaml'
    invalid_path.write_text('invalid: [', encoding='utf-8')
    monkeypatch.setattr('tkinter.filedialog.askopenfilename', lambda **kwargs: str(invalid_path))
    with pytest.raises(SeedingGuiError, match='Could not read configuration'):
        app.open_yaml()
    assert app.model is original
    assert app.session is original.session
    assert len(app.session.patches) == 1
    assert app.model.tracks is not None
    monkeypatch.setattr('tkinter.filedialog.askopenfilename', lambda **kwargs: '')
    app.open_yaml()
    assert app.model is original


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
    model, tracks, seeds = app.model, app.model.tracks, app.model.seeds
    original_colours = app.plot.track_artist.get_colors().copy()
    monkeypatch.setattr('tkinter.filedialog.askopenfilename', lambda **kwargs: str(saved))
    app.open_json()
    assert app.model is model and app.model.tracks is tracks and app.model.seeds is seeds
    np.testing.assert_allclose(app.plot.track_artist.get_colors(), original_colours)
    np.testing.assert_allclose(app.plot.ax.get_xlim(), (-1, 4))
    png = tmp_path / 'desktop.png'
    monkeypatch.setattr('tkinter.filedialog.asksaveasfilename', lambda **kwargs: str(png))
    app.export_png()
    assert png.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')


def test_gui_limits_follow_navigation_and_survive_tracks_layers_and_patches(desktop):
    app = desktop
    home = app.plot.ax.get_xlim(), app.plot.ax.get_ylim()
    app.load_associated()
    np.testing.assert_allclose((app.plot.ax.get_xlim(), app.plot.ax.get_ylim()), home)
    app.plot.ax.set_xlim(-20, 30)
    app.plot.ax.set_ylim(-10, 15)
    assert [float(app.view_limits[key].get()) for key in ('xmin', 'xmax', 'ymin', 'ymax')] == [-20, 30, -10, 15]
    assert app.session.xlim == (-20, 30) and app.session.ylim == (-10, 15)
    app.load_associated()
    app.add_patches()
    app.colormap.set('Vintage')
    app.maximum.set('2')
    app.apply_layers()
    assert len(app.model.tracks) == 2
    np.testing.assert_allclose(app.plot.ax.get_xlim(), (-20, 30))
    np.testing.assert_allclose(app.plot.ax.get_ylim(), (-10, 15))
    app.fit_map()
    np.testing.assert_allclose((app.plot.ax.get_xlim(), app.plot.ax.get_ylim()), home)


def test_gui_editable_limits_and_title_persist_across_yaml_load(desktop, case, monkeypatch):
    app = desktop
    for key, value in zip(('xmin', 'xmax', 'ymin', 'ymax'), (-1, 9, -3, 5), strict=True):
        app.view_limits[key].set(str(value))
    app.plot_title.set('Experiment A')
    app.apply_view()
    np.testing.assert_allclose(app.plot.ax.get_xlim(), (-1, 9))
    np.testing.assert_allclose(app.plot.ax.get_ylim(), (-3, 5))
    assert app.plot.ax.get_title() == 'Experiment A'
    path, _ = case
    other = path.parent / 'other.yaml'
    other.write_text(path.read_text())
    monkeypatch.setattr('tkinter.filedialog.askopenfilename', lambda **kwargs: str(other))
    app.open_yaml()
    app.add_patches()
    assert app.plot.ax.get_title() == 'Experiment A'
    assert app.plot_title.get() == 'Experiment A'
    np.testing.assert_allclose(app.plot.ax.get_xlim(), (-1, 9))
    np.testing.assert_allclose(app.plot.ax.get_ylim(), (-3, 5))
    app.plot_title.set('')
    app.apply_view()
    assert app.plot.ax.get_title() == 'other.yaml'
    monkeypatch.setattr('tkinter.filedialog.askopenfilename', lambda **kwargs: str(path))
    app.open_yaml()
    assert app.plot.ax.get_title() == 'case.yaml'


@pytest.mark.parametrize('key,value', [('xmin', 'nan'), ('ymax', 'inf'), ('ymin', 'bad'), ('xmax', '0')])
def test_gui_rejects_invalid_limits_atomically(desktop, key, value):
    app = desktop
    original = app.session
    before = app.plot.ax.get_xlim(), app.plot.ax.get_ylim()
    app.view_limits['xmin'].set('0')
    app.view_limits[key].set(value)
    app.plot_title.set('Should not apply')
    with pytest.raises(ValueError):
        app.apply_view()
    assert app.session is original
    np.testing.assert_allclose((app.plot.ax.get_xlim(), app.plot.ax.get_ylim()), before)
    assert app.plot.ax.get_title() == 'case.yaml'


def test_gui_selects_actual_bathymetry_frame_without_replacing_overlays(evolving_desktop):
    app = evolving_desktop
    app.add_patches()
    app.load_associated()
    app.plot.ax.set_xlim(-1, 4)
    app.plot.ax.set_ylim(-1, 3)
    app.plot_title.set('Evolving bed')
    app.apply_view()
    model, seeds, tracks = app.model, app.model.seeds, app.model.tracks
    artists = tuple(app.plot.artists)
    first_values = model.bathymetry.values.copy()
    assert model.bathymetry.variable == 'zb_mean'
    assert model.bathymetry.time_count == 3
    app.bathymetry_index.set('1')
    app.apply_bathymetry()
    assert app.model is model and model.seeds is seeds and model.tracks is tracks
    assert tuple(app.plot.artists) == artists
    np.testing.assert_allclose(model.bathymetry.values, first_values + 0.25)
    assert model.bathymetry.time_index == app.session.bathymetry_index == 1
    assert 'meantime' in app.bathymetry_label.cget('text') and '1800' in app.bathymetry_label.cget('text')
    assert app.plot.vmin == first_values.min() + 0.25
    assert app.plot.vmax == first_values.max() + 0.25
    app.bathymetry_index.set('2')
    app.apply_bathymetry()
    assert len(model.bathymetry.values) == len(first_values) - 1
    assert len(app.plot.triangulation.x) == len(model.bathymetry.values)
    app.colormap.set('Grayscale')
    assert app.session.bathymetry_index == 2
    assert len(app.plot.figure.axes) == 2
    assert app.plot.ax.get_title() == 'Evolving bed'
    np.testing.assert_allclose(app.plot.ax.get_xlim(), (-1, 4))
    np.testing.assert_allclose(app.plot.ax.get_ylim(), (-1, 3))


@pytest.mark.parametrize('kind', ['index', 'negative', 'fraction', 'empty', 'degenerate'])
def test_gui_bad_bathymetry_frame_preserves_display(evolving_desktop, evolving_case, monkeypatch, kind):
    app = evolving_desktop
    if kind in ('empty', 'degenerate'):
        path, _ = evolving_case
        bathy_path = path.parent / 'bathy.nc'
        with xr.open_dataset(bathy_path) as source:
            data = source.load()
        data['zb_mean'][1] = np.nan
        if kind == 'degenerate':
            data['zb_mean'][1, 0, :2] = [0, 1]
        data.to_netcdf(bathy_path, mode='w')
    before, artist = app.model.bathymetry, app.plot.bathymetry_artist
    limits = app.plot.ax.get_xlim(), app.plot.ax.get_ylim()
    app.bathymetry_index.set({'index': '99', 'negative': '-1', 'fraction': '0.1'}.get(kind, '1'))
    errors = []
    monkeypatch.setattr('tkinter.messagebox.showerror', lambda *args, **kwargs: errors.append(args))
    app._run(app.apply_bathymetry)
    assert len(errors) == 1 and 'Error:' in app.status.cget('text')
    assert app.model.bathymetry is before and app.plot.bathymetry_artist is artist
    assert app.session.bathymetry_index == 0 and app.bathymetry_index.get() == '0'
    np.testing.assert_allclose((app.plot.ax.get_xlim(), app.plot.ax.get_ylim()), limits)
    assert len(app.plot.figure.axes) == 2


def test_gui_single_frame_bathymetry_disables_spinner(desktop):
    assert str(desktop.bathymetry_spinner.cget('state')) == 'disabled'
    assert desktop.model.bathymetry.time_count == 1


def test_new_yaml_starts_at_first_bathymetry_frame_and_keeps_zoom(evolving_desktop, evolving_case, monkeypatch):
    app = evolving_desktop
    app.bathymetry_index.set('2')
    app.apply_bathymetry()
    app.plot.ax.set_xlim(-8, 9)
    app.plot.ax.set_ylim(-3, 4)
    path, _ = evolving_case
    config = yaml.safe_load(path.read_text())
    config['inputs']['data'] = 'static.nc'
    x, y = np.meshgrid(np.arange(4.0), np.arange(3.0))
    xr.Dataset(
        {
            'globalx': (('ny', 'nx'), x),
            'globaly': (('ny', 'nx'), y),
            'zb': (('ny', 'nx'), -x - y),
        }
    ).to_netcdf(path.parent / 'static.nc')
    other = path.parent / 'static.yaml'
    other.write_text(yaml.safe_dump(config))
    monkeypatch.setattr('tkinter.filedialog.askopenfilename', lambda **kwargs: str(other))
    app.open_yaml()
    assert app.session.bathymetry_index == 0 and app.model.bathymetry.time_count == 1
    assert app.model.bathymetry.time_label == 'Static bathymetry'
    np.testing.assert_allclose(app.plot.ax.get_xlim(), (-8, 9))
    np.testing.assert_allclose(app.plot.ax.get_ylim(), (-3, 4))


@pytest.mark.parametrize('version', [1, 2])
def test_gui_imports_only_patch_data_and_keeps_current_run(evolving_desktop, tmp_path, monkeypatch, version):
    app = evolving_desktop
    app.load_associated()
    app.bathymetry_index.set('1')
    app.apply_bathymetry()
    app.plot.ax.set_xlim(-1, 8)
    app.plot.ax.set_ylim(-2, 6)
    app.plot_title.set('Keep this title')
    app.apply_view()
    model, seeds, tracks, bathymetry = app.model, app.model.seeds, app.model.tracks, app.model.bathymetry
    payload = {
        'version': version,
        'patches': [{'name': 'new', 'x': 0, 'y': 0, 'size': 0.5, 'rgb': [0, 0, 255]}],
        'background': [10, 20, 30],
    }
    if version == 1:
        payload.update(config_path='missing.yaml', results_path='missing.nc', show_seeds=False, xlim=[99, 100])
    saved = tmp_path / 'layout.json'
    saved.write_text(json.dumps(payload))
    monkeypatch.setattr('tkinter.filedialog.askopenfilename', lambda **kwargs: str(saved))
    app.open_json()
    assert app.model is model and model.seeds is seeds and model.tracks is tracks
    assert model.bathymetry is bathymetry
    assert app.session.patches == [SquarePatch('new', 0, 0, 0.5, (0, 0, 255))]
    assert app.session.background == (10, 20, 30)
    assert app.session.show_seeds and app.session.bathymetry_index == 1
    assert app.plot.ax.get_title() == 'Keep this title'
    np.testing.assert_allclose(app.plot.ax.get_xlim(), (-1, 8))
    np.testing.assert_allclose(app.plot.ax.get_ylim(), (-2, 6))


def test_gui_can_import_and_save_layout_without_yaml(tk_root, tmp_path, monkeypatch):
    patches = [SquarePatch('portable', 1, 2, 3, (20, 40, 60))]
    source, target = tmp_path / 'source.json', tmp_path / 'target.json'
    save_session(source, PatchSession(patches=patches, background=(10, 20, 30)))
    app = PatchGuiApp(tk_root, PatchSession())
    try:
        monkeypatch.setattr('tkinter.filedialog.askopenfilename', lambda **kwargs: str(source))
        app.open_json()
        assert app.model is None and app.session.patches == patches
        monkeypatch.setattr('tkinter.filedialog.asksaveasfilename', lambda **kwargs: str(target))
        app.maximum.set('unrelated invalid rendering field')
        app.save_json()
        assert load_session(target) == PatchSession(patches=patches, background=(10, 20, 30))
    finally:
        for widget in tk_root.winfo_children():
            widget.destroy()


def test_legacy_json_retains_source_and_render_settings_on_startup(case, tmp_path):
    path, output = case
    saved = tmp_path / 'legacy.json'
    saved.write_text(
        json.dumps(
            {
                'version': 1,
                'config_path': path.name,
                'results_path': output.name,
                'xlim': [-10, 10],
                'ylim': [-5, 5],
                'show_tracks': False,
                'colormap': 'Vintage',
                'max_particles': 2,
            }
        )
    )
    session = load_session(saved)
    assert session.config_path == path and session.results_path == output
    assert session.xlim == (-10, 10) and session.ylim == (-5, 5)
    assert not session.show_tracks and session.colormap == 'Vintage' and session.max_particles == 2


def test_portable_json_rejects_embedded_render_or_source_settings(tmp_path):
    saved = tmp_path / 'bad.json'
    saved.write_text(json.dumps({'version': 2, 'config_path': 'case.yaml'}))
    with pytest.raises(ValueError, match='only patches and background'):
        load_session(saved)


def test_gui_rejects_duplicate_names_without_changing_patches(desktop):
    desktop.add_patches()
    with pytest.raises(ValueError, match='unique'):
        desktop.add_patches()
    assert len(desktop.session.patches) == 1


def test_gui_callback_reports_error(desktop, monkeypatch):
    errors = []
    monkeypatch.setattr('tkinter.messagebox.showerror', lambda *args, **kwargs: errors.append(args))
    desktop.background['r'].set('invalid')
    desktop._run(desktop.apply_layers)
    assert errors
    assert 'Error:' in desktop.status.cget('text')
    assert desktop.session.background == (120, 120, 120)


def test_launch_combines_config_and_portable_layout(case, tmp_path, monkeypatch):
    path, output = case
    saved = tmp_path / 'patches.json'
    patches = [SquarePatch('p', 0, 0, 1, (255, 0, 0))]
    save_session(saved, PatchSession(patches=patches))
    sessions = []
    root = SimpleNamespace(protocol=lambda *args: None, mainloop=lambda: None, quit=lambda: None, destroy=lambda: None)
    monkeypatch.setattr('tkinter.Tk', lambda: root)
    monkeypatch.setattr(
        'sedtrails.application_interfaces.patch_gui.PatchGuiApp',
        lambda root, session: sessions.append(session),
    )
    launch_patch_gui(config_path=path, session_path=saved, results_path=output)
    assert sessions[0].config_path == path
    assert sessions[0].results_path == output
    assert sessions[0].patches == patches


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
