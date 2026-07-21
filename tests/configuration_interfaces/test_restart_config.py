from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

import sedtrails.application_interfaces.restart as restart_module
from sedtrails.application_interfaces.restart import create_restart_from_netcdf


def _trajectory_dataset(
    *,
    x,
    y,
    time,
    population_id=None,
    status_alive=None,
    status_domain=None,
    attrs=None,
):
    """Create a SedTRAILS v2 time-major trajectory fixture."""
    time_values = np.asarray(time, dtype=float)
    data_vars = {
        'x': (('n_timesteps', 'n_particles'), np.asarray(x)),
        'y': (('n_timesteps', 'n_particles'), np.asarray(y)),
        'time': (('n_timesteps',), time_values),
    }
    if population_id is not None:
        data_vars['population_id'] = (('n_particles',), np.asarray(population_id, dtype=int))
    if status_alive is not None:
        data_vars['status_alive'] = (('n_timesteps', 'n_particles'), np.asarray(status_alive))
    if status_domain is not None:
        data_vars['status_domain'] = (('n_timesteps', 'n_particles'), np.asarray(status_domain))

    output_attrs = {'trajectory_layout': 'time_particle', 'written_slots': int(time_values.size)}
    if attrs:
        output_attrs.update(attrs)

    return xr.Dataset(data_vars=data_vars, attrs=output_attrs)


def test_open_restart_dataset_uses_netcdf4_engine(monkeypatch, tmp_path):
    observed = {}

    def fake_open_dataset(path, **kwargs):
        observed['path'] = path
        observed['kwargs'] = kwargs
        return xr.Dataset()

    monkeypatch.setattr(restart_module.xr, 'open_dataset', fake_open_dataset)
    netcdf_file = tmp_path / 'results.nc'

    ds = restart_module._open_restart_dataset(netcdf_file)

    assert isinstance(ds, xr.Dataset)
    assert observed['path'] == netcdf_file
    assert observed['kwargs']['engine'] == 'netcdf4'
    assert observed['kwargs']['decode_times'] is False


def test_create_restart_rejects_forcing_file_with_clear_message(tmp_path):
    base_config = {
        'time': {'start': '2020-01-01 00:00:00', 'timestep': '60S', 'duration': '1D'},
        'particles': {
            'populations': [
                {
                    'name': 'population_1',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2020-01-01 00:00:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                }
            ]
        },
    }
    config_file = tmp_path / 'base.yaml'
    with open(config_file, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)

    forcing = xr.Dataset(
        data_vars={
            'bedlevel': (('time', 'mesh2d_nFaces'), np.zeros((1, 2))),
            'sea_water_x_velocity': (('time', 'mesh2d_nFaces'), np.ones((1, 2))),
        }
    )
    forcing_file = tmp_path / 'forcing.nc'
    forcing.to_netcdf(forcing_file)

    with pytest.raises(ValueError, match='not a SedTRAILS trajectory output file.*Eulerian forcing files'):
        create_restart_from_netcdf(
            netcdf_file=str(forcing_file),
            base_config_file=str(config_file),
            output_config_file=str(tmp_path / 'restart.yaml'),
        )


def test_create_restart_from_netcdf_generates_yaml_and_seed_files(tmp_path):
    base_config = {
        'time': {'start': '2020-01-01 00:00:00', 'timestep': '60S', 'duration': '1D'},
        'particles': {
            'populations': [
                {
                    'name': 'population_1',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2020-01-01 00:00:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                },
                {
                    'name': 'population_2',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2020-01-01 00:00:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                },
            ]
        },
    }

    config_file = tmp_path / 'base.yaml'
    with open(config_file, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)

    ds = _trajectory_dataset(
        x=np.array(
            [
                [0.0, 5.0, 10.0],
                [1.0, 6.0, 11.0],
                [2.0, np.nan, 12.0],
            ]
        ),
        y=np.array(
            [
                [0.0, 4.0, 7.0],
                [0.5, 4.5, 7.5],
                [1.0, np.nan, 8.0],
            ]
        ),
        time=np.array([0.0, 60.0, 120.0]),
        population_id=np.array([0, 0, 1], dtype=int),
        status_alive=np.array(
            [
                [1.0, 1.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
            ]
        ),
        status_domain=np.ones((3, 3), dtype=float),
    )

    netcdf_file = tmp_path / 'results.nc'
    ds.to_netcdf(netcdf_file)

    out_config = tmp_path / 'restart.yaml'
    out_seed_dir = tmp_path / 'restart_points'

    summary = create_restart_from_netcdf(
        netcdf_file=str(netcdf_file),
        base_config_file=str(config_file),
        output_config_file=str(out_config),
        seed_points_dir=str(out_seed_dir),
    )

    assert summary.output_config == out_config
    assert summary.restart_time == '2020-01-01 00:02:00'
    assert summary.retained_particles == 2

    assert (out_seed_dir / 'population_1.restart_points.csv').exists()
    assert (out_seed_dir / 'population_2.restart_points.csv').exists()

    with open(out_config, 'r', encoding='utf-8') as handle:
        restart_cfg = yaml.safe_load(handle)

    assert restart_cfg['time']['start'] == '2020-01-01 00:02:00'
    assert restart_cfg['time']['duration'] == '23H58M'

    pop1 = restart_cfg['particles']['populations'][0]
    pop2 = restart_cfg['particles']['populations'][1]

    assert pop1['seeding']['release_start'] == '2020-01-01 00:02:00'
    assert pop1['seeding']['quantity'] == 1
    assert 'file_points' in pop1['seeding']['strategy']

    assert pop2['seeding']['release_start'] == '2020-01-01 00:02:00'
    assert pop2['seeding']['quantity'] == 1
    assert 'file_points' in pop2['seeding']['strategy']

    pop1_file = Path(pop1['seeding']['strategy']['file_points']['path'])
    pop2_file = Path(pop2['seeding']['strategy']['file_points']['path'])

    assert pop1_file.name == 'population_1.restart_points.csv'
    assert pop2_file.name == 'population_2.restart_points.csv'


def test_create_restart_from_time_major_netcdf_reads_final_written_slot(tmp_path):
    base_config = {
        'time': {'start': '2020-01-01 00:00:00', 'timestep': '60S', 'duration': '1D'},
        'particles': {
            'populations': [
                {
                    'name': 'population_1',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2020-01-01 00:00:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                },
                {
                    'name': 'population_2',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2020-01-01 00:00:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                },
            ]
        },
    }

    config_file = tmp_path / 'base.yaml'
    with open(config_file, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)

    ds = xr.Dataset(
        data_vars={
            'x': (
                ('n_timesteps', 'n_particles'),
                np.array(
                    [
                        [0.0, 5.0, 10.0],
                        [1.0, 6.0, 11.0],
                        [2.0, 7.0, 12.0],
                        [np.nan, np.nan, np.nan],
                    ],
                    dtype=np.float32,
                ),
            ),
            'y': (
                ('n_timesteps', 'n_particles'),
                np.array(
                    [
                        [0.0, 4.0, 7.0],
                        [0.5, 4.5, 7.5],
                        [1.0, 5.0, 8.0],
                        [np.nan, np.nan, np.nan],
                    ],
                    dtype=np.float32,
                ),
            ),
            'time': (('n_timesteps',), np.array([0.0, 60.0, 120.0, np.nan])),
            'population_id': (('n_particles',), np.array([0, 0, 1], dtype=int)),
            'status_alive': (
                ('n_timesteps', 'n_particles'),
                np.array(
                    [
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 0, 1],
                        [0, 0, 0],
                    ],
                    dtype=np.uint8,
                ),
            ),
            'status_domain': (
                ('n_timesteps', 'n_particles'),
                np.ones((4, 3), dtype=np.uint8),
            ),
        },
        attrs={'trajectory_layout': 'time_particle', 'written_slots': 3},
    )

    netcdf_file = tmp_path / 'results_v2.nc'
    ds.to_netcdf(netcdf_file)

    summary = create_restart_from_netcdf(
        netcdf_file=str(netcdf_file),
        base_config_file=str(config_file),
        output_config_file=str(tmp_path / 'restart.yaml'),
    )

    assert summary.restart_time == '2020-01-01 00:02:00'
    assert summary.retained_particles == 2


def test_create_restart_from_checkpoint_netcdf(tmp_path):
    base_config = {
        'time': {'start': '2020-01-01 00:00:00', 'timestep': '60S', 'duration': '1D'},
        'particles': {
            'populations': [
                {
                    'name': 'population_1',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2020-01-01 00:00:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                },
                {
                    'name': 'population_2',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2020-01-01 00:00:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                },
            ]
        },
    }

    config_file = tmp_path / 'base.yaml'
    with open(config_file, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)

    ds = xr.Dataset(
        data_vars={
            'x': (('n_particles',), np.array([2.0, 12.0], dtype=np.float32)),
            'y': (('n_particles',), np.array([1.0, 8.0], dtype=np.float32)),
            'time': ((), 180.0),
            'population_id': (('n_particles',), np.array([0, 1], dtype=int)),
            'status_alive': (('n_particles',), np.array([1, 1], dtype=np.uint8)),
            'status_domain': (('n_particles',), np.array([1, 1], dtype=np.uint8)),
        },
        attrs={'sedtrails_file_kind': 'checkpoint'},
    )

    netcdf_file = tmp_path / 'checkpoint.nc'
    ds.to_netcdf(netcdf_file)

    summary = create_restart_from_netcdf(
        netcdf_file=str(netcdf_file),
        base_config_file=str(config_file),
        output_config_file=str(tmp_path / 'restart.yaml'),
    )

    assert summary.restart_time == '2020-01-01 00:03:00'
    assert summary.retained_particles == 2


def test_restart_state_rejects_particle_major_legacy_output():
    ds = xr.Dataset(
        data_vars={
            'x': (('n_particles', 'n_timesteps'), np.array([[0.0, 1.0]])),
            'y': (('n_particles', 'n_timesteps'), np.array([[0.0, 1.0]])),
            'time': (('n_particles', 'n_timesteps'), np.array([[0.0, 60.0]])),
            'population_id': (('n_particles',), np.array([0], dtype=int)),
        }
    )

    with pytest.raises(ValueError, match='Legacy particle-major trajectory files are not supported'):
        restart_module._extract_restart_state(ds)


def test_restart_rejects_incompatible_coordinate_context():
    """A checkpoint must not be combined with a different runtime geometry."""
    ds = xr.Dataset(
        attrs={
            'coordinate_system': 'geographic',
            'runtime_geometry': 'geodetic',
            'source_crs': 'EPSG:4326',
            'surface_model': 'sphere',
            'earth_radius_m': 6_371_008.8,
        }
    )
    config = {
        'general': {
            'input_model': {
                'coordinate_system': 'geographic',
                'runtime_geometry': 'planar',
                'source_crs': 'EPSG:4326',
                'surface_model': 'sphere',
                'earth_radius_m': 6_371_008.8,
            }
        }
    }

    with pytest.raises(ValueError, match='runtime_geometry'):
        restart_module._validate_restart_coordinate_compatibility(ds, config)


def test_restart_accepts_equivalent_geographic_alias_and_crs():
    """Equivalent geographic aliases and CRS labels should remain restartable."""
    ds = xr.Dataset(
        attrs={
            'coordinate_system': 'geographic',
            'runtime_geometry': 'geodetic',
            'source_crs': 'EPSG:4326',
            'surface_model': 'sphere',
            'earth_radius_m': 6_371_008.8,
        }
    )
    config = {
        'general': {
            'input_model': {
                'coordinate_system': 'spherical',
                'runtime_geometry': 'geodetic',
                'source_crs': 'WGS84',
                'surface_model': 'sphere',
                'earth_radius_m': 6_371_008.8,
            }
        }
    }

    restart_module._validate_restart_coordinate_compatibility(ds, config)


def test_create_restart_uses_reference_date_for_netcdf_time(tmp_path):
    base_config = {
        'general': {'input_model': {'reference_date': '1970-01-01'}},
        'time': {'start': '2016-09-21 19:20:00', 'timestep': '60S', 'duration': '1D'},
        'particles': {
            'populations': [
                {
                    'name': 'population_1',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2016-09-21 19:30:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                }
            ]
        },
    }

    config_file = tmp_path / 'base.yaml'
    with open(config_file, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)

    ds = _trajectory_dataset(
        x=np.array([[0.0], [1.0]]),
        y=np.array([[0.0], [1.0]]),
        # 2016-09-21 19:40:28 UTC in epoch seconds
        time=np.array([1474485628.0, 1474486828.0]),
        population_id=np.array([0], dtype=int),
    )

    netcdf_file = tmp_path / 'results.nc'
    ds.to_netcdf(netcdf_file)

    out_config = tmp_path / 'restart.yaml'

    summary = create_restart_from_netcdf(
        netcdf_file=str(netcdf_file),
        base_config_file=str(config_file),
        output_config_file=str(out_config),
    )

    assert summary.restart_time == '2016-09-21 19:40:28'

    with open(out_config, 'r', encoding='utf-8') as handle:
        restart_cfg = yaml.safe_load(handle)
    assert restart_cfg['time']['duration'] == '23H39M32S'
    assert restart_cfg['general']['input_model']['reference_date'] == '1970-01-01'


def test_create_restart_keeps_cf_time_values_as_seconds(tmp_path):
    base_config = {
        'general': {'input_model': {'reference_date': '1970-01-01'}},
        'time': {'start': '2016-09-21 19:20:00', 'timestep': '60S', 'duration': '1D'},
        'particles': {
            'populations': [
                {
                    'name': 'population_1',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2016-09-21 19:20:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                }
            ]
        },
    }

    config_file = tmp_path / 'base.yaml'
    with open(config_file, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)

    ds = _trajectory_dataset(
        x=np.array([[0.0], [1.0]]),
        y=np.array([[0.0], [1.0]]),
        time=np.array([1474485600.0, 1474489200.0]),
        population_id=np.array([0], dtype=int),
        attrs={'reference_date': '1970-01-01', 'time_units': 'seconds since 1970-01-01'},
    )
    ds['time'].attrs['units'] = 'seconds since 1970-01-01'
    ds['time'].attrs['reference_date'] = '1970-01-01'
    netcdf_file = tmp_path / 'results.nc'
    ds.to_netcdf(netcdf_file)

    summary = create_restart_from_netcdf(
        netcdf_file=str(netcdf_file),
        base_config_file=str(config_file),
        output_config_file=str(tmp_path / 'restart.yaml'),
    )

    assert summary.restart_time == '2016-09-21 20:20:00'


def test_create_restart_validates_generated_time_against_original_forcing(tmp_path, monkeypatch):
    """A 1970 restart generated for 2016 forcing should fail before writing a bad config."""

    class FakeFormatConverter:
        def __init__(self, config):
            self.config = config

        def get_time_bounds(self):
            return 1474485600.0, 1474489200.0

    monkeypatch.setattr(restart_module, 'FormatConverter', FakeFormatConverter)

    base_config = {
        'general': {'input_model': {'format': 'fm_netcdf', 'reference_date': '1970-01-01'}},
        'inputs': {'data': 'forcing.nc'},
        'time': {'start': '2016-09-21 19:20:00', 'timestep': '60S', 'duration': '1D'},
        'particles': {
            'populations': [
                {
                    'name': 'population_1',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2016-09-21 19:20:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                }
            ]
        },
    }

    config_file = tmp_path / 'base.yaml'
    with open(config_file, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)

    ds = _trajectory_dataset(
        x=np.array([[0.0]]),
        y=np.array([[0.0]]),
        time=np.array([15400.0]),
        population_id=np.array([0], dtype=int),
    )
    netcdf_file = tmp_path / 'results.nc'
    ds.to_netcdf(netcdf_file)

    out_config = tmp_path / 'restart.yaml'
    with pytest.raises(ValueError, match='outside the original input forcing window'):
        create_restart_from_netcdf(
            netcdf_file=str(netcdf_file),
            base_config_file=str(config_file),
            output_config_file=str(out_config),
        )

    assert not out_config.exists()


def test_create_restart_with_valid_forcing_preserves_original_reference_date(tmp_path, monkeypatch):
    """Valid restart generation should keep the base input-model reference date unchanged."""

    class FakeFormatConverter:
        def __init__(self, config):
            self.config = config

        def get_time_bounds(self):
            return 1474485600.0, 1474489200.0

    monkeypatch.setattr(restart_module, 'FormatConverter', FakeFormatConverter)

    base_config = {
        'general': {'input_model': {'format': 'fm_netcdf', 'reference_date': '1970-01-01'}},
        'inputs': {'data': 'forcing.nc'},
        'time': {'start': '2016-09-21 19:20:00', 'timestep': '60S', 'duration': '1D'},
        'particles': {
            'populations': [
                {
                    'name': 'population_1',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2016-09-21 19:20:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                }
            ]
        },
    }

    config_file = tmp_path / 'base.yaml'
    with open(config_file, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)

    ds = _trajectory_dataset(
        x=np.array([[0.0], [1.0]]),
        y=np.array([[0.0], [1.0]]),
        time=np.array([1474485600.0, 1474486828.0]),
        population_id=np.array([0], dtype=int),
    )
    netcdf_file = tmp_path / 'results.nc'
    ds.to_netcdf(netcdf_file)

    out_config = tmp_path / 'restart.yaml'
    summary = create_restart_from_netcdf(
        netcdf_file=str(netcdf_file),
        base_config_file=str(config_file),
        output_config_file=str(out_config),
    )

    assert summary.restart_time == '2016-09-21 19:40:28'

    with open(out_config, 'r', encoding='utf-8') as handle:
        restart_cfg = yaml.safe_load(handle)
    assert restart_cfg['time']['start'] == '2016-09-21 19:40:28'
    assert restart_cfg['general']['input_model']['reference_date'] == '1970-01-01'


def test_create_restart_prefers_output_time_units_reference_date(tmp_path, monkeypatch):
    """Future output metadata can define the time axis reference date explicitly."""

    class FakeFormatConverter:
        def __init__(self, config):
            self.config = config

        def get_time_bounds(self):
            return 1474485600.0, 1474489200.0

    monkeypatch.setattr(restart_module, 'FormatConverter', FakeFormatConverter)

    base_config = {
        'general': {'input_model': {'format': 'fm_netcdf', 'reference_date': '1970-01-01'}},
        'inputs': {'data': 'forcing.nc'},
        'time': {'start': '2016-09-21 19:20:00', 'timestep': '60S', 'duration': '1D'},
        'particles': {
            'populations': [
                {
                    'name': 'population_1',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2016-09-21 19:20:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                }
            ]
        },
    }

    config_file = tmp_path / 'base.yaml'
    with open(config_file, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)

    ds = _trajectory_dataset(
        x=np.array([[0.0], [1.0]]),
        y=np.array([[0.0], [1.0]]),
        time=np.array([0.0, 1228.0]),
        population_id=np.array([0], dtype=int),
        attrs={'time_units': 'seconds since 2016-09-21 19:20:00'},
    )
    netcdf_file = tmp_path / 'results.nc'
    ds.to_netcdf(netcdf_file)

    summary = create_restart_from_netcdf(
        netcdf_file=str(netcdf_file),
        base_config_file=str(config_file),
        output_config_file=str(tmp_path / 'restart.yaml'),
    )

    assert summary.restart_time == '2016-09-21 19:40:28'


def test_create_restart_writes_seed_paths_relative_to_cwd(tmp_path, monkeypatch):
    examples_dir = tmp_path / 'examples'
    examples_dir.mkdir(parents=True)

    base_config = {
        'time': {'start': '2020-01-01 00:00:00', 'timestep': '60S', 'duration': '1D'},
        'particles': {
            'populations': [
                {
                    'name': 'populaton_1',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2020-01-01 00:00:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                }
            ]
        },
    }

    config_file = examples_dir / 'base.yaml'
    with open(config_file, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)

    ds = _trajectory_dataset(
        x=np.array([[0.0], [1.0]]),
        y=np.array([[0.0], [1.0]]),
        time=np.array([0.0, 60.0]),
        population_id=np.array([0], dtype=int),
    )
    netcdf_file = examples_dir / 'results.nc'
    ds.to_netcdf(netcdf_file)

    monkeypatch.chdir(tmp_path)
    out_config = examples_dir / 'restart.yaml'

    create_restart_from_netcdf(
        netcdf_file=str(netcdf_file),
        base_config_file=str(config_file),
        output_config_file=str(out_config),
    )

    with open(out_config, 'r', encoding='utf-8') as handle:
        restart_cfg = yaml.safe_load(handle)

    seed_path = restart_cfg['particles']['populations'][0]['seeding']['strategy']['file_points']['path']
    assert seed_path == './examples/restart_seeds/populaton_1.restart_points.csv'


def test_create_restart_raises_if_no_remaining_duration(tmp_path):
    base_config = {
        'time': {'start': '2020-01-01 00:00:00', 'timestep': '60S', 'duration': '1D'},
        'particles': {
            'populations': [
                {
                    'name': 'population_1',
                    'particle_type': 'sand',
                    'seeding': {
                        'release_start': '2020-01-01 00:00:00',
                        'quantity': 1,
                        'strategy': {'random': {'bbox': '0,0 1,1', 'nlocations': 1}},
                    },
                }
            ]
        },
    }

    config_file = tmp_path / 'base.yaml'
    with open(config_file, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)

    ds = _trajectory_dataset(
        x=np.array([[0.0], [1.0]]),
        y=np.array([[0.0], [1.0]]),
        # Last available time is exactly at end of original duration.
        time=np.array([0.0, 86400.0]),
        population_id=np.array([0], dtype=int),
    )
    netcdf_file = tmp_path / 'results.nc'
    ds.to_netcdf(netcdf_file)

    with pytest.raises(ValueError, match='No remaining duration'):
        create_restart_from_netcdf(
            netcdf_file=str(netcdf_file),
            base_config_file=str(config_file),
            output_config_file=str(tmp_path / 'restart.yaml'),
        )
def test_create_restart_streams_final_state_chunks_without_materializing_restart_state(tmp_path, monkeypatch):
    """Public restart generation reads final particle fields in bounded chunks."""
    base_config = {
        'time': {'start': '2020-01-01 00:00:00', 'timestep': '60S', 'duration': '1D'},
        'particles': {
            'populations': [
                {'name': 'population_1', 'seeding': {'quantity': 1}},
                {'name': 'population_2', 'seeding': {'quantity': 1}},
            ]
        },
    }
    config_file = tmp_path / 'base.yaml'
    with open(config_file, 'w', encoding='utf-8') as handle:
        yaml.safe_dump(base_config, handle, sort_keys=False)

    ds = _trajectory_dataset(
        x=np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [10.0, 11.0, np.nan, 13.0, 14.0]]),
        y=np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [20.0, 21.0, 22.0, 23.0, 24.0]]),
        time=np.array([0.0, 60.0]),
        population_id=np.array([0, 0, 0, 1, 1], dtype=int),
    )
    netcdf_file = tmp_path / 'results.nc'
    ds.to_netcdf(netcdf_file)

    monkeypatch.setattr(restart_module, 'RESTART_SEED_WRITE_CHUNK_SIZE', 2)
    monkeypatch.setattr(
        restart_module,
        '_extract_restart_state',
        lambda _: (_ for _ in ()).throw(AssertionError('public restart must stream final-state chunks')),
    )
    chunk_sizes = []
    original_stream = restart_module._stream_restart_chunks

    def recording_stream(dataset, context):
        for x_values, y_values, pop_ids, keep_mask in original_stream(dataset, context):
            chunk_sizes.append(x_values.size)
            yield x_values, y_values, pop_ids, keep_mask

    monkeypatch.setattr(restart_module, '_stream_restart_chunks', recording_stream)
    summary = create_restart_from_netcdf(
        netcdf_file=str(netcdf_file),
        base_config_file=str(config_file),
        output_config_file=str(tmp_path / 'restart.yaml'),
    )

    assert max(chunk_sizes) <= 2
    assert chunk_sizes == [2, 2, 1]
    assert summary.retained_particles == 4
    assert (tmp_path / 'restart_seeds' / 'population_1.restart_points.csv').read_text(encoding='utf-8').splitlines() == [
        'x,y',
        '10.00000000,20.00000000',
        '11.00000000,21.00000000',
    ]
    assert (tmp_path / 'restart_seeds' / 'population_2.restart_points.csv').read_text(encoding='utf-8').splitlines() == [
        'x,y',
        '13.00000000,23.00000000',
        '14.00000000,24.00000000',
    ]