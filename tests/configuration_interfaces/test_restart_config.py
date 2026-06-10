from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from sedtrails.application_interfaces.restart import create_restart_from_netcdf


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

    ds = xr.Dataset(
        data_vars={
            'x': (
                ('n_particles', 'n_timesteps'),
                np.array(
                    [
                        [0.0, 1.0, 2.0],
                        [5.0, 6.0, np.nan],
                        [10.0, 11.0, 12.0],
                    ]
                ),
            ),
            'y': (
                ('n_particles', 'n_timesteps'),
                np.array(
                    [
                        [0.0, 0.5, 1.0],
                        [4.0, 4.5, np.nan],
                        [7.0, 7.5, 8.0],
                    ]
                ),
            ),
            'time': (
                ('n_particles', 'n_timesteps'),
                np.array(
                    [
                        [0.0, 60.0, 120.0],
                        [0.0, 60.0, np.nan],
                        [0.0, 60.0, 120.0],
                    ]
                ),
            ),
            'population_id': (('n_particles',), np.array([0, 0, 1], dtype=int)),
            'status_alive': (
                ('n_particles', 'n_timesteps'),
                np.array(
                    [
                        [1.0, 1.0, 1.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0],
                    ]
                ),
            ),
            'status_domain': (
                ('n_particles', 'n_timesteps'),
                np.ones((3, 3), dtype=float),
            ),
        }
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

    ds = xr.Dataset(
        data_vars={
            'x': (('n_particles', 'n_timesteps'), np.array([[0.0, 1.0]])),
            'y': (('n_particles', 'n_timesteps'), np.array([[0.0, 1.0]])),
            # 2016-09-21 19:40:28 UTC in epoch seconds
            'time': (('n_particles', 'n_timesteps'), np.array([[1474485628.0, 1474486828.0]])),
            'population_id': (('n_particles',), np.array([0], dtype=int)),
        }
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

    ds = xr.Dataset(
        data_vars={
            'x': (('n_particles', 'n_timesteps'), np.array([[0.0, 1.0]])),
            'y': (('n_particles', 'n_timesteps'), np.array([[0.0, 1.0]])),
            'time': (('n_particles', 'n_timesteps'), np.array([[0.0, 60.0]])),
            'population_id': (('n_particles',), np.array([0], dtype=int)),
        }
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

    ds = xr.Dataset(
        data_vars={
            'x': (('n_particles', 'n_timesteps'), np.array([[0.0, 1.0]])),
            'y': (('n_particles', 'n_timesteps'), np.array([[0.0, 1.0]])),
            # Last available time is exactly at end of original duration.
            'time': (('n_particles', 'n_timesteps'), np.array([[0.0, 86400.0]])),
            'population_id': (('n_particles',), np.array([0], dtype=int)),
        }
    )
    netcdf_file = tmp_path / 'results.nc'
    ds.to_netcdf(netcdf_file)

    with pytest.raises(ValueError, match='No remaining duration'):
        create_restart_from_netcdf(
            netcdf_file=str(netcdf_file),
            base_config_file=str(config_file),
            output_config_file=str(tmp_path / 'restart.yaml'),
        )
