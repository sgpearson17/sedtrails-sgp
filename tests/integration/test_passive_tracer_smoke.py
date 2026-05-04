from pathlib import Path

import yaml

from sedtrails.application_interfaces.api import run_simulation


def test_passive_tracer_smoke(tmp_path):
    """Run a tiny passive-tracer simulation as an integration smoke test."""
    config_path = Path('examples/config.example_soulsby.yaml')
    sample_data = Path('sample-data/inlet_sedtrails.nc')

    if not config_path.exists() or not sample_data.exists():
        raise AssertionError('Bundled example config or sample data is missing')

    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    config['inputs']['data'] = str(sample_data)
    config['outputs']['directory'] = str(tmp_path / 'results')
    config['visualization']['dashboard']['enable'] = False

    config['particles']['populations'] = config['particles']['populations'][:1]
    population = config['particles']['populations'][0]
    population['particle_type'] = 'passive'
    population['tracer_methods'] = {
        'passive_tracer': {
            'flow_field_name': ['depth_avg_flow_velocity'],
        }
    }
    population['transport_probability'] = 'no_probability'
    population['seeding']['strategy']['random']['nlocations'] = 1
    population['seeding']['quantity'] = 1

    config['time']['duration'] = '60S'
    config['time']['timestep'] = '60S'
    config['inputs']['read_interval'] = '60S'

    config_file = tmp_path / 'smoke_config.yml'
    config_file.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')

    output_dir = run_simulation(str(config_file), verbose=False, enable_dashboard=False)

    assert Path(output_dir).exists()
