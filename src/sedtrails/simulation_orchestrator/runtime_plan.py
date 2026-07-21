"""Runtime planning for population-specific tracer physics."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from sedtrails.exceptions.exceptions import ConfigurationError
from sedtrails.transport_converter.physics_converter import PhysicsConfig, PhysicsConverter


DEFAULT_PASSIVE_TRACER_FLOW_FIELDS = ('depth_avg_flow_velocity',)
SUPPORTED_TRACER_METHODS = frozenset({'passive_tracer', 'soulsby', 'vanwesten'})
DEFAULT_TRANSPORT_PROBABILITY_METHOD = 'no_probability'
_BASE_RUNTIME_INPUT_FIELDS = ('bed_level',)
_METHOD_RUNTIME_INPUT_FIELDS = {
    'passive_tracer': ('depth_avg_flow_velocity',),
    'soulsby': (
        'depth_avg_flow_velocity',
        'mean_bed_shear_stress',
        'max_bed_shear_stress',
    ),
    'vanwesten': (
        'depth_avg_flow_velocity',
        'mean_bed_shear_stress',
        'max_bed_shear_stress',
        'bed_load_transport',
        'suspended_transport',
    ),
}
_METHOD_TRANSIENT_ARRAY_COUNTS = {
    'passive_tracer': 0,
    'soulsby': 40,
    'vanwesten': 32,
}
_DERIVED_MEMORY_SAFETY_FACTOR = 1.25
_SOURCE_FIELD_ARRAY_COUNTS = {
    'bed_level': 1,
    'depth_avg_flow_velocity': 3,
    'bed_load_transport': 3,
    'suspended_transport': 3,
    'water_depth': 1,
    'mean_bed_shear_stress': 1,
    'max_bed_shear_stress': 1,
    'sediment_concentration': 1,
    'nonlinear_wave_velocity': 3,
}
_MIN_SOURCE_ITEMSIZE = np.dtype(np.int8).itemsize


@dataclass(frozen=True)
class TracerRuntimePlan:
    """Runtime state for one population tracer method."""

    method_name: str
    method_config: Mapping[str, Any]
    flow_field_names: tuple[str, ...]
    transport_probability_method: str
    required_physics_fields: tuple[str, ...]
    converter: PhysicsConverter


@dataclass(frozen=True)
class PopulationRuntimePlan:
    """Runtime state for one particle population."""

    population_index: int
    population_config: Mapping[str, Any]
    population: Any
    tracer: TracerRuntimePlan


def build_population_runtime_plans(
    population_configs: Sequence[Mapping[str, Any]],
    populations: Sequence[Any],
    base_physics_config: PhysicsConfig | Mapping[str, Any],
) -> tuple[PopulationRuntimePlan, ...]:
    """
    Build population-scoped tracer runtime plans.

    Parameters
    ----------
    population_configs : Sequence[Mapping[str, Any]]
        Population configuration mappings.
    populations : Sequence[Any]
        Particle populations to process.
    base_physics_config : PhysicsConfig | Mapping[str, Any]
        Base physics configuration shared by populations.

    Returns
    -------
    tuple[PopulationRuntimePlan, ...]
        Tuple containing the computed values.
    """

    if len(population_configs) != len(populations):
        raise ConfigurationError(
            f'Population config count ({len(population_configs)}) does not match seeded population count '
            f'({len(populations)}).'
        )

    return tuple(
        _build_population_runtime_plan(index, population_config, population, base_physics_config)
        for index, (population_config, population) in enumerate(zip(population_configs, populations, strict=True))
    )


def validate_population_runtime_configurations(
    population_configs: Sequence[Mapping[str, Any]],
) -> None:
    """Validate config-only runtime constraints before particle seeding.

    Parameters
    ----------
    population_configs : Sequence[Mapping[str, Any]]
        Population configuration mappings to validate.

    Returns
    -------
    None
        The function raises ConfigurationError when validation fails.
    """
    for population_index, population_config in enumerate(population_configs):
        tracer_methods = population_config.get('tracer_methods')
        if not isinstance(tracer_methods, Mapping) or len(tracer_methods) != 1:
            continue
        if 'passive_tracer' not in tracer_methods:
            continue

        transport_probability_method = population_config.get(
            'transport_probability', DEFAULT_TRANSPORT_PROBABILITY_METHOD
        )
        _validate_passive_tracer_configuration(
            population_index,
            population_config,
            transport_probability_method,
        )


def unique_flow_field_names(runtime_plans: Sequence[PopulationRuntimePlan]) -> list[str]:
    """
    Return configured flow field names across plans, preserving first-seen order.

    Parameters
    ----------
    runtime_plans : Sequence[PopulationRuntimePlan]
        Population runtime plans to inspect.

    Returns
    -------
    list[str]
        String result of the conversion.
    """

    return _unique_preserving_order(
        flow_field_name
        for runtime_plan in runtime_plans
        for flow_field_name in runtime_plan.tracer.flow_field_names
    )


def required_input_fields(
    runtime_plans: Sequence[PopulationRuntimePlan],
) -> tuple[str, ...]:
    """Return Eulerian source fields needed by the active runtime plans.

    Parameters
    ----------
    runtime_plans : Sequence[PopulationRuntimePlan]
        Population runtime plans to inspect.

    Returns
    -------
    tuple[str, ...]
        Unique SedTRAILS source field names in stable order.
    """
    fields = list(_BASE_RUNTIME_INPUT_FIELDS)
    for runtime_plan in runtime_plans:
        tracer_plan = runtime_plan.tracer
        fields.extend(_METHOD_RUNTIME_INPUT_FIELDS[tracer_plan.method_name])
        fields.extend(tracer_plan.flow_field_names)
        if (
            tracer_plan.method_name == 'vanwesten'
            and getattr(
                tracer_plan.converter.config,
                'suspended_velocity_method',
                'soulsby_2011',
            )
            == 'macdonald_2006'
        ):
            fields.append('water_depth')
    return tuple(_unique_preserving_order(fields))


def build_plan_sedtrails_data(
    sedtrails_data: Any,
    tracer_plan: TracerRuntimePlan,
    population_config: Mapping[str, Any] | None = None,
    default_fraction_index: int = 0,
    default_fraction_name: str | None = None,
) -> Any:
    """
    Run plan physics and return a clone containing only plan-required physics fields.

    Parameters
    ----------
    sedtrails_data : Any
        SedTRAILS data object to process.
    tracer_plan : TracerRuntimePlan
        Runtime plan for the tracer population.
    population_config : Mapping[str, Any], optional
        Configuration for the population. Its sediment fraction selection takes
        precedence over the input-model defaults.
    default_fraction_index : int, default 0
        Input-model fallback sediment fraction index.
    default_fraction_name : str, optional
        Input-model fallback sediment fraction name. When provided, it takes
        precedence over the fallback index.

    Returns
    -------
    Any
        Requested value.
    """

    fraction_selected_data = _select_population_fraction_data(
        sedtrails_data,
        population_config=population_config,
        default_fraction_index=default_fraction_index,
        default_fraction_name=default_fraction_name,
    )

    working_data = _shallow_sedtrails_data_clone(fraction_selected_data)
    tracer_plan.converter.convert_physics(
        sedtrails_data=working_data,
        transport_probability_method=tracer_plan.transport_probability_method,
    )

    plan_data = _shallow_sedtrails_data_clone(fraction_selected_data)
    plan_data.forcing_identity = plan_forcing_identity(
        tracer_plan,
        population_config,
        default_fraction_index,
        default_fraction_name,
    )
    for field_name in tracer_plan.required_physics_fields:
        if working_data.has_physics_field(field_name):
            plan_data.add_physics_field(
                field_name,
                _share_immutable_physics_value(getattr(working_data, field_name)),
            )
    return plan_data


def _build_population_runtime_plan(
    population_index: int,
    population_config: Mapping[str, Any],
    population: Any,
    base_physics_config: PhysicsConfig | Mapping[str, Any],
) -> PopulationRuntimePlan:
    tracer_methods = population_config.get('tracer_methods')
    if not isinstance(tracer_methods, Mapping) or not tracer_methods:
        raise ConfigurationError(f'Population {population_index} must define exactly one tracer method.')

    if len(tracer_methods) != 1:
        raise ConfigurationError(
            f'Population {population_index} defines multiple tracer methods '
            f'({", ".join(tracer_methods.keys())}); only one method per population is supported.'
        )

    method_name, method_config = next(iter(tracer_methods.items()))
    if method_name not in SUPPORTED_TRACER_METHODS:
        raise ConfigurationError(
            f'Population {population_index} uses unsupported tracer method {method_name!r}. '
            f'Supported methods: {", ".join(sorted(SUPPORTED_TRACER_METHODS))}.'
        )

    if not isinstance(method_config, Mapping):
        raise ConfigurationError(f'Population {population_index} tracer method {method_name!r} must be a mapping.')

    flow_field_names = _get_flow_field_names(population_index, method_name, method_config)
    transport_probability_method = population_config.get(
        'transport_probability', DEFAULT_TRANSPORT_PROBABILITY_METHOD
    )
    if method_name == 'passive_tracer':
        _validate_passive_tracer_configuration(
            population_index,
            population_config,
            transport_probability_method,
        )
    physics_config = build_physics_config(base_physics_config, population_config, method_name, method_config)
    tracer_config = {method_name: dict(method_config)}
    converter = PhysicsConverter(physics_config, tracer_config)

    return PopulationRuntimePlan(
        population_index=population_index,
        population_config=population_config,
        population=population,
        tracer=TracerRuntimePlan(
            method_name=method_name,
            method_config=method_config,
            flow_field_names=flow_field_names,
            transport_probability_method=transport_probability_method,
            required_physics_fields=required_physics_fields(method_name, flow_field_names),
            converter=converter,
        ),
    )


def build_physics_config(
    base_physics_config: PhysicsConfig | Mapping[str, Any],
    population_config: Mapping[str, Any],
    method_name: str,
    method_config: Mapping[str, Any],
) -> PhysicsConfig:
    """
    Build method-specific physics config for one population.

    Parameters
    ----------
    base_physics_config : PhysicsConfig | Mapping[str, Any]
        Base physics configuration shared by populations.
    population_config : Mapping[str, Any]
        Configuration for a single particle population.
    method_name : str
        Name of the tracer or physics method.
    method_config : Mapping[str, Any]
        Configuration mapping for the selected method.

    Returns
    -------
    PhysicsConfig
        Constructed physics configuration.
    """

    base_config = _physics_config_to_dict(base_physics_config)
    base_config['tracer_method'] = method_name

    characteristics = population_config.get('characteristics', {})
    if isinstance(characteristics, Mapping):
        if 'density' in characteristics:
            base_config['particle_density'] = characteristics['density']
        if 'grain_size' in characteristics:
            base_config['grain_diameter'] = characteristics['grain_size']
        elif 'size' in characteristics:
            base_config['grain_diameter'] = characteristics['size']

    return PhysicsConfig.from_dict(config=base_config, tracer_config={method_name: dict(method_config)})


def required_physics_fields(method_name: str, flow_field_names: Sequence[str]) -> tuple[str, ...]:
    """
    Return physics fields that must be preserved for a method plan.

    Parameters
    ----------
    method_name : str
        Name of the tracer or physics method.
    flow_field_names : Sequence[str]
        Flow-field names required by the runtime plans.

    Returns
    -------
    tuple[str, ...]
        Tuple containing the computed values.
    """

    if method_name == 'vanwesten':
        return tuple(
            _unique_preserving_order(
                (
                    *flow_field_names,
                    'mixing_layer_thickness',
                    *(flow_field_name.replace('velocity', 'probability') for flow_field_name in flow_field_names),
                )
            )
        )

    if method_name == 'soulsby':
        return tuple(_unique_preserving_order((*flow_field_names, 'mixing_layer_thickness', 'soulsby_a', 'soulsby_b')))

    if method_name == 'passive_tracer':
        return tuple(_unique_preserving_order(flow_field_names))

    raise ConfigurationError(f'Unsupported tracer method {method_name!r}.')


def _get_flow_field_names(
    population_index: int, method_name: str, method_config: Mapping[str, Any]
) -> tuple[str, ...]:
    flow_field_names = method_config.get('flow_field_name')
    if method_name == 'passive_tracer' and flow_field_names is None:
        return DEFAULT_PASSIVE_TRACER_FLOW_FIELDS

    if not isinstance(flow_field_names, Sequence) or isinstance(flow_field_names, str) or not flow_field_names:
        raise ConfigurationError(
            f'Population {population_index} tracer method {method_name!r} must define a non-empty '
            '`flow_field_name` list.'
        )

    if not all(isinstance(flow_field_name, str) and flow_field_name for flow_field_name in flow_field_names):
        raise ConfigurationError(
            f'Population {population_index} tracer method {method_name!r} has invalid flow field names.'
        )

    return tuple(flow_field_names)


def _validate_passive_tracer_configuration(
    population_index: int,
    population_config: Mapping[str, Any],
    transport_probability_method: str,
) -> None:
    particle_type = population_config.get('particle_type')
    if particle_type != 'passive':
        raise ConfigurationError(
            f'Population {population_index} uses tracer method "passive_tracer" but particle_type is '
            f'{particle_type!r}. Set particle_type to "passive".'
        )

    seeding = population_config.get('seeding', {})
    if isinstance(seeding, Mapping) and 'burial_depth' in seeding:
        raise ConfigurationError(
            f'Population {population_index} uses tracer method "passive_tracer" but defines '
            'seeding.burial_depth. Passive tracer does not support burial depth; remove '
            'seeding.burial_depth from this population.'
        )

    if transport_probability_method != DEFAULT_TRANSPORT_PROBABILITY_METHOD:
        raise ConfigurationError(
            f'Population {population_index} uses tracer method "passive_tracer" with '
            f'transport_probability={transport_probability_method!r}. Only "no_probability" is allowed '
            'for passive_tracer.'
        )


def _physics_config_to_dict(config: PhysicsConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(config, PhysicsConfig):
        return asdict(config)
    if is_dataclass(config):
        return asdict(config)
    return dict(config)


def _unique_preserving_order(values: Sequence[str] | Any) -> list[str]:
    unique_values = []
    seen = set()
    for value in values:
        if value not in seen:
            unique_values.append(value)
            seen.add(value)
    return unique_values


def _shallow_sedtrails_data_clone(sedtrails_data: Any) -> Any:
    cloned_data = copy.copy(sedtrails_data)
    cloned_data._physics_fields = {}
    return cloned_data


def _share_immutable_physics_value(value: Any) -> Any:
    """Transfer physics values without duplicating full numeric arrays."""
    if isinstance(value, dict):
        return {
            key: _share_immutable_physics_value(item)
            for key, item in value.items()
        }
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
        return value
    return value


def plan_forcing_identity(
    tracer_plan: TracerRuntimePlan,
    population_config: Mapping[str, Any] | None,
    default_fraction_index: int,
    default_fraction_name: str | None,
) -> tuple:
    """Return a stable identity for equivalent population forcing plans.

    Parameters
    ----------
    tracer_plan : TracerRuntimePlan
        Runtime physics and flow-field configuration.
    population_config : Mapping[str, Any] or None
        Population configuration containing an optional fraction selection.
    default_fraction_index : int
        Fallback sediment fraction index.
    default_fraction_name : str or None
        Fallback sediment fraction name.

    Returns
    -------
    tuple
        Hashable identity shared only by equivalent forcing plans.
    """
    fraction_selection = _resolve_fraction_selection(
        population_config,
        default_fraction_index=default_fraction_index,
        default_fraction_name=default_fraction_name,
    )
    return (
        'runtime-plan',
        tracer_plan.method_name,
        _freeze_identity_value(tracer_plan.method_config),
        tuple(tracer_plan.flow_field_names),
        tracer_plan.transport_probability_method,
        tuple(tracer_plan.required_physics_fields),
        _freeze_identity_value(getattr(tracer_plan.converter, 'config', None)),
        _freeze_identity_value(fraction_selection),
    )


def estimate_runtime_physics_reserve_bytes(
    runtime_plans: Sequence[PopulationRuntimePlan],
    spatial_size: int,
    *,
    time_planes: int = 2,
    default_fraction_index: int = 0,
    default_fraction_name: str | None = None,
    geodetic_runtime: bool = False,
) -> int:
    """Estimate peak bytes retained by unique runtime physics plans.

    Parameters
    ----------
    runtime_plans : Sequence[PopulationRuntimePlan]
        Active population runtime plans.
    spatial_size : int
        Number of Eulerian locations in one field plane.
    time_planes : int, default=2
        Number of temporal interpolation planes retained concurrently.
    default_fraction_index : int, default=0
        Fallback sediment fraction index.
    default_fraction_name : str, optional
        Fallback sediment fraction name.
    geodetic_runtime : bool, default=False
        Include prepared ECEF vector fields used by geodetic integration.

    Returns
    -------
    int
        Conservative derived-field and conversion-transient reserve in bytes.

    Notes
    -----
    Source fields are budgeted by the format reader. This estimate covers
    derived arrays retained by distinct plan identities plus the largest
    method-specific conversion workspace, with a 25 percent safety margin.
    """
    spatial_count = max(0, int(spatial_size))
    plane_count = max(1, int(time_planes))
    if spatial_count == 0 or not runtime_plans:
        return 0

    unique_plans = {}
    for runtime_plan in runtime_plans:
        identity = plan_forcing_identity(
            runtime_plan.tracer,
            runtime_plan.population_config,
            default_fraction_index,
            default_fraction_name,
        )
        unique_plans.setdefault(identity, runtime_plan)

    retained_arrays = 0
    peak_arrays = 0
    geodetic_vector_arrays = 0
    for runtime_plan in unique_plans.values():
        tracer_plan = runtime_plan.tracer
        method_name = tracer_plan.method_name
        retained_for_plan = sum(
            3 if field_name.endswith('velocity') else 1
            for field_name in tracer_plan.required_physics_fields
            if not (
                method_name == 'passive_tracer'
                and field_name == 'depth_avg_flow_velocity'
            )
        )
        transient_arrays = _METHOD_TRANSIENT_ARRAY_COUNTS[method_name]
        peak_arrays = max(
            peak_arrays,
            retained_arrays + retained_for_plan + transient_arrays,
        )
        retained_arrays += retained_for_plan
        if geodetic_runtime:
            geodetic_vector_arrays += 3 * len(set(tracer_plan.flow_field_names))

    peak_arrays = max(
        peak_arrays,
        retained_arrays + geodetic_vector_arrays,
    )
    raw_bytes = (
        peak_arrays
        * plane_count
        * spatial_count
        * np.dtype(np.float64).itemsize
    )
    return int(np.ceil(raw_bytes * _DERIVED_MEMORY_SAFETY_FACTOR))


def split_eulerian_memory_budget(
    total_memory_bytes: int,
    runtime_plans: Sequence[PopulationRuntimePlan],
    spatial_size: int,
    *,
    default_fraction_index: int = 0,
    default_fraction_name: str | None = None,
    source_bytes_per_plane: int | None = None,
    geodetic_runtime: bool = False,
) -> tuple[int, int]:
    """Split total Eulerian memory between source and derived runtime fields.

    Parameters
    ----------
    total_memory_bytes : int
        Complete configured Eulerian memory budget.
    runtime_plans : Sequence[PopulationRuntimePlan]
        Active population runtime plans.
    spatial_size : int
        Number of Eulerian locations in one field plane.
    default_fraction_index : int, default=0
        Fallback sediment fraction index.
    default_fraction_name : str, optional
        Fallback sediment fraction name.
    source_bytes_per_plane : int, optional
        Exact selected source bytes per plane reported by the active plugin.
        The generic one-byte itemsize lower bound is used when unavailable.
    geodetic_runtime : bool, default=False
        Include prepared ECEF vector fields used by geodetic integration.

    Returns
    -------
    tuple[int, int]
        Source-reader budget and derived-runtime reserve in bytes.

    Raises
    ------
    MemoryError
        If derived runtime fields alone consume the complete budget.
    """
    total_bytes = int(total_memory_bytes)
    derived_bytes_per_plane = estimate_runtime_physics_reserve_bytes(
        runtime_plans,
        spatial_size,
        time_planes=1,
        default_fraction_index=default_fraction_index,
        default_fraction_name=default_fraction_name,
        geodetic_runtime=geodetic_runtime,
    )
    source_array_count = sum(
        _SOURCE_FIELD_ARRAY_COUNTS.get(field_name, 0)
        for field_name in required_input_fields(runtime_plans)
    )
    if source_bytes_per_plane is None or int(source_bytes_per_plane) <= 0:
        source_bytes_per_plane = (
            max(0, int(spatial_size))
            * source_array_count
            * _MIN_SOURCE_ITEMSIZE
        )
    else:
        source_bytes_per_plane = int(source_bytes_per_plane)
    if derived_bytes_per_plane == 0 or source_bytes_per_plane == 0:
        return total_bytes, 0

    total_bytes_per_plane = source_bytes_per_plane + derived_bytes_per_plane
    source_bytes = int(
        total_bytes * source_bytes_per_plane // total_bytes_per_plane
    )
    reserve_bytes = total_bytes - source_bytes
    if source_bytes <= 0:
        raise MemoryError(
            'inputs.max_eulerian_memory_mb cannot reserve a positive source '
            'forcing budget after accounting for the per-plane derived '
            f'runtime working set ({derived_bytes_per_plane} bytes/plane). '
            'Increase the memory budget or reduce grid size or distinct '
            'tracer configurations.'
        )
    return source_bytes, reserve_bytes


def _freeze_identity_value(value: Any) -> Any:
    """Convert configuration values to a deterministic hashable identity."""
    if is_dataclass(value):
        return _freeze_identity_value(asdict(value))
    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (str(key), _freeze_identity_value(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, np.ndarray):
        return (
            'ndarray',
            value.dtype.str,
            value.shape,
            np.ascontiguousarray(value).tobytes(),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(_freeze_identity_value(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _select_population_fraction_data(
    sedtrails_data: Any,
    population_config: Mapping[str, Any] | None,
    default_fraction_index: int,
    default_fraction_name: str | None,
) -> Any:
    """Return a fraction-selected clone when multi-fraction data is available."""
    fractions = int(getattr(sedtrails_data, 'fractions', 1) or 1)
    if fractions <= 1:
        return sedtrails_data

    selected_fraction_index, selected_fraction_name = _resolve_fraction_selection(
        population_config,
        default_fraction_index=default_fraction_index,
        default_fraction_name=default_fraction_name,
    )

    if selected_fraction_name:
        available_labels = _available_fraction_labels(sedtrails_data)
        if not available_labels:
            raise ConfigurationError(
                f"Configured sediment_fraction_name '{selected_fraction_name}' could not be resolved because "
                f'fraction labels are not available in input metadata. Detected {fractions} sediment fractions. '
                'Set sediment_fraction_index explicitly for this population, for example:\n'
                'particles:\n'
                '  populations:\n'
                '    - name: your_population_name\n'
                '      sediment_fraction_index: 0'
            )
        else:
            normalized_labels = [str(label).strip().lower() for label in available_labels]
            requested_name = str(selected_fraction_name).strip().lower()
            if requested_name not in normalized_labels:
                raise ConfigurationError(
                    f"Configured sediment_fraction_name '{selected_fraction_name}' was not found. "
                    f'Available NAMCON labels: {available_labels}'
                )
            selected_fraction_index = normalized_labels.index(requested_name)

    try:
        selected_fraction_index = int(selected_fraction_index)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f'Configured sediment_fraction_index must be an integer, got {selected_fraction_index!r}'
        ) from exc

    if selected_fraction_index < 0 or selected_fraction_index >= fractions:
        raise ConfigurationError(
            f'Configured sediment_fraction_index={selected_fraction_index} is out of bounds for '
            f'available fractions={fractions}.'
        )

    selected_data = _shallow_sedtrails_data_clone(sedtrails_data)
    selected_data.fractions = 1

    for field_name, value in list(vars(selected_data).items()):
        if field_name.startswith('_') or field_name == 'fractions':
            continue
        selected_value = _select_fraction_value(value, fractions, selected_fraction_index)
        if selected_value is not value:
            setattr(selected_data, field_name, selected_value)

    return selected_data


def _resolve_fraction_selection(
    population_config: Mapping[str, Any] | None,
    *,
    default_fraction_index: int,
    default_fraction_name: str | None,
) -> tuple[Any, str | None]:
    """Return one population selection, falling back to the global selection."""
    if not isinstance(population_config, Mapping):
        return default_fraction_index, default_fraction_name

    population_fraction_name = population_config.get('sediment_fraction_name')
    if population_fraction_name:
        return population_config.get('sediment_fraction_index', 0), str(population_fraction_name)

    if 'sediment_fraction_index' in population_config:
        return population_config.get('sediment_fraction_index'), None

    return default_fraction_index, default_fraction_name


def _select_fraction_value(value: Any, fractions: int, fraction_index: int) -> Any:
    """Select a single fraction from arrays or vector-field dictionaries when present."""
    if isinstance(value, dict) and {'x', 'y', 'magnitude'}.issubset(value.keys()):
        selected_value = dict(value)
        selection_applied = False
        for component_name in ('x', 'y', 'magnitude'):
            component = np.asarray(value[component_name])
            if component.ndim >= 3 and component.shape[1] == fractions:
                selected_value[component_name] = component[:, fraction_index, ...]
                selection_applied = True
        return selected_value if selection_applied else value

    if isinstance(value, np.ndarray) and value.ndim >= 3 and value.shape[1] == fractions:
        return value[:, fraction_index, ...]

    return value


def _available_fraction_labels(sedtrails_data: Any) -> list[str] | None:
    """Return normalized sediment fraction labels from data metadata when available."""
    if not hasattr(sedtrails_data, 'metadata'):
        return None
    metadata = sedtrails_data.metadata

    labels = None
    if hasattr(metadata, 'get'):
        labels = metadata.get('sediment_fraction_labels', None)
    elif hasattr(metadata, 'sediment_fraction_labels'):
        labels = metadata.sediment_fraction_labels

    if labels is None:
        return None
    return [str(label).strip() for label in labels]
