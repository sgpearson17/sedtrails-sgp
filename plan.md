# Multi-Population Tracer Runtime Plan

## Summary

- Implement mixed population tracer methods by making physics runtime state population-scoped instead of global.
- Support exactly one tracer method per population in this implementation; different populations may use different methods.
- Avoid plugin field collisions by copying each population/method plan's required physics fields into a plan-local `SedtrailsData` clone immediately after conversion.
- Archive this plan in `plan.md` before feature code changes, then execute in committed blocks on the active branch.

## Key Changes

- Add an internal runtime planning helper, preferably `src/sedtrails/simulation_orchestrator/runtime_plan.py`, with frozen dataclasses for:
  - `TracerRuntimePlan`: method name, method config, flow field names, transport probability method, required physics fields, converter.
  - `PopulationRuntimePlan`: population index/config/object plus its one tracer plan.
- Refactor `Simulation` so it no longer stores one global `PhysicsConverter`, global `tracer_methods`, or first-population `flow_field_names`.
- Build runtime plans after seeding, preserving population order and deriving each `PhysicsConfig` from global constants plus that population's characteristics and method config.
- Fix `PhysicsConfig.from_dict()` so method-specific nested config is selected from the resolved `PhysicsConfig.tracer_method`, not from `getattr(config, ...)` on a possible dict.
- Update the simulation loop:
  - On each new input chunk, run each population plan's converter.
  - Copy only required physics fields into a plan-local shallow `SedtrailsData` clone.
  - Use one `FieldDataRetriever` per population plan.
  - Compute CFL from every population plan's configured flow fields.
  - Update each population using its own method, flow fields, mixing depth, and probability fields.
- Preserve current movement behavior: if a method lists multiple flow fields, the population is updated once per listed flow field per timestep.
- Preserve current method behavior:
  - `vanwesten`: use `mixing_layer_thickness`, derived probability fields, and `update_burial_depth()`.
  - `soulsby`: use configured `grain_velocity`, unit transport probability, and skip van Westen burial-depth update.
- Update population schema validation to set `tracer_methods.maxProperties = 1` and require `flow_field_name` for supported method configs.

## Execution Blocks And Commits

- Block 0, baseline pending fixes:
  - Commit existing SciPy dependency and Windows plugin discovery fixes.
  - Commit message: `fix: declare scipy and harden physics plugin discovery`.
  - Verify: `mamba run -n sedtrails python -m pytest -q tests/transport_converter/plugins/test_physics_plugin.py`.
- Block 1, archive plan:
  - Create `plan.md` containing this implementation plan.
  - Commit message: `docs: archive multi-population tracer plan`.
  - Verify: `git diff --check`.
- Block 2, runtime planning model:
  - Add runtime plan dataclasses and builder tests.
  - Add validation for unknown methods, missing flow fields, and multiple methods per population.
  - Commit message: `refactor: add population tracer runtime plan`.
  - Verify:
    - `mamba run -n sedtrails python -m ruff check src/sedtrails/simulation_orchestrator tests/simulation_orchestrator`
    - `mamba run -n sedtrails python -m pytest -q tests/simulation_orchestrator`.
- Block 3, converter/config isolation:
  - Fix `PhysicsConfig.from_dict()`.
  - Ensure each runtime plan owns its own `PhysicsConverter`.
  - Add unit tests proving Soulsby and van Westen nested configs are flattened correctly and do not leak between converter instances.
  - Commit message: `fix: isolate physics converter config per tracer method`.
  - Verify:
    - `mamba run -n sedtrails python -m ruff check src/sedtrails/transport_converter tests/transport_converter`
    - `mamba run -n sedtrails python -m pytest -q tests/transport_converter`.
- Block 4, simulation loop refactor:
  - Replace first/last-population loop state with `PopulationRuntimePlan`.
  - Add plan-local physics field clone/cache helper.
  - Route CFL, scalar retrieval, flow retrieval, status updates, and dashboard data through the correct population plan retriever.
  - Commit message: `fix: run tracer physics per population`.
  - Verify:
    - `mamba run -n sedtrails python -m ruff check src/sedtrails/simulation_orchestrator src/sedtrails/particle_tracer tests/simulation_orchestrator`
    - `mamba run -n sedtrails python -m pytest -q tests/simulation_orchestrator`.
- Block 5, regression coverage and final checks:
  - Add regression tests for two populations using different tracer methods and for two populations using the same method with different configs.
  - Add schema validation tests for single-method enforcement.
  - Run full suite and final Ruff check.
  - Commit message: `test: cover mixed population tracer methods`.
  - Verify:
    - `mamba run -n sedtrails python -m ruff check .`
    - `mamba run -n sedtrails python -m pytest -q`.

## Test Cases

- Runtime plan builder:
  - one `vanwesten` population creates one converter with van Westen config.
  - one `soulsby` population creates one converter with Soulsby config.
  - mixed `vanwesten` and `soulsby` populations preserve each population's method and flow fields.
  - multiple methods in one population raises `ConfigurationError`.
  - unknown method or missing `flow_field_name` raises `ConfigurationError`.
- Converter:
  - nested Soulsby config applies `tracer_grain_size`, `background_grain_size`, and Soulsby parameters.
  - separate converter instances do not share plugin or grain-property cache.
- Simulation manager:
  - mixed populations call the correct converter with each population's `transport_probability`.
  - CFL uses all population plan flow fields, not only the first population.
  - update loop uses each population's method and does not reuse last population tracer methods.
  - plan-local physics clones keep same-named fields from overwriting another population's retrieved fields.

## Assumptions

- Full multi-method per single population is out of scope for this implementation and will fail fast.
- Plugin public APIs stay unchanged; collision avoidance is handled in the simulation orchestrator.
- Existing YAML object shape for `tracer_methods` stays backward compatible for valid one-method-per-population configs.
- Existing pending dependency/test fixes should be committed before new feature work.
