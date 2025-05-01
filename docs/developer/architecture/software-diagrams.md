# Component Diagrams

## Component Diagram For the Simulation Configuration Interface

### User Interaction

#### Modeler
- **Role**: End user who aims to track sand particles in a coastal system.
- **Function**: Provides simulation configurations via:
  - Terminal commands (CLI)
  - YAML configuration files

---

### Simulation Configuration Interface (container)
This container abstracts the workflow needed to prepare simulation configurations for SedTRAILS. It contains the following components:

#### Simulation CLI (component)
- **Technology**: `Python`, `Typer`
- **Purpose**: Presents the commands to the user via a Command Line Interface.
- **Function**:
  - Collects user input for simulation setup.
  - Passes configuration data and instructions to the **Configuration Controller**.

#### Configuration Controller (component)
- **Technology**: `Python`
- **Purpose**: Central logic handler for configuration (i.e., model input parameters) management.
- **Function**:
  - Reads and interprets YAML configuration files.
  - Applies default configuration values when needed.
  - Distributes complete configuration data to downstream components.

#### Validator (component)
- **Technology**: `YAML`
- **Purpose**: Validates input configuration files against the SedTRAILS schema.
- **Interaction**: Operates on file content provided by the **Configuration Controller**.

---

### External Components

These components receive configuration data from the Simulation Configuration Interface and are responsible for simulation execution and data transformation.

#### Lagrangian Particle Tracer (component)
- **Technology**: `Python`
- **Description**: Core simulation engine.
- **Function**: Computes particle positions \((x, y, t)\) in a coastal system using provided configuration.
- **Input**: Receives validated configurations from the **Configuration Controller**.

#### Transport Converter (component)
- **Technology**: `Python`
- **Description**: Data adapter for external flow-field datasets.
- **Function**:
  - Reads and converts 'transport flow–field data' into SedTRAILS-compatible formats and physical units.
  - Interfaces with the **Configuration Controller** to ensure compatibility with the rest of the pipeline.

---

### Data Flow Summary

1. **Modeler** initiates the simulation via CLI or configuration file.
2. **Simulation CLI** exposes commands and forwards user inputs.
3. **Configuration Controller**:
   - Reads and interprets configuration files.
   - Sends content to **Validator** for schema validation.
   - Distributes validated configurations (i.e., model parameter settings) to **Lagrangian Particle Tracer** and **Transport Converter**.
4. **Lagrangian Particle Tracer** and **Transport Converter** execute tasks based on configurations (i.e., model parameter settings).

## Component Diagram for Transport Converter

The **Transport Converter** is a core component of the SedTRAILS architecture that enables the use of external Eulerian flow modeling systems by transforming their output into SedTRAILS-compatible formats and physical units. It acts as an adapter layer that standardizes, configures, and converts transport flow–field data for use in sediment particle simulations.

---

### User Interaction

#### Modeler
- **Role**: End user who defines the simulation parameters and output configuration.
- **Function**: Provides a YAML configuration file that specifies source data and conversion preferences.

---

### Transport Converter (container)
This container provides the functionality for importing and converting external flow–field data. It includes the following components:

#### Format Converter Interface (component)
- **Technology**: `Python`
- **Purpose**: Reads flow–field data from external Eulerian modeling systems.
- **Function**:
  - Reads flow–field input (e.g., in `NetCDF` format).
  - Converts the custom format to the internal **SedTRAILS** standard data structure.
  - Provides data to the **Physics Converter**.

#### Physics Converter (component)
- **Technology**: `Python`
- **Purpose**: Applies physical transformations to flow–field data.
- **Function**:
  - Accesses user-specified conversion methods from the **Simulation Configuration Interface**.
  - Converts flow-field physical units using methods available in the **Physics Conversion Library**.
  - Provides converted transport flow-field data to **Lagrangian Particle Tracer** and **Data Management Module**.

#### Physics Conversion Library (component)
- **Technology**: `Python`
- **Purpose**: Provides a suite of conversion methods.
- **Function**:
  - Defines reusable functions for physical unit conversion.
  - Provides conversion methods to the **Physics Converter**.

---

### Connected Components

These components work alongside or consume output from the Transport Converter:

#### Simulation Configuration Interface (component)
- **Technology**: `Python`
- **Purpose**: Reads the user’s configuration file and provides setup parameters.
- **Function**: Provides configuration details to the **Physics Converter**.

#### Lagrangian Particle Tracer (component)
- **Technology**: `Python`
- **Purpose**: Executes the core simulation to compute particle trajectories.
- **Input**: Converted flow–field data for particle transport simulation from the **Physics Converter**.

#### Data Management Module (component)
- **Technology**: `Python`
- **Purpose**: Handles data and file I/O.
- **Function**:
  - Receives converted flow–field data.
  - Writes data to disk using the local file system in formats like `NetCDF`.

#### SedTRAILS Logger (component)
- **Technology**: `Python`
- **Purpose**: Logs simulation states, intermediate outputs, errors and performance metrics.
- **Function**: Tracks execution start/end times of routines used in the **Physics Converter**.

---

### External Systems

#### Eulerian Flow Modeling System (external system)
- **Technology**: `[Any]`
- **Description**: External system that simulates water/wind dynamics and outputs results in a custom structure.
- **Interaction**: Source of the flow–field input data to the **Format Converter Interface**.

#### Local File System (external system)
- **Technology**: `Windows`, `Linux`, `MacOS`
- **Purpose**: Stores converted flow–field data.
- **Interaction**: Used by the **Data Management Module** for reading/writing data in `NetCDF` or other supported formats.

---

### Data Flow Summary

1. **Modeler** defines simulation and conversion preferences via a YAML file.
2. **Simulation Configuration Interface** parses configuration and passes relevant parameters to the **Physics Converter**.
3. **Format Converter Interface** reads flow–field input data from an external Eulerian model (e.g., in NetCDF).
4. **Physics Converter**:
   - Receives the standard-formatted data.
   - Applies physical conversions using methods from the **Physics Conversion Library**.
   - Produces SedTRAILS-compatible transport flow–field data.
5. Converted data is used by:
   - **Lagrangian Particle Tracer** for simulation,
   - **Data Management Module** for storage,
   - **SedTRAILS Logger** for tracking computation metrics.
