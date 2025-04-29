# Component Diagrams

## Component Diagram For the Simulation Configuration Interface

### User Interaction

#### Modeler
- **Role**: End user who aims to track sand particles in a coastal system.
- **Input**: Provides simulation configurations via:
  - Terminal commands (CLI)
  - YAML configuration files

---

### Architectural Layers

#### Simulation Configuration Interface (container)
This container abstracts the workflow needed to prepare simulation configurations for SedTRAILS. It contains the following components:

##### Simulation CLI (component)
- **Technology**: `Python`, `Typer`
- **Purpose**: Presents the commands to the user via a Command Line Interface.
- **Responsibilities**:
  - Collects user input for simulation setup.
  - Passes configuration data and instructions to the **Configuration Controller**.

##### Configuration Controller (component)
- **Technology**: `Python`
- **Purpose**: Central logic handler for configuration management.
- **Responsibilities**:
  - Reads and interprets YAML configuration files.
  - Applies default configuration values when needed.
  - Distributes complete configuration data to downstream components.

##### Validator (component)
- **Technology**: `YAML`
- **Purpose**: Validates input configuration files against the SedTRAILS schema.
- **Interaction**: Operates on file content provided by the **Configuration Controller**.

---

### External Components

These components receive configuration data from the Simulation Configuration Interface and are responsible for simulation execution and data transformation.

##### Lagrangian Particle Tracer (component)
- **Technology**: `Python`
- **Description**: Core simulation engine.
- **Function**: Computes particle positions \((x, y, t)\) in a coastal system using provided configuration.
- **Input**: Receives validated configurations from the **Configuration Controller**.

##### Transport Converter (component)
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
   - Distributes validated configurations to **Lagrangian Particle Tracer** and **Transport Converter**.
4. **Lagrangian Particle Tracer** and **Transport Converter** execute tasks based on configurations.
