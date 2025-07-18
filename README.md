[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TUDelft DCC](https://img.shields.io/badge/tu_delft-DCC-black?style=flat&label=TU%20Delft&labelColor=%23000000%20&color=%2300A6D6)](https://dcc.tudelft.nl)
[![Deploy Sphinx Documentation](https://github.com/sedtrails/sedtrails/actions/workflows/publish.yml/badge.svg)](https://github.com/sedtrails/sedtrails/actions/workflows/publish.yml)
[![Ruff](https://github.com/sedtrails/sedtrails/actions/workflows/ruff.yml/badge.svg?branch=dev)](https://github.com/sedtrails/sedtrails/actions/workflows/ruff.yml)
[![Pytest](https://github.com/sedtrails/sedtrails/actions/workflows/pytest.yml/badge.svg?branch=dev&event=push)](https://github.com/sedtrails/sedtrails/actions/workflows/pytest.yml)

# SedTRAILS
**Sed**iment **TRA**nsport v**I**sualization and **L**agrangian **S**imulator.

## Installation

SedTRAILS requires Python 3.12 and above, but it is likely to work with Python 3.10 and above.

1. Clone the repository:
```bash

git clone https://github.com/sedtrails/sedtrails.git
```

2. Change the directory:
```bash
cd sedtrails
```
3. Install the dependencies:
```bash
pip install .
```

## Software Architecture
The `diagrams/c4` directory was obtained from the [C4-PlantUML GitHub repository](https://github.com/plantuml-stdlib/C4-PlantUML). The directory contains some of the files required to run PlantUML, as well as a short [README.md](https://github.com/sedtrails/sedtrails/blob/main/diagrams/c4/README.md) that explains how to use the library.

**Requirements** 
- Java
- GraphViz

You can read more about the PlantUML syntax on the [PlantUML documentation](https://plantuml.com/command-line). They provide extensive support, which may be an overkill at the moment.
- If you have the requirements already installed, you can simply clone the repository and get started. 
- TL;DR: the basic command to save your diagram to a PNG file: 
    ```shell
    # Bash terminal
    plantuml -tpng <file.puml>
    ```

## Contributing Guidelines

Read the [contributing guidelines](CONTRIBUTING.md) to know how can you take part in this project. 

## License

SetTRAILS is licensed under the [MIT License](LICENSE).

&copy; (2025) SedTRAILS Team, Delft, The Netherlands. 

## Citation

Pannozzo, N., Shafiei, H., van Westen, B., Pearson, S. G., Reyns, J., & Aguilera Chaves, M. SedTRAILS (Version 0.1) [Computer software]

### Author Contributions:
| [Role](https://credit.niso.org/contributor-roles-defined/) | Author |
|------|--------|
| Conceptualization |  |
| Funding acquisition | |
| Project management |  |
| Research |  |
| Software |  |
| Supervision |  |

## Acknowlegdements

> The development of *SedTRAILS Software* was supported by the [Digital Competence Centre](https://dcc.tudelft.nl/), Delft University of Technology. 
