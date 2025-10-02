
# Development Environment

Developers and contributors can set up a local development environment to work on the SedTRAILS codebase. Follow the steps below to get started.

1. Make sure Python (3.13 or newer) and a code editor (e.g., Visual Studio Code) is installed.

1. Fork the SedTRAILS repository on GitHub to your own account. If it is your first time using contritubing to SedTRAILS, please read the [contributing guidelines](contribution.md). 

1. Clone the SedTRAILS repository from github in your desired directory:
```bash
git clone git@github.com:<your-user-name>/sedtrails.git
```

1. Change the directory:
```bash
cd sedtrails
```

1. Change to the development branch:
```bash
git swiltch dev
```

1. Create and activate a virtual environment to manage dependencies using the tool of your choice. For example, using `venv` or `conda`. 

```bash
# If using venv
python -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`

# If using conda
conda create --name sedtrails-dev python=3.13
conda activate sedtrails-dev

```

1. Install the dependencies: ([more details](https://github.com/sedtrails/sedtrails/blob/dev/CONTRIBUTING.md))
```bash
pip install -e .[dev]
```

1. Confirming the installation  by typing on the terminal:
```bash
sedtrails -v

# The installed version should be displayed
```

For help:
```bash
sedtrails -h
```

## Simulation Test

To fullly test your development environment, you can run a sample simulation using a provided dataset.

1. Download the dataset file named `inlet_sedtrails.nc` from [this link](https://surfdrive.surf.nl/files/index.php/s/VUGKZm7QexAXuD9?path=%2Fdfm).

2. Change the directory:
```bash
cd examples
```

3. In the file `config.example.yaml`, update the directory for `input_data` to read the downloaded dataset: `input_data: ./<path-to/inlet_sedtrails.nc`


4. Run the model:
```bash
sedtrails run -c config.example.yaml
```
