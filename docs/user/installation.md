# Installation

:::warning
SedTRAILS is currently on its **beta** version. Users are welcome to test it, but compatibility might break in future versions.  Its use in research requires discretion. We aim for an official release by February 2026.
:::

**Requirements:**

SedTRAILS is tested on Python 3.13 and above, but it is likely to work with Python 3.10 and above.


## Using pip 
You can install the latest beta version directly from PyPI using pip. Check the [releases page](https://pypi.org/project/sedtrails/#history) for the latest version (x).

```bash
pip install sedtrails=1.0.0-beta.x
```

## From source
To install SedTRAILS from source, follow these steps:

1. Clone the repository:
```bash

git clone https://github.com/sedtrails/sedtrails.git
```

2. Change to the `sedtrails` directory:
```bash
cd sedtrails
```

3. Install the package using `pip` or `conda`:

```bash
# For pip
pip install .
```

```bash
# For conda
# Create and activate the conda environment
conda env create -f environment.yml
conda activate sedtrails
```

## Confirming the installation

Writing the following command in the terminal will show the installed SedTRAILS version:
```bash
sedtrails -v
# E.g.: SedTRAILS 1.0.0-beta.0
```

The following command provides help about the SedTRAILS commands:
```bash
sedtrails -h

```

Check the [Simulations Guide](./simulations.md) to learn how to set up and run a simulation.

