# Installation

::: warning
Update this section
:::

 SedTRAILS is currently in an alpha stage. It is not recommended for production use.

This is the step-by-step guide on how to install SedTRAILS. Make sure Python (3.13 or newer) and a code editor (e.g., Visual Studio) is installed.


## Installation via PIP:

Simply type the following command in terminal:
```bash
pip install sedtrails
```

## Installation via Github repository:

1. Clone the SedTRAILS repository from github in your desired directory:
```bash
cd <your desired directory>
git clone https://github.com/sedtrails/sedtrails.git
```

2. Go to the directory in which the SedTRAILS repository is cloned into:
```bash
cd sedtrails
```

3. Install the dependencies: ([See this webpage](https://github.com/sedtrails/sedtrails/blob/dev/CONTRIBUTING.md) for more details about installing as a developer)
```bash
pip install .
```

## Confirming the installation

Writing the following command in the terminal will show the installed SedTRAILS version:
```bash
sedtrails -v
```

The following command provides help about SedTRAILS installation:
```bash
sedtrails -h

```