# Installation

OCCSFrD (Open-shell Coupled Cluster using Spin-Free Diagrammatic construction) is a
Python package for deriving and implementing spin-adapted, spin-free single- and
multi-reference coupled cluster methods for open-shell molecular electronic structure.

## Requirements

- Python 3.11 or newer
- A working Fortran compiler (e.g. `gfortran`) when building from source
- Runtime dependencies: `numpy`, `pyscf`, `networkx`, `requests`

## Install from PyPI

OCCSFrD is hosted on the Python Package Index, whence it can be installed using pip:

```bash
python -m pip install occsfrd
```

## Install from source (development)

```bash
git clone https://github.com/AGunasekera/OCCSFrD.git
cd OCCSFrD
python -m pip install --upgrade pip
python -m pip install meson meson-python numpy
python -m pip install -e .
```

The editable install invokes the Meson backend, which compiles the Fortran
extension (`occsfrd.wick.contract.contractEXT`) with `f2py`.

## Verify the installation

```python
import occsfrd
print(occsfrd.__version__)
```

## Documentation

The full documentation is hosted on Read the Docs at `occsfrd.readthedocs.io`.
