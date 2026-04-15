# OCCSFrD
Open-shell Coupled Cluster using Spin-Free Diagrammatic construction (OCCSFrD): A Python package to derive and implement spin-adapted and spin-free single- and multi-reference coupled cluster methods for open shell molecular electronic structure

## Installation
OCCSFrD is currently hosted on the test instance of the Python Package Index, and will be migrated to the production instance when the stable version 1 is released.
Current version 0 can be installed:
```bash
$ python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple occsfrd
```

## Usage
Occsfrd as a python API
```python
import occsfrd
```
Documentation for the OCCSFrD API is available at occsfrd.readthedocs.io

TODO:
entry points to invoke separate scripts from the command line for generating equations and for solving
interfacing with integrals from external package
interfacing with zenodo for equation hosting

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`occsfrd` was created by Alexander Gunasekera. It is licensed under the terms of the MIT license.

## Credits

`occsfrd` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
