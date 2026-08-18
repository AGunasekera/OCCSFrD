# Packaging and publishing

This document describes how to build the project for publication (sdist + wheels) and how to build/install locally for development. The project uses `meson-python` (PEP 517) via `pyproject.toml` and Meson (`meson.build`) to compile the Fortran extension with `f2py`.

## Prerequisites

- Python 3.11+
- pip
- For publication: `build`, `meson`, `meson-python`, `wheel`, `twine`, `numpy`
- For development: `meson`, `meson-python`, `numpy` and a Fortran compiler (e.g. `gfortran`)

Install common tools:

```bash
python -m pip install --upgrade pip
python -m pip install build meson meson-python wheel twine numpy
```

On Debian/Ubuntu you may need:

```bash
sudo apt-get update && sudo apt-get install -y build-essential gfortran python3-dev
```

## Build for publication (sdist + wheel)

This produces a source distribution and a wheel suitable for publishing on PyPI. The build will invoke Meson (via `meson-python`) to compile the Fortran extension.

1. Ensure package metadata in `pyproject.toml` and `meson.build` is correct (name, version, license, authors).
2. Build sdist and wheel:

```bash
python -m build
```

Artefacts are written to `dist/` (for example `dist/occsfrd-0.1.4.a.tar.gz` and a wheel file).

3. (Optional) Inspect the sdist contains Fortran sources and wrapper inputs:

```bash
tar -tf dist/*.tar.gz | grep -E "\.f90$|contract"
```

4. Upload with `twine` (use an API token in CI or env variable locally):

```bash
python -m pip install --upgrade twine
# set environment variable for CI or local: export PYPI_API_TOKEN='pypi-...'
python -m twine upload dist/*
```

Notes:
- Wheels are platform-specific. Use `cibuildwheel` in CI to produce wheels for Linux/macOS/Windows.
- Ensure `numpy` is installed on build machines so `f2py` include dirs are discoverable.

## Local development (editable install)

Use an editable install so Python imports use your working tree and Meson builds the extension on install.

1. Install runtime/build deps for development:

```bash
python -m pip install --upgrade pip
python -m pip install meson meson-python numpy
```

2. Install the package in editable mode:

```bash
python -m pip install -e .
```

This will invoke the Meson backend to build the Fortran extension and place the package in editable mode. If you change Fortran sources, re-run:

```bash
# rebuild with Meson
meson compile -C build
```

or reinstall the editable package to rebuild native artefacts:

```bash
pip uninstall -y occsfrd
pip install -e .
```

## CI publishing notes (GitHub Actions)

- The repo contains a workflow that runs `cibuildwheel` to build wheels across platforms and uploads artefacts. To automatically publish on tag releases, add a step that uses `PYPI_API_TOKEN` (a repository secret) and runs `twine upload` (or use `pypa/gh-action-pypi-publish`). Example publish step (after building wheels):

```yaml
- name: Publish to PyPI
  if: startsWith(github.ref, 'refs/tags/')
  env:
    TWINE_USERNAME: __token__
    TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
  run: |
    python -m pip install --upgrade twine
    python -m twine upload wheelhouse/*.whl
```

If CI images lack a Fortran compiler, install it in a pre-step (e.g. `apt-get install -y gfortran`) or use `CIBW_BEFORE_BUILD` to install extra OS packages.

## Troubleshooting

- Missing `numpy` at build time: install `numpy` before building so Meson/`f2py` can find `numpy.f2py.get_include()`.
- Missing Fortran compiler: install `gfortran` (or the appropriate compiler) on the build machine.
- If the sdist lacks Fortran sources, ensure `MANIFEST.in` includes `src/occsfrd/wick/contract/*.f90` and that `pyproject.toml` is present in the repo root.

## References

- Meson Python backend: https://meson-python.readthedocs.io/
- cibuildwheel: https://cibuildwheel.readthedocs.io/
- Packaging guide: https://packaging.python.org/
