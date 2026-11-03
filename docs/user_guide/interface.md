# Interface utilities

## PySCF bridge

`occsfrd.interface.pyscfoperator` converts between PySCF wavefunctions and
OCCSFrD operators, and builds matrix representations:

- `pyscf_to_vec`, `vec_to_pyscf`
- `apply_OperatorProduct`, `apply_OperatorSum`
- `operator_to_matrix`

## Storing and loading equations

```python
from occsfrd.interface import storeequations

storeequations.save(name, equations, tensors, specificIndices=[])
loaded = storeequations.load(name)   # loads name.pkl
```

`load` registers compatibility shims so that pickles created under the old
`openshellcc` name remain loadable.

## Zenodo download

```python
eqn = storeequations.downloadFromZenodo("N0S0_qCCSD")
```

Downloads the equation file from the Zenodo sandbox deposition and loads it.

## LaTeX output

```python
from occsfrd.interface import texify

texify(equations, "output")  # writes output.tex
```
