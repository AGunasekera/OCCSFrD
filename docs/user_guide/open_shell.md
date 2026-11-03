# Open-shell (normal-ordered) coupled cluster

The open-shell module is `occsfrd.ansatz.normalorderedcc`.

## Amplitude tensors

Amplitude tensors for open-shell methods are usually built as `wick.tensor.Tensor`
objects with active orbital types (`a`) and specific indices for open-shell
orbitals:

```python
from occsfrd import wick
from occsfrd.reference import UGA

ref = UGA.singleCSF_ref(nActiveOrbs=2, GelfandTsetlinCSF=(1, 1, [1, 1]), prefactor=1.0)
activeIndices = ref["indices"]

t_aa = wick.tensor.Tensor("t", ["a", "a"], ["a", "a"])  # doubles among active
```

## Generating an ansatz

```python
from occsfrd.ansatz import normalorderedcc

transformedH, fockTensor, h2Tensor = normalorderedcc.genNormalOrderedCCAnsatz(
    [t_aa], order=2
)
```

## Deriving equations

```python
energyEq = normalorderedcc.getEnergyEquation(transformedH, referenceOperator=ref["reference"])
ampEq = normalorderedcc.getAmplitudeEquation(transformedH, t_aa, referenceOperator=ref["reference"])
```

For the unlinked formalism used by the production solver, use
`getEnergyEquationUnlinked`, `getAmplitudeEquation_UnlinkedFormalism`,
`getCollectedAmplitudeEquation_UnlinkedFormalism`, etc.

## Reference states

`UGA.singleCSF_ref` and `UGA.multipleCSF_ref` generate the operator that
creates a CSF reference out of the Fermi vacuum, together with the active
orbital indices. See the [reference states](reference_states.md) section.
