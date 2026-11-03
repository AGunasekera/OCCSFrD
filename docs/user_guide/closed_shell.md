# Closed-shell coupled cluster

The closed-shell module is `occsfrd.ansatz.closedshellcc`.

## Amplitude and projection tensors

```python
from occsfrd import wick
from occsfrd.ansatz import closedshellcc

t2 = closedshellcc.amplitude(2)            # t^{ab}_{ij}
phi2 = closedshellcc.projectionManifold(2) # Phi^{ij}_{ab}
```

## Generating an ansatz

```python
excitationLevels = [1, 2]   # CCSD
BCH, fockTensor, h2Tensor, amplitudes = closedshellcc.genClosedShellCCAnsatz(
    excitationLevels, trunc=4
)
```

`genClosedShellCCAnsatz` returns the BCH-expanded similarity-transformed
Hamiltonian plus the fock and two-electron tensors and the amplitude tensors.

## Deriving equations

```python
energyEq = closedshellcc.getEnergyEquation(BCH)
ampEq = closedshellcc.getAmplitudeEquation(BCH, excitationLevel=2)
```

`getEnergyEquationNew` / `getAmplitudeEquationNew` use the updated contraction
routine and return un-collected forms suitable for the unlinked (multi-reference)
solver.

## Saving equations

```python
from occsfrd.interface import storeequations

storeequations.save(
    "CCSD",
    [energyEq, ampEq],
    [fockTensor, h2Tensor] + amplitudes,
)
```
