# Reference states (open shell)

Open-shell references are configuration state functions (CSFs) generated with
the unitary group approach (UGA).

## Single CSF

```python
from occsfrd.reference import UGA

ref = UGA.singleCSF_ref(
    nActiveOrbs=2,
    GelfandTsetlinCSF=(S, M, dvec),
    prefactor=1.0,
    activeOrbitalIndices_=None,
)
# ref["reference"]  -> operator.OperatorSum creating the CSF from the vacuum
# ref["indices"]    -> list of SpecificOrbitalIndex for the active orbitals
```

- `S`: total spin.
- `M`: projection of total spin.
- `dvec`: list of orbital occupation patterns (0/1/2/3) for the active orbitals.

## Multiple CSFs

```python
ref = UGA.multipleCSF_ref(
    nActiveOrbs=2,
    GelfandTsetlinCSFlist=[(S1, M1, dvec1, c1), (S2, M2, dvec2, c2)],
)
```

Each tuple is `(S, M, dvec, prefactor)`.

## Clebsch-Gordan helper

`UGA.CGCoeff(Sold, Mold, s, m, S, M)` computes a general Clebsch-Gordan
coefficient used internally when coupling orbitals into CSFs.
