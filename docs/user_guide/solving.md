# Solving coupled-cluster equations

The solver lives in `occsfrd.solve.cc`.

## Main entry point

```python
result = occsfrd.solve.cc.runUnlinkedCC(
    mf,
    equationsDict,
    levelShift=0,
    verbosity=0,
    biorthogonal=False,
    Rtol=10,
    Etol=8,
    maxIter=100,
    nDIIS=0,
    maxOrder=2,
    onlyConnect=False,
)
```

Returns a dictionary containing, among others:

- `"total electronic energy"`: correlation energy relative to the reference.
- `"amplitudes"`: converged amplitude tensors.
- `"iterations"`: number of iterations performed.

The total energy is `mf.e_tot + result["total electronic energy"]`.

## Building integrals / frozen core

UNDER CONSTRUCTION

`occsfrd.integrals.fock.buildFock` builds the one- and two-electron integrals
from a PySCF mean field and injects them into an equations dictionary. It
supports a frozen-core approximation via the `frozenCore` argument.

## DIIS

`occsfrd.solve.diis` provides DIIS amplitude update routines used internally by
the solver. `nDIIS` controls how many previous error vectors are retained.

## Reference energy

`occsfrd.solve.cc.getReferenceEnergy` evaluates the reference (intermediate
normalisation) energy without iterating amplitudes.
